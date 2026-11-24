use candle::{DType, Device, Result, Tensor, D};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    rope_type: String,
    factor: Option<f32>,
}

impl RopeParameters {
    /// Applies `linear` scaling to the inverse frequencies, the only scaled `rope_type` supported
    pub fn scale(&self, inv_freqs: Tensor) -> Result<Tensor> {
        match (self.rope_type.as_str(), self.factor) {
            ("default", _) => Ok(inv_freqs),
            ("linear", Some(factor)) => inv_freqs / factor as f64,
            ("linear", None) => candle::bail!("`rope_type` `linear` requires a `factor`"),
            (rope_type, _) => candle::bail!("`rope_type` `{rope_type}` is not supported"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum RopeScaling {
    Llama3 {
        #[serde(alias = "type")]
        rope_type: String,
        factor: f32,
        high_freq_factor: f32,
        low_freq_factor: f32,
        original_max_position_embeddings: usize,
    },
    Ntk {
        #[serde(alias = "type")]
        rope_type: String,
        factor: f32,
    },
}

pub fn get_inv_freqs(
    dim: usize,
    base: f32,
    device: &Device,
    rope_scaling: Option<&RopeScaling>,
) -> Result<Tensor> {
    let get_inv_freqs_inner = |dim: usize, base: f32, device: &Device| {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        Tensor::from_vec(inv_freq, (1, inv_freq_len), device)
    };

    if let Some(rope_scaling) = rope_scaling {
        match rope_scaling {
            RopeScaling::Llama3 {
                rope_type: _,
                factor,
                high_freq_factor,
                low_freq_factor,
                original_max_position_embeddings,
            } => {
                let old_context_len = *original_max_position_embeddings as f32;
                let low_freq_wavelen = old_context_len / low_freq_factor;
                let high_freq_wavelen = old_context_len / high_freq_factor;

                let inv_freq: Vec<_> = (0..dim)
                    .step_by(2)
                    .map(|i| {
                        let freq_idx = i as f32 / dim as f32;
                        // Compute base inverse frequency
                        let inv_freq_base = 1.0 / base.powf(freq_idx);

                        // Compute wavelength from inverse frequency
                        let wavelen = 2.0 * std::f32::consts::PI / inv_freq_base;

                        // Apply Llama3 scaling logic
                        if wavelen < high_freq_wavelen {
                            // High frequency: no scaling
                            inv_freq_base
                        } else if wavelen > low_freq_wavelen {
                            // Low frequency: scale by factor
                            inv_freq_base / factor
                        } else {
                            // Medium frequency: smooth interpolation
                            let smooth_factor = (old_context_len / wavelen - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                            let inv_freq_llama = inv_freq_base / factor;
                            (1.0 - smooth_factor) * inv_freq_llama + smooth_factor * inv_freq_base
                        }
                    })
                    .collect();
                let inv_freq_len = inv_freq.len();
                return Tensor::from_vec(inv_freq, (1, inv_freq_len), device);
            }
            RopeScaling::Ntk {
                rope_type: _,
                factor,
            } => {
                let inv_freqs = get_inv_freqs_inner(dim, base * factor, device)?;
                let s = factor.powf(2.0 / dim as f32) as f64;
                return inv_freqs / s;
            }
        }
    }
    get_inv_freqs_inner(dim, base, device)
}

pub fn get_cos_sin(
    length: usize,
    inv_freqs: &Tensor,
    dtype: DType,
    repeat_freqs: bool,
) -> Result<(Tensor, Tensor)> {
    let t = Tensor::arange(0u32, length as u32, inv_freqs.device())?
        .to_dtype(DType::F32)?
        .reshape((length, 1))?;
    let mut freqs = t.matmul(inv_freqs)?;
    if repeat_freqs {
        freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
    }

    let cos = freqs.cos()?.to_dtype(dtype)?;
    let sin = freqs.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

pub fn apply_rotary(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    attention_head_size: usize,
) -> Result<Tensor> {
    let dim = attention_head_size / 2;
    let x1 = x.narrow(D::Minus1, 0, dim)?;
    let x2 = x.narrow(D::Minus1, dim, dim)?;
    let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    let rope = (x.broadcast_mul(cos)? + rotate_x.broadcast_mul(sin)?)?;
    Ok(rope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_rope_parameters_match_ntk_rope_scaling() -> Result<()> {
        // `gte-base-en-v1.5` and `gte-multilingual-base`, before and after their native config
        for (theta, ntk_factor, rope_parameters) in [
            (
                500000.0,
                2.0,
                r#"{"factor": 1.0218971486541166, "rope_theta": 1000000.0, "rope_type": "linear"}"#,
            ),
            (
                20000.0,
                8.0,
                r#"{"factor": 1.0671404006768237, "rope_theta": 160000.0, "rope_type": "linear"}"#,
            ),
        ] {
            let ntk = RopeScaling::Ntk {
                rope_type: "ntk".to_string(),
                factor: ntk_factor,
            };
            let old = get_inv_freqs(64, theta, &Device::Cpu, Some(&ntk))?;
            let linear: RopeParameters = serde_json::from_str(rope_parameters).unwrap();
            let new = linear.scale(get_inv_freqs(64, linear.rope_theta, &Device::Cpu, None)?)?;

            let diff = (old - new)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(diff < 1e-7, "max abs diff {diff} for {rope_parameters}");
        }
        Ok(())
    }

    #[test]
    fn invalid_rope_parameters_are_rejected() {
        for rope_parameters in [
            r#"{"factor": 4.0, "rope_theta": 10000.0, "rope_type": "yarn"}"#,
            r#"{"rope_theta": 10000.0, "rope_type": "linear"}"#,
        ] {
            let params: RopeParameters = serde_json::from_str(rope_parameters).unwrap();
            let inv_freqs = get_inv_freqs(64, params.rope_theta, &Device::Cpu, None).unwrap();
            assert!(params.scale(inv_freqs).is_err(), "{rope_parameters}");
        }
    }
}
