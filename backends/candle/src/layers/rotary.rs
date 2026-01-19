use candle::{DType, Device, Result, Tensor, D};
use serde::Deserialize;

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
                // Llama 3 NTK-by-parts scaling with frequency-dependent factors
                let scale = factor / *original_max_position_embeddings as f32;
                let inv_freq: Vec<_> = (0..dim)
                    .step_by(2)
                    .enumerate()
                    .map(|(_idx, i)| {
                        let freq_idx = i as f32 / dim as f32;
                        // Compute wavelength relative to original context
                        let wavelength = 2.0 * std::f32::consts::PI * base.powf(freq_idx);
                        let original_context = *original_max_position_embeddings as f32;
                        
                        // Ramp function: transition from low to high freq scaling
                        let alpha = 1.0;
                        let beta = 32.0;
                        let r = wavelength / original_context;
                        
                        let gamma = if r < alpha {
                            0.0
                        } else if r > beta {
                            1.0
                        } else {
                            (r - alpha) / (beta - alpha)
                        };
                        
                        // Blend low and high frequency scaling
                        let scale_factor = (1.0 - gamma) * low_freq_factor + gamma * high_freq_factor;
                        1f32 / (base.powf(freq_idx) * scale_factor * scale)
                    })
                    .collect();
                let inv_freq_len = inv_freq.len();
                return Tensor::from_vec(inv_freq, (1, inv_freq_len), device);
            }
            RopeScaling::Ntk { rope_type: _, factor } => {
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
