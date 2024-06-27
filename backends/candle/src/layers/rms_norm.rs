use candle::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct RMSNorm {
    weight: Tensor,
    epsilon: f32,
    span: tracing::Span,
}

impl RMSNorm {
    pub fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get(hidden_size, "weight")
                .or_else(|_| vb.get(hidden_size, "gamma"))?,
            epsilon,
            span: tracing::span!(tracing::Level::TRACE, "rms-norm"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        match hidden_states.device() {
            Device::Cpu | Device::Metal(_) => {
                let mut hidden_states = hidden_states.clone();
                let residual_add = if let Some(residual) = residual {
                    let residual_add = hidden_states.add(residual)?;
                    hidden_states = residual_add.clone();
                    residual_add
                } else {
                    hidden_states.clone()
                };

                let hidden_states_dtype = hidden_states.dtype();
                let internal_dtype = match hidden_states_dtype {
                    DType::F16 | DType::BF16 => DType::F32,
                    d => d,
                };
                let hidden_size = hidden_states.dim(D::Minus1)?;
                let hidden_states = hidden_states.to_dtype(internal_dtype)?;
                let norm_hidden_states =
                    (hidden_states.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
                let hidden_states_normed = hidden_states
                    .broadcast_div(&(norm_hidden_states + self.epsilon as f64)?.sqrt()?)?;
                Ok((
                    hidden_states_normed
                        .to_dtype(hidden_states_dtype)?
                        .broadcast_mul(&self.weight)?,
                    residual_add,
                ))
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    use candle_layer_norm::{fused_add_rms_norm, rms_norm};

                    let original_shape = hidden_states.shape();
                    let hidden_states = hidden_states.flatten_to(D::Minus2)?;

                    if let Some(residual) = residual {
                        let residual = residual.flatten_to(D::Minus2)?;

                        let (result, residual_add) = fused_add_rms_norm(
                            &hidden_states,
                            &residual,
                            &self.weight,
                            None,
                            self.epsilon,
                        )?;
                        Ok((
                            result.reshape(original_shape)?,
                            residual_add.reshape(original_shape)?,
                        ))
                    } else {
                        let residual_add = hidden_states.clone();

                        let result = rms_norm(&hidden_states, &self.weight, None, self.epsilon)?;

                        Ok((
                            result.reshape(original_shape)?,
                            residual_add.reshape(original_shape)?,
                        ))
                    }
                }
                #[cfg(not(feature = "cuda"))]
                candle::bail!("`cuda` feature is not enabled")
            }
        }
    }
}
