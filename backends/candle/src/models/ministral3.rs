use crate::layers::{apply_rotary, get_cublas_lt_wrapper, HiddenAct, Linear, RMSNorm};
use crate::models::Model;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;

use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Ministral3RopeParameters {
    pub rope_theta: f32,
    pub rope_type: Option<String>,
    pub factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    pub attention_factor: Option<f32>,
    pub beta_fast: Option<f32>,
    pub beta_slow: Option<f32>,
    pub mscale: Option<f32>,
    pub mscale_all_dim: Option<f32>,
    pub truncate: Option<bool>,
    pub llama_4_scaling_beta: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Ministral3Config {
    pub vocab_size: usize,
    pub head_dim: Option<usize>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_parameters: Ministral3RopeParameters,
    pub sliding_window: Option<usize>,
    pub pad_token_id: Option<u32>,
    // NOTE: Transformers defaults to causal attention when `is_causal` is not set
    pub is_causal: Option<bool>,
}

/// Multimodal (Pixtral vision tower + Ministral3 language model) configuration, out of which only
/// the language model is used, as only text inputs are supported
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Mistral3Config {
    pub text_config: Ministral3Config,
}

/// Computes the RoPE inverse frequencies and the scaling factor to apply to the cos/sin, as in
/// https://github.com/huggingface/transformers/blob/v5.20.0/src/transformers/modeling_rope_utils.py
pub(crate) fn get_inv_freqs(dim: usize, config: &Ministral3Config) -> Result<(Vec<f32>, f32)> {
    let rope = &config.rope_parameters;
    let base = rope.rope_theta;

    let pos_freqs = (0..dim)
        .step_by(2)
        .map(|i| base.powf(i as f32 / dim as f32));

    match rope.rope_type.as_deref() {
        None | Some("default") => Ok((pos_freqs.map(|f| 1.0 / f).collect(), 1.0)),
        Some("yarn") => {
            let Some(original_max_position_embeddings) = rope.original_max_position_embeddings
            else {
                candle::bail!("`rope_parameters.original_max_position_embeddings` is required when `rope_type=yarn`")
            };
            let original_max_position_embeddings = original_max_position_embeddings as f64;

            let factor = match (rope.factor, config.max_position_embeddings) {
                (Some(factor), _) => factor,
                (None, Some(max_position_embeddings)) => {
                    (max_position_embeddings as f64 / original_max_position_embeddings) as f32
                }
                (None, None) => candle::bail!("Either `rope_parameters.factor` or `max_position_embeddings` is required when `rope_type=yarn`"),
            };

            let get_mscale = |scale: f32, mscale: f32| {
                if scale <= 1.0 {
                    1.0
                } else {
                    0.1 * mscale * scale.ln() + 1.0
                }
            };
            let attention_factor = match (rope.attention_factor, rope.mscale, rope.mscale_all_dim) {
                (Some(attention_factor), _, _) => attention_factor,
                (None, Some(mscale), Some(mscale_all_dim))
                    if mscale != 0.0 && mscale_all_dim != 0.0 =>
                {
                    get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
                }
                _ => get_mscale(factor, 1.0),
            };

            let beta_fast = rope.beta_fast.filter(|b| *b != 0.0).unwrap_or(32.0) as f64;
            let beta_slow = rope.beta_slow.filter(|b| *b != 0.0).unwrap_or(1.0) as f64;

            // Inverse dimension formula to find the dimension based on the number of rotations
            let find_correction_dim = |num_rotations: f64| {
                (dim as f64
                    * (original_max_position_embeddings
                        / (num_rotations * 2.0 * std::f64::consts::PI))
                        .ln())
                    / (2.0 * (base as f64).ln())
            };
            let mut low = find_correction_dim(beta_fast);
            let mut high = find_correction_dim(beta_slow);
            if rope.truncate.unwrap_or(true) {
                low = low.floor();
                high = high.ceil();
            }
            let low = low.max(0.0) as f32;
            let mut high = high.min(dim as f64 - 1.0) as f32;
            if low == high {
                high += 0.001; // Prevent singularity
            }

            // Interpolate (i.e., scale by `factor`) the low frequencies, extrapolate the high ones
            let inv_freqs = pos_freqs
                .enumerate()
                .map(|(i, pos_freq)| {
                    let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                    (1.0 / (factor * pos_freq)) * ramp + (1.0 / pos_freq) * (1.0 - ramp)
                })
                .collect();

            Ok((inv_freqs, attention_factor))
        }
        Some(rope_type) => candle::bail!("`rope_type={rope_type}` is not supported for Ministral3"),
    }
}

/// Returns the `(beta, original_max_position_embeddings)` of the Llama 4 attention scaling, if any
pub(crate) fn get_llama_4_scaling(config: &Ministral3Config) -> Result<Option<(f32, usize)>> {
    match (
        config.rope_parameters.llama_4_scaling_beta,
        config.rope_parameters.original_max_position_embeddings,
    ) {
        (Some(beta), Some(original_max_position_embeddings)) => {
            Ok(Some((beta, original_max_position_embeddings)))
        }
        (Some(_), None) => candle::bail!("`rope_parameters.original_max_position_embeddings` is required when `rope_parameters.llama_4_scaling_beta` is set"),
        (None, _) => Ok(None),
    }
}

/// Computes the Llama 4 scaling to apply to the queries given their positions i.e.,
/// `1 + beta * ln(1 + floor(position / original_max_position_embeddings))`. Returns `None` if all
/// the positions are within `original_max_position_embeddings`, as the scaling is 1 for those.
pub(crate) fn get_llama_4_query_scale(
    position_ids: &[u32],
    llama_4_scaling: Option<(f32, usize)>,
) -> Option<Vec<f32>> {
    let (beta, original_max_position_embeddings) = llama_4_scaling?;

    if position_ids
        .iter()
        .all(|&p| (p as usize) < original_max_position_embeddings)
    {
        return None;
    }

    Some(
        position_ids
            .iter()
            .map(|&p| {
                let interval = (p as usize / original_max_position_embeddings) as f32;
                1.0 + beta * (1.0 + interval).ln()
            })
            .collect(),
    )
}

struct Ministral3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    attention_head_size: usize,

    softmax_scale: f64,

    span: tracing::Span,
}

impl Ministral3Attention {
    pub fn load(vb: VarBuilder, config: &Ministral3Config) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let num_key_value_heads = config.num_key_value_heads;
        let hidden_size = config.hidden_size;

        let query_weight = vb.pp("q_proj").get(
            (num_attention_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let q_proj = Linear::new(query_weight, None, None);

        let key_weight = vb.pp("k_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let k_proj = Linear::new(key_weight, None, None);

        let value_weight = vb.pp("v_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let v_proj = Linear::new(value_weight, None, None);

        let o_proj_weight = vb.pp("o_proj").get(
            (hidden_size, num_attention_heads * attention_head_size),
            "weight",
        )?;
        let o_proj = Linear::new(o_proj_weight, None, None);

        let softmax_scale = 1.0 / (attention_head_size as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads,
            num_key_value_heads,
            attention_head_size,
            softmax_scale,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        query_scale: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let device = hidden_states.device();

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let input_dims = hidden_states.dims();
        let input_shape = &input_dims[..input_dims.len() - 1];

        // Reshape to [batch, seq_len, heads, head_dim]
        let q = q.reshape(
            [
                input_shape,
                &[self.num_attention_heads, self.attention_head_size],
            ]
            .concat(),
        )?;
        let k = k.reshape(
            [
                input_shape,
                &[self.num_key_value_heads, self.attention_head_size],
            ]
            .concat(),
        )?;
        let v = v.reshape(
            [
                input_shape,
                &[self.num_key_value_heads, self.attention_head_size],
            ]
            .concat(),
        )?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = apply_rotary(&q, cos, sin, self.attention_head_size)?;
        let k = apply_rotary(&k, cos, sin, self.attention_head_size)?;

        // Llama 4 position-dependent attention scaling, applied to the queries
        let q = match query_scale {
            Some(query_scale) => q.broadcast_mul(query_scale)?,
            None => q,
        };

        // For simplicity, expand k and v to match number of q heads if needed (GQA)
        let k = if self.num_key_value_heads != self.num_attention_heads {
            let repeat_factor = self.num_attention_heads / self.num_key_value_heads;
            let (b, h, s, d) = k.shape().dims4()?;
            let k = k.unsqueeze(2)?.expand((b, h, repeat_factor, s, d))?;
            k.reshape((b, h * repeat_factor, s, d))?
        } else {
            k
        };

        let v = if self.num_key_value_heads != self.num_attention_heads {
            let repeat_factor = self.num_attention_heads / self.num_key_value_heads;
            let (b, h, s, d) = v.shape().dims4()?;
            let v = v.unsqueeze(2)?.expand((b, h, repeat_factor, s, d))?;
            v.reshape((b, h * repeat_factor, s, d))?
        } else {
            v
        };

        #[allow(unused_variables)]
        let context_layer = if let (Device::Cuda(_), Some(cublaslt)) =
            (device, get_cublas_lt_wrapper())
        {
            #[cfg(feature = "cuda")]
            {
                let (batch_size, _, seq_len, _) = k.shape().dims4()?;
                let q = q.flatten(0, 1)?;
                let k = k.flatten(0, 1)?;
                let v = v.flatten(0, 1)?;
                // NOTE: cuBLASLt needs the bias to match the shape of the attention scores
                let attention_bias = attention_bias
                    .map(|bias| {
                        bias.broadcast_as((batch_size, self.num_attention_heads, seq_len, seq_len))?
                            .flatten(0, 1)?
                            .contiguous()
                    })
                    .transpose()?;

                let beta = match attention_bias.is_some() {
                    true => Some(1.0),
                    false => None,
                };

                let attention_scores = cublaslt.batch_matmul(
                    &k,
                    &q,
                    attention_bias.as_ref(),
                    Some(self.softmax_scale as f32),
                    beta,
                    None,
                    None,
                )?;
                let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

                let context_layer = cublaslt.batch_matmul(
                    &v.t()?.contiguous()?,
                    &attention_probs,
                    Some(&q),
                    None,
                    None,
                    None,
                    None,
                )?;

                context_layer.reshape((
                    batch_size,
                    self.num_attention_heads,
                    seq_len,
                    self.attention_head_size,
                ))
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle::bail!("`cuda` feature is not enabled")
            }
        } else {
            let attn_weights = q.matmul(&k.t()?)?;
            let mut attn_weights = (attn_weights * self.softmax_scale)?;

            if let Some(attention_bias) = attention_bias {
                attn_weights = attn_weights.broadcast_add(attention_bias)?;
            }

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v.contiguous()?)
        }?;

        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;

        self.o_proj.forward(&context_layer)
    }
}

struct Ministral3MLP {
    gate_up_proj: Linear,
    down_proj: Linear,

    activation: HiddenAct,
    intermediate_size: usize,

    span: tracing::Span,
}

impl Ministral3MLP {
    pub fn load(vb: VarBuilder, config: &Ministral3Config) -> Result<Self> {
        let intermediate_size = config.intermediate_size;

        let gate_proj_weight = vb
            .pp("gate_proj")
            .get((intermediate_size, config.hidden_size), "weight")?;

        let up_proj_weight = vb
            .pp("up_proj")
            .get((intermediate_size, config.hidden_size), "weight")?;

        let gate_up_proj_weight = Tensor::cat(&[&gate_proj_weight, &up_proj_weight], 0)?;
        let gate_up_proj = Linear::new(gate_up_proj_weight, None, None);

        let down_proj_weight = vb
            .pp("down_proj")
            .get((config.hidden_size, intermediate_size), "weight")?;
        let down_proj = Linear::new(down_proj_weight, None, None);

        Ok(Self {
            gate_up_proj,
            down_proj,
            activation: config.hidden_act.clone(),
            intermediate_size,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.gate_up_proj.forward(hidden_states)?;
        let gate_states = gate_up_states.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up_states =
            gate_up_states.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;

        let gate_states = self.activation.forward(&gate_states)?;
        self.down_proj.forward(&(gate_states * up_states)?)
    }
}

struct Ministral3Layer {
    attention: Ministral3Attention,
    mlp: Ministral3MLP,
    input_layer_norm: RMSNorm,
    post_attention_layer_norm: RMSNorm,

    span: tracing::Span,
}

impl Ministral3Layer {
    pub fn load(vb: VarBuilder, config: &Ministral3Config) -> Result<Self> {
        let attention = Ministral3Attention::load(vb.pp("self_attn"), config)?;
        let mlp = Ministral3MLP::load(vb.pp("mlp"), config)?;

        let input_layer_norm = RMSNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let post_attention_layer_norm = RMSNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            attention,
            mlp,
            input_layer_norm,
            post_attention_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        query_scale: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (normed_hidden_states, residual) =
            self.input_layer_norm.forward(hidden_states, None)?;

        let attn_output =
            self.attention
                .forward(&normed_hidden_states, attention_bias, cos, sin, query_scale)?;

        let (normed_attn_res_output, attn_res) = self
            .post_attention_layer_norm
            .forward(&attn_output, Some(&residual))?;

        let mlp_output = self.mlp.forward(&normed_attn_res_output)?;

        let output = (&mlp_output + &attn_res)?;

        Ok(output)
    }
}

pub struct Ministral3Model {
    embeddings: Embedding,
    layers: Vec<Ministral3Layer>,
    norm: RMSNorm,
    inv_freqs: Tensor,
    rotary_scale: f32,
    rotary_dim: usize,
    // NOTE: `(beta, original_max_position_embeddings)`
    llama_4_scaling: Option<(f32, usize)>,
    pool: Pool,
    pad_token_id: u32,

    dtype: DType,
    device: Device,

    span: tracing::Span,
}

impl Ministral3Model {
    pub fn load(vb: VarBuilder, config: &Ministral3Config, model_type: ModelType) -> Result<Self> {
        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Ministral3")
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("`splade` is not supported for Ministral3");
                }
                pool
            }
        };

        if config.is_causal.unwrap_or(true) {
            candle::bail!(
                "Ministral3 is only supported with bidirectional attention i.e., `is_causal=false`"
            );
        }

        if config.sliding_window.is_some() {
            candle::bail!("Sliding window is not supported for Ministral3");
        }

        // The Mistral3 (multimodal) models contain the `language_model` key
        // https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16
        let model_prefix = if vb.contains_tensor("language_model.embed_tokens.weight") {
            "language_model."
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            "model."
        } else {
            ""
        };

        let embeddings = Embedding::new(
            vb.pp(format!("{model_prefix}embed_tokens"))
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );

        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                Ministral3Layer::load(vb.pp(format!("{model_prefix}layers.{index}")), config)
            })
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(
            vb.pp(format!("{model_prefix}norm")),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        let rotary_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        // NOTE: The cos/sin are computed per batch out of the position ids rather than cached, as
        // caching those for `max_position_embeddings=262144` would require ~256MB
        let (inv_freqs, rotary_scale) = get_inv_freqs(rotary_dim, config)?;
        let inv_freqs = Tensor::from_vec(inv_freqs, (1, rotary_dim / 2), vb.device())?;

        let llama_4_scaling = get_llama_4_scaling(config)?;

        Ok(Self {
            embeddings,
            layers,
            norm,
            inv_freqs,
            rotary_scale,
            rotary_dim,
            llama_4_scaling,
            pool,
            pad_token_id: config.pad_token_id.unwrap_or(0),
            dtype: vb.dtype(),
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let shape = (batch_size, max_length);
        let elems = batch_size * max_length;

        let mut input_ids = Vec::with_capacity(elems);
        let mut position_ids = Vec::with_capacity(elems);
        let mut attention_bias = Vec::with_capacity(elems);
        let mut input_lengths = Vec::with_capacity(batch_size);
        let mut masking = false;

        for i in 0..batch_size {
            let start = batch.cumulative_seq_lengths[i] as usize;
            let end = batch.cumulative_seq_lengths[i + 1] as usize;
            let seq_length = end - start;
            input_lengths.push(seq_length);

            for j in start..end {
                input_ids.push(batch.input_ids[j]);
                position_ids.push(batch.position_ids[j]);
                attention_bias.push(0.0);
            }

            // Right padding
            let padding = max_length - seq_length;
            if padding > 0 {
                masking = true;
                for _ in 0..padding {
                    input_ids.push(self.pad_token_id);
                    position_ids.push(0);
                    attention_bias.push(f32::NEG_INFINITY);
                }
            }
        }

        let query_scale = get_llama_4_query_scale(&position_ids, self.llama_4_scaling)
            .map(|scale| {
                Tensor::from_vec(scale, (batch_size, 1, max_length, 1), &self.device)?
                    .to_dtype(self.dtype)
            })
            .transpose()?;

        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;

        // NOTE: The attention is bidirectional, so the bias is only needed to mask the padding
        let attention_bias = if masking {
            let attention_bias =
                Tensor::from_vec(attention_bias, (batch_size, 1, 1, max_length), &self.device)?
                    .to_dtype(self.dtype)?;
            Some(attention_bias)
        } else {
            None
        };

        let positions: Vec<f32> = position_ids.iter().map(|&p| p as f32).collect();
        let freqs = Tensor::from_vec(positions, (elems, 1), &self.device)?
            .broadcast_mul(&self.inv_freqs)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

        let (mut cos, mut sin) = (freqs.cos()?, freqs.sin()?);
        if self.rotary_scale != 1.0 {
            cos = (cos * self.rotary_scale as f64)?;
            sin = (sin * self.rotary_scale as f64)?;
        }

        let cos =
            cos.to_dtype(self.dtype)?
                .reshape((batch_size, 1, max_length, self.rotary_dim))?;
        let sin =
            sin.to_dtype(self.dtype)?
                .reshape((batch_size, 1, max_length, self.rotary_dim))?;

        let mut hidden_states = self.embeddings.forward(&input_ids)?;

        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                attention_bias.as_ref(),
                &cos,
                &sin,
                query_scale.as_ref(),
            )?;
        }

        let (outputs, _) = self.norm.forward(&hidden_states, None)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            let results = batch
                .pooled_indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    let length = input_lengths[i];

                    match self.pool {
                        Pool::Cls => outputs.i((i, 0))?.unsqueeze(0),
                        Pool::LastToken => outputs.i((i, length - 1))?.unsqueeze(0),
                        Pool::Mean => {
                            // NOTE: Summed in fp32, as it overflows in fp16 for long sequences
                            let embeddings = outputs.i((i, ..length))?.to_dtype(DType::F32)?;
                            (embeddings.sum_keepdim(0)? / (length as f64))?.to_dtype(self.dtype)
                        }
                        Pool::Splade => {
                            unreachable!("Splade is not supported for Ministral3");
                        }
                    }
                })
                .collect::<Result<Vec<Tensor>>>()?;

            Some(Tensor::cat(&results, 0)?)
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            let results = batch
                .raw_indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    outputs.i((i, ..input_lengths[i]))
                })
                .collect::<Result<Vec<Tensor>>>()?;

            Some(Tensor::cat(&results, 0)?)
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for Ministral3Model {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(rope_parameters: serde_json::Value) -> Ministral3Config {
        serde_json::from_value(serde_json::json!({
            "vocab_size": 131072,
            "head_dim": 128,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 16,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-05,
            "rope_parameters": rope_parameters,
            "sliding_window": null,
            "pad_token_id": 11,
            "is_causal": false,
        }))
        .unwrap()
    }

    #[test]
    fn yarn_inv_freqs_match_transformers() {
        // https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16/blob/main/config.json
        let config = config(serde_json::json!({
            "apply_yarn_scaling": false,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 16.0,
            "llama_4_scaling_beta": 0.1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 16384,
            "rope_theta": 1000000.0,
            "rope_type": "yarn",
            "type": "yarn"
        }));

        let (inv_freqs, attention_factor) = get_inv_freqs(128, &config).unwrap();
        assert_eq!(inv_freqs.len(), 64);
        assert_eq!(attention_factor, 1.0);

        // Reference values from `transformers.modeling_rope_utils._compute_yarn_parameters`
        let expected = [
            (0, 1.0),
            (1, 8.058_422e-1),
            (20, 1.333_521_5e-2),
            (30, 6.907_023e-4),
            (40, 1.111_424_6e-5),
            (63, 7.755_861e-8),
        ];
        for (i, value) in expected {
            let rel = ((inv_freqs[i] - value) / value).abs();
            assert!(rel < 1e-4, "inv_freqs[{i}]={} != {value}", inv_freqs[i]);
        }
    }

    #[test]
    fn default_inv_freqs() {
        let config = config(serde_json::json!({"rope_theta": 10000.0, "rope_type": "default"}));

        let (inv_freqs, attention_factor) = get_inv_freqs(4, &config).unwrap();
        assert_eq!(inv_freqs, vec![1.0, 0.01]);
        assert_eq!(attention_factor, 1.0);
    }

    #[test]
    fn llama_4_query_scale() {
        let scaling = Some((0.1, 16384));
        assert_eq!(get_llama_4_query_scale(&[0, 16383], scaling), None);
        assert_eq!(get_llama_4_query_scale(&[0, 16384], None), None);

        let scale = get_llama_4_query_scale(&[0, 16383, 16384, 32768], scaling).unwrap();
        let expected = [1.0, 1.0, 1.0 + 0.1 * 2f32.ln(), 1.0 + 0.1 * 3f32.ln()];
        for (scale, expected) in scale.iter().zip(expected) {
            assert!((scale - expected).abs() < 1e-6);
        }
    }
}
