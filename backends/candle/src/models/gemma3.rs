use crate::layers::{
    apply_rotary, get_cos_sin, get_cublas_lt_wrapper, get_inv_freqs, HiddenAct, Linear,
    RopeParameters,
};
use crate::models::Model;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma3Config {
    pub attention_bias: bool,
    pub pad_token_id: u32,
    pub head_dim: Option<usize>,
    pub hidden_activation: HiddenAct,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub query_pre_attn_scalar: usize,
    pub rms_norm_eps: f32,
    pub rope_local_base_freq: f32,
    pub rope_theta: Option<f32>,
    pub rope_parameters: Option<RopeParameters>,
    pub sliding_window: Option<usize>,
    #[serde(rename = "_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    pub vocab_size: usize,
}

#[derive(Debug)]
pub struct Gemma3RMSNorm {
    weight: Tensor,
    epsilon: f32,

    span: tracing::Span,
}

impl Gemma3RMSNorm {
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
                        // NOTE: Gemma3 multiplies by (1.0 + weight) for scaling after normalization
                        .broadcast_mul(&(&self.weight + 1.0)?)?,
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
                            &(&self.weight + 1.0)?,
                            None,
                            self.epsilon,
                        )?;
                        Ok((
                            result.reshape(original_shape)?,
                            residual_add.reshape(original_shape)?,
                        ))
                    } else {
                        let residual_add = hidden_states.clone();

                        let result =
                            rms_norm(&hidden_states, &(&self.weight + 1.0)?, None, self.epsilon)?;

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

enum Gemma3AttentionType {
    FullAttention,
    SlidingAttention,
}

struct Gemma3Attention {
    qkv_proj: Linear,
    o_proj: Linear,

    q_norm: Gemma3RMSNorm,
    k_norm: Gemma3RMSNorm,

    attention_head_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    scaling: f64,

    sliding_window: Option<usize>,

    span: tracing::Span,
}

impl Gemma3Attention {
    pub fn load(
        vb: VarBuilder,
        config: &Gemma3Config,
        attention_type: Gemma3AttentionType,
    ) -> Result<Self> {
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
        let key_weight = vb.pp("k_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let value_weight = vb.pp("v_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;

        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;

        let qkv_bias = if config.attention_bias {
            let query_bias = vb
                .pp("q_proj")
                .get(num_attention_heads * attention_head_size, "bias")?;
            let key_bias = vb
                .pp("k_proj")
                .get(num_key_value_heads * attention_head_size, "bias")?;
            let value_bias = vb
                .pp("v_proj")
                .get(num_key_value_heads * attention_head_size, "bias")?;
            Some(Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?)
        } else {
            None
        };

        let qkv_proj = Linear::new(qkv_weight, qkv_bias, None);

        let output_weight = vb.pp("o_proj").get(
            (hidden_size, num_attention_heads * attention_head_size),
            "weight",
        )?;
        let output_bias = if config.attention_bias {
            Some(vb.pp("o_proj").get(hidden_size, "bias")?)
        } else {
            None
        };
        let o_proj = Linear::new(output_weight, output_bias, None);

        let q_norm =
            Gemma3RMSNorm::load(vb.pp("q_norm"), attention_head_size, config.rms_norm_eps)?;
        let k_norm =
            Gemma3RMSNorm::load(vb.pp("k_norm"), attention_head_size, config.rms_norm_eps)?;

        let scaling = 1.0 / (config.query_pre_attn_scalar as f64).sqrt();

        match attention_type {
            Gemma3AttentionType::FullAttention => Ok(Self {
                qkv_proj,
                o_proj,
                q_norm,
                k_norm,
                attention_head_size,
                num_attention_heads,
                num_key_value_heads,
                scaling,
                sliding_window: None,
                span: tracing::span!(tracing::Level::TRACE, "full_attention"),
            }),
            Gemma3AttentionType::SlidingAttention => Ok(Self {
                qkv_proj,
                o_proj,
                q_norm,
                k_norm,
                attention_head_size,
                num_attention_heads,
                num_key_value_heads,
                scaling,
                sliding_window: config.sliding_window,
                span: tracing::span!(tracing::Level::TRACE, "sliding_attention"),
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_causal_mask(
        &self,
        batch_size: usize,
        dim: usize,
        seq_len: usize,
        device: &Device,
        dtype: DType,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        let min_value = match dtype {
            DType::F32 => f32::MIN,
            _ => -65504.0, // f16 minimum value
        };

        let mask: Vec<u8> = if let Some(window_size) = sliding_window {
            // Bi-directional sliding window mask, meaning a token can attend to any
            // other token if their absolute distance is within half the sliding window size
            let half_window = window_size / 2;
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        let distance = i.abs_diff(j);
                        (distance <= half_window) as u8
                    })
                })
                .collect()
        } else {
            // Full attention mask, meaning a token can attend to all tokens
            vec![1u8; seq_len * seq_len]
        };

        let mask_tensor = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let expanded_mask = mask_tensor.expand(&[batch_size, dim, seq_len, seq_len])?;

        let zeros = Tensor::zeros((batch_size, dim, seq_len, seq_len), dtype, device)?;
        let negatives = Tensor::full(min_value, (batch_size, dim, seq_len, seq_len), device)?
            .to_dtype(dtype)?;

        expanded_mask.where_cond(&zeros, &negatives)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let device = hidden_states.device();

        let qkv = self.qkv_proj.forward(hidden_states)?;

        let input_dims = hidden_states.dims();
        let input_shape = &input_dims[..input_dims.len() - 1];

        let q_size = self.num_attention_heads * self.attention_head_size;
        let k_size = self.num_key_value_heads * self.attention_head_size;
        let v_size = self.num_key_value_heads * self.attention_head_size;

        let q = qkv.narrow(D::Minus1, 0, q_size)?;
        let k = qkv.narrow(D::Minus1, q_size, k_size)?;
        let v = qkv.narrow(D::Minus1, q_size + k_size, v_size)?;

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

        let (q, _) = self.q_norm.forward(&q, None)?;
        let (k, _) = self.k_norm.forward(&k, None)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = apply_rotary(&q, cos, sin, self.attention_head_size)?;
        let k = apply_rotary(&k, cos, sin, self.attention_head_size)?;

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

        let attention_bias = match attention_bias {
            Some(attention_bias) => {
                let (batch_size, dim, seq_length, _) = attention_bias.shape().dims4()?;
                let causal_mask = self.create_causal_mask(
                    batch_size,
                    dim,
                    seq_length,
                    attention_bias.device(),
                    attention_bias.dtype(),
                    self.sliding_window,
                )?;
                Some(attention_bias.broadcast_add(&causal_mask)?)
            }
            None => None,
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
                let attention_bias = attention_bias.map(|mask| mask.flatten(0, 1)).transpose()?;

                let beta = match attention_bias.is_some() {
                    true => Some(1.0),
                    false => None,
                };

                let attention_scores = cublaslt.batch_matmul(
                    &k,
                    &q,
                    attention_bias.as_ref(),
                    Some(self.scaling as f32),
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
            let mut attn_weights = (attn_weights * self.scaling)?;

            if let Some(attention_bias) = attention_bias {
                attn_weights = attn_weights.broadcast_add(&attention_bias)?;
            }

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v.contiguous()?)
        }?;

        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;
        self.o_proj.forward(&context_layer)
    }
}

struct Gemma3MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    hidden_activation: HiddenAct,

    intermediate_size: usize,

    span: tracing::Span,
}

impl Gemma3MLP {
    pub fn load(vb: VarBuilder, config: &Gemma3Config) -> Result<Self> {
        let gate_proj_weight = vb
            .pp("gate_proj")
            .get((config.intermediate_size, config.hidden_size), "weight")?;

        let up_proj_weight = vb
            .pp("up_proj")
            .get((config.intermediate_size, config.hidden_size), "weight")?;

        let gate_up_proj_weight = Tensor::cat(&[&gate_proj_weight, &up_proj_weight], 0)?;
        let gate_up_proj = Linear::new(gate_up_proj_weight, None, None);

        let down_proj_weight = vb
            .pp("down_proj")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let down_proj = Linear::new(down_proj_weight, None, None);

        Ok(Self {
            gate_up_proj,
            down_proj,
            hidden_activation: config.hidden_activation.clone(),
            intermediate_size: config.intermediate_size,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.gate_up_proj.forward(hidden_states)?;

        let gate_states = gate_up_states.narrow(D::Minus1, 0, self.intermediate_size)?;
        let gate_states = self.hidden_activation.forward(&gate_states)?;

        let up_states =
            gate_up_states.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;

        self.down_proj.forward(&(gate_states * up_states)?)
    }
}

struct Gemma3Layer {
    input_layernorm: Gemma3RMSNorm,
    self_attn: Gemma3Attention,
    post_attention_layernorm: Gemma3RMSNorm,

    pre_feedforward_layernorm: Gemma3RMSNorm,
    mlp: Gemma3MLP,
    post_feedforward_layernorm: Gemma3RMSNorm,

    span: tracing::Span,
}

impl Gemma3Layer {
    pub fn load(
        vb: VarBuilder,
        config: &Gemma3Config,
        attention_type: Gemma3AttentionType,
    ) -> Result<Self> {
        let input_layernorm = Gemma3RMSNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let self_attn = Gemma3Attention::load(vb.pp("self_attn"), config, attention_type)?;
        let post_attention_layernorm = Gemma3RMSNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        let pre_feedforward_layernorm = Gemma3RMSNorm::load(
            vb.pp("pre_feedforward_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let mlp = Gemma3MLP::load(vb.pp("mlp"), config)?;
        let post_feedforward_layernorm = Gemma3RMSNorm::load(
            vb.pp("post_feedforward_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            mlp,
            post_feedforward_layernorm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let residual = hidden_states.clone();

        let (hidden_states, _) = self.input_layernorm.forward(hidden_states, None)?;
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, attention_bias, cos, sin)?;
        let (hidden_states, _) = self
            .post_attention_layernorm
            .forward(&hidden_states, None)?;
        let hidden_states = residual.broadcast_add(&hidden_states)?;

        let residual = hidden_states.clone();
        let (hidden_states, _) = self
            .pre_feedforward_layernorm
            .forward(&hidden_states, None)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let (hidden_states, _) = self
            .post_feedforward_layernorm
            .forward(&hidden_states, None)?;
        residual.broadcast_add(&hidden_states)
    }
}

pub struct Gemma3Embedding {
    embedding: Embedding,
    scale: f64,

    span: tracing::Span,
}

impl Gemma3Embedding {
    pub fn load(vb: VarBuilder, config: &Gemma3Config) -> Result<Self> {
        let embedding = Embedding::new(
            vb.get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );
        let scale = (config.hidden_size as f64).sqrt();

        Ok(Self {
            embedding,
            scale,
            span: tracing::span!(tracing::Level::TRACE, "embed_tokens"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.embedding.forward(input_ids)?;
        hidden_states * self.scale
    }
}

pub struct Gemma3Model {
    embed_tokens: Gemma3Embedding,
    layers: Vec<Gemma3Layer>,
    norm: Gemma3RMSNorm,

    rotary_cache: (Tensor, Tensor),
    rotary_cache_local_attention: (Tensor, Tensor),
    rotary_dim: usize,

    num_attention_heads: usize,
    pad_token_id: u32,
    pool: Pool,

    dtype: DType,
    device: Device,

    span: tracing::Span,
}

impl Gemma3Model {
    pub fn load(vb: VarBuilder, config: &Gemma3Config, model_type: ModelType) -> Result<Self> {
        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Gemma3")
            }
            ModelType::Embedding(pool) => pool,
        };

        let embed_tokens = Gemma3Embedding::load(vb.pp("embed_tokens"), config)?;

        let layers = (0..config.num_hidden_layers)
            .map(|layer_idx| {
                let attention_type = match (layer_idx + 1) % config.sliding_window_pattern > 0 {
                    false => Gemma3AttentionType::FullAttention,
                    true => Gemma3AttentionType::SlidingAttention,
                };
                Gemma3Layer::load(vb.pp(format!("layers.{layer_idx}")), config, attention_type)
            })
            .collect::<Result<Vec<Gemma3Layer>>>()?;

        let norm = Gemma3RMSNorm::load(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        let rotary_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        // NOTE: https://github.com/huggingface/transformers/pull/39847
        let rope_theta = match config.rope_theta {
            Some(rope_theta) => rope_theta,
            None => match &config.rope_parameters {
                Some(rope_parameters) => rope_parameters.rope_theta,
                None => candle::bail!("Neither `rope_theta` nor `rope_parameters.rope_theta` are defined in the `config.json`"),
            },
        };

        let inv_freqs = get_inv_freqs(rotary_dim, rope_theta, vb.device(), None)?;
        let rotary_cache =
            get_cos_sin(config.max_position_embeddings, &inv_freqs, vb.dtype(), true)?;

        let inv_freqs_local =
            get_inv_freqs(rotary_dim, config.rope_local_base_freq, vb.device(), None)?;
        let rotary_cache_local_attention = get_cos_sin(
            config.max_position_embeddings,
            &inv_freqs_local,
            vb.dtype(),
            true,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_cache,
            rotary_cache_local_attention,
            rotary_dim,
            pool,
            pad_token_id: config.pad_token_id,
            num_attention_heads: config.num_attention_heads,
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

        let (input_ids, position_ids, input_lengths, attention_bias) = if batch_size > 1 {
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

            let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
            let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;

            let attention_bias = if masking {
                let attention_bias =
                    Tensor::from_vec(attention_bias, (batch_size, 1, 1, max_length), &self.device)?
                        .to_dtype(self.dtype)?;

                let attention_bias = attention_bias
                    .broadcast_as((batch_size, self.num_attention_heads, max_length, max_length))?
                    .contiguous()?;
                Some(attention_bias)
            } else {
                None
            };

            (input_ids, position_ids, input_lengths, attention_bias)
        } else {
            let input_ids = Tensor::from_vec(
                batch.input_ids.clone(),
                (1, batch.input_ids.len()),
                &self.device,
            )?;
            let position_ids = Tensor::from_vec(
                batch.position_ids.clone(),
                (1, batch.position_ids.len()),
                &self.device,
            )?;
            let input_lengths = vec![batch.input_ids.len()];

            let seq_len = batch.input_ids.len();
            let attention_bias = Tensor::zeros(
                (1, self.num_attention_heads, seq_len, seq_len),
                self.dtype,
                &self.device,
            )?;

            (input_ids, position_ids, input_lengths, Some(attention_bias))
        };

        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;

        let cos = self
            .rotary_cache
            .0
            .index_select(&position_ids.flatten_all()?, 0)?;
        let cos = cos.reshape((batch_size, 1, max_length, self.rotary_dim))?;
        let sin = self
            .rotary_cache
            .1
            .index_select(&position_ids.flatten_all()?, 0)?;
        let sin = sin.reshape((batch_size, 1, max_length, self.rotary_dim))?;

        let cos_local = self
            .rotary_cache_local_attention
            .0
            .index_select(&position_ids.flatten_all()?, 0)?;
        let cos_local = cos_local.reshape((batch_size, 1, max_length, self.rotary_dim))?;
        let sin_local = self
            .rotary_cache_local_attention
            .1
            .index_select(&position_ids.flatten_all()?, 0)?;
        let sin_local = sin_local.reshape((batch_size, 1, max_length, self.rotary_dim))?;

        for layer in &self.layers {
            hidden_states = if layer.self_attn.sliding_window.is_some() {
                layer.forward(
                    &hidden_states,
                    attention_bias.as_ref(),
                    &cos_local,
                    &sin_local,
                )?
            } else {
                layer.forward(&hidden_states, attention_bias.as_ref(), &cos, &sin)?
            };
        }

        let (outputs, _) = self.norm.forward(&hidden_states, None)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                Pool::Cls | Pool::LastToken | Pool::Splade => {
                    unreachable!("Only Mean Pooling is supported for Gemma3, neither CLS, nor Last-Token, nor SPLADE");
                }
                Pool::Mean => {
                    if batch_size > 1 {
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .iter()
                            .map(|&i| {
                                let i = i as usize;
                                let length = input_lengths[i];
                                let embeddings = outputs.i((i, ..length))?;
                                embeddings.sum_keepdim(0)? / (length as f64)
                            })
                            .collect();

                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        let length = input_lengths[0];
                        let embeddings = outputs.i((0, ..length))?;
                        Some((embeddings.sum_keepdim(0)? / (length as f64))?)
                    }
                }
            }
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                let mut final_embeddings = Vec::new();
                for &i in &batch.raw_indices {
                    let i = i as usize;
                    let length = input_lengths[i];
                    final_embeddings.push(outputs.i((i, ..length))?);
                }
                Some(Tensor::cat(&final_embeddings, 0)?)
            } else {
                // Single batch or all raw requests
                if batch_size == 1 {
                    let length = input_lengths[0];
                    Some(outputs.i((0, ..length))?)
                } else {
                    // Multiple sequences, all raw
                    let mut all_embeddings = Vec::new();
                    for (i, &length) in input_lengths.iter().enumerate().take(batch_size) {
                        all_embeddings.push(outputs.i((i, ..length))?);
                    }
                    Some(Tensor::cat(&all_embeddings, 0)?)
                }
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for Gemma3Model {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
