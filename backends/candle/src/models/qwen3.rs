use crate::layers::{
    apply_rotary, get_cos_sin, get_cublas_lt_wrapper, get_inv_freqs, HiddenAct, Linear, RMSNorm,
    RopeParameters,
};
use crate::models::Model;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;

use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3Config {
    pub attention_bias: bool,
    pub vocab_size: usize,
    pub head_dim: Option<usize>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: Option<f32>,
    pub rope_parameters: Option<RopeParameters>,
    pub sliding_window: Option<usize>,
    pub use_sliding_window: bool,
    pub eos_token_id: usize,
    // TODO(alvarobartt): Migrate to `is_causal` instead
    // https://github.com/huggingface/transformers/pull/43705
    #[serde(default)]
    pub use_bidirectional_attention: Option<bool>,
    #[serde(default)]
    pub num_labels: Option<usize>,
}

struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    attention_head_size: usize,

    softmax_scale: f64,

    span: tracing::Span,
}

impl Qwen3Attention {
    pub fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        if config.use_sliding_window {
            candle::bail!("Sliding window is not supported for Qwen3");
        }

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
        let query_bias = if config.attention_bias {
            Some(
                vb.pp("q_proj")
                    .get(num_attention_heads * attention_head_size, "bias")?,
            )
        } else {
            None
        };
        let q_proj = Linear::new(query_weight, query_bias, None);

        let key_weight = vb.pp("k_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let key_bias = if config.attention_bias {
            Some(
                vb.pp("k_proj")
                    .get(num_key_value_heads * attention_head_size, "bias")?,
            )
        } else {
            None
        };
        let k_proj = Linear::new(key_weight, key_bias, None);

        let value_weight = vb.pp("v_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let value_bias = if config.attention_bias {
            Some(
                vb.pp("v_proj")
                    .get(num_key_value_heads * attention_head_size, "bias")?,
            )
        } else {
            None
        };
        let v_proj = Linear::new(value_weight, value_bias, None);

        let o_proj_weight = vb.pp("o_proj").get(
            (hidden_size, num_attention_heads * attention_head_size),
            "weight",
        )?;
        let o_proj = Linear::new(o_proj_weight, None, None);

        let q_norm = RMSNorm::load(vb.pp("q_norm"), attention_head_size, config.rms_norm_eps)?;
        let k_norm = RMSNorm::load(vb.pp("k_norm"), attention_head_size, config.rms_norm_eps)?;

        let softmax_scale = 1.0 / (attention_head_size as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
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
                attn_weights = attn_weights.add(attention_bias)?;
            }

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v.contiguous()?)
        }?;

        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;

        self.o_proj.forward(&context_layer)
    }
}

struct Qwen3MLP {
    gate_up_proj: Linear,
    down_proj: Linear,

    activation: HiddenAct,
    intermediate_size: usize,

    span: tracing::Span,
}

impl Qwen3MLP {
    pub fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
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

struct Qwen3Layer {
    attention: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layer_norm: RMSNorm,
    post_attention_layer_norm: RMSNorm,

    span: tracing::Span,
}

impl Qwen3Layer {
    pub fn load(vb: VarBuilder, config: &Qwen3Config) -> Result<Self> {
        let attention = Qwen3Attention::load(vb.pp("self_attn"), config)?;
        let mlp = Qwen3MLP::load(vb.pp("mlp"), config)?;

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
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (normed_hidden_states, residual) =
            self.input_layer_norm.forward(hidden_states, None)?;

        let attn_output =
            self.attention
                .forward(&normed_hidden_states, attention_bias, cos, sin)?;

        let (normed_attn_res_output, attn_res) = self
            .post_attention_layer_norm
            .forward(&attn_output, Some(&residual))?;

        let mlp_output = self.mlp.forward(&normed_attn_res_output)?;

        let output = (&mlp_output + &attn_res)?;

        Ok(output)
    }
}

pub struct Qwen3Model {
    embeddings: Embedding,
    layers: Vec<Qwen3Layer>,
    norm: RMSNorm,
    // TODO(alvarobartt): Eventually extend Qwen3 for Voyage instead of adding `projection` here
    projection: Option<Linear>,
    rotary_cache: (Tensor, Tensor),
    rotary_dim: usize,
    pool: Pool,
    num_attention_heads: usize,
    pad_token_id: u32,
    use_bidirectional_attention: bool,

    dtype: DType,
    device: Device,

    span: tracing::Span,
}

impl Qwen3Model {
    pub fn load(vb: VarBuilder, config: &Qwen3Config, model_type: ModelType) -> Result<Self> {
        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Qwen3")
            }
            ModelType::Embedding(pool) => pool,
        };

        // The Qwen3-Reranker models contain the `model` key
        // https://huggingface.co/collections/Qwen/qwen3-reranker-6841b22d0192d7ade9cdefea
        let model_prefix = if vb.contains_tensor("model.embed_tokens.weight") {
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
            .map(|index| Qwen3Layer::load(vb.pp(format!("{model_prefix}layers.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(
            vb.pp(format!("{model_prefix}norm")),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        let projection = if let Some(num_labels) = config.num_labels {
            if vb.contains_tensor("linear.weight") {
                let projection_weight =
                    vb.get((num_labels, config.hidden_size), "linear.weight")?;
                Some(Linear::new(projection_weight, None, None))
            } else {
                tracing::warn!(
                    "num_labels is set but linear.weight not found, skipping projection layer"
                );
                None
            }
        } else {
            None
        };

        let use_bidirectional_attention = config.use_bidirectional_attention.unwrap_or(false);

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

        Ok(Self {
            embeddings,
            layers,
            norm,
            projection,
            rotary_cache,
            rotary_dim,
            pool,
            pad_token_id: config.eos_token_id as u32,
            num_attention_heads: config.num_attention_heads,
            use_bidirectional_attention,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    fn get_causal_attention_bias(&self, attention_bias: Tensor) -> Result<Tensor> {
        let (bs, dim, seq_len, _) = attention_bias.dims4()?;

        let mask: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| (j > i) as u8))
            .collect();

        let device = attention_bias.device();
        let causal_mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let causal_mask = causal_mask.expand(&[bs, dim, seq_len, seq_len])?;

        let min_value = match self.dtype {
            DType::F32 => f32::MIN,
            _ => -65504.0, // f16 minimum value
        };

        let negatives =
            Tensor::full(min_value, attention_bias.shape(), device)?.to_dtype(self.dtype)?;
        let zeros = Tensor::zeros_like(&attention_bias)?.to_dtype(self.dtype)?;

        let causal_mask = causal_mask
            .where_cond(&negatives, &zeros)?
            .to_device(device)?;

        attention_bias.broadcast_add(&causal_mask)
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let shape = (batch_size, max_length);

        let (input_ids, position_ids, input_lengths, attention_bias) = if batch_size > 1 {
            // Prepare padded batch
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

                // Left padding for Qwen3-Embedding (pad at the beginning)
                let padding = max_length - seq_length;
                if padding > 0 {
                    masking = true;
                    for _ in 0..padding {
                        input_ids.push(self.pad_token_id);
                        position_ids.push(0);
                        attention_bias.push(f32::NEG_INFINITY);
                    }
                }

                // Then add the actual sequence
                for j in start..end {
                    input_ids.push(batch.input_ids[j]);
                    position_ids.push(batch.position_ids[j]);
                    attention_bias.push(0.0);
                }
            }

            let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
            let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;

            let attention_bias = if masking {
                let attention_bias =
                    Tensor::from_vec(attention_bias, (batch_size, 1, 1, max_length), &self.device)?
                        .to_dtype(self.dtype)?;
                // Broadcast once instead of at every layer
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
            // Create attention bias for causal masking even for single sequences
            let attention_bias = Tensor::zeros(
                (1, self.num_attention_heads, seq_len, seq_len),
                self.dtype,
                &self.device,
            )?;

            (input_ids, position_ids, input_lengths, Some(attention_bias))
        };

        let attention_bias = if self.use_bidirectional_attention {
            attention_bias
        } else if let Some(attn_bias) = attention_bias {
            Some(self.get_causal_attention_bias(attn_bias)?)
        } else {
            None
        };

        let mut hidden_states = self.embeddings.forward(&input_ids)?;

        let cos = self
            .rotary_cache
            .0
            .index_select(&position_ids.flatten_all()?, 0)?;
        let sin = self
            .rotary_cache
            .1
            .index_select(&position_ids.flatten_all()?, 0)?;

        let cos = cos.reshape((batch_size, 1, max_length, self.rotary_dim))?;
        let sin = sin.reshape((batch_size, 1, max_length, self.rotary_dim))?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_bias.as_ref(), &cos, &sin)?;
        }

        let (outputs, _) = self.norm.forward(&hidden_states, None)?;

        let outputs = if let Some(ref projection) = self.projection {
            projection.forward(&outputs)?
        } else {
            outputs
        };

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                Pool::Cls => {
                    if batch_size > 1 {
                        let pooled_indices = Tensor::from_vec(
                            batch.pooled_indices.clone(),
                            batch.pooled_indices.len(),
                            &self.device,
                        )?;

                        let cls_indices = if has_raw_requests {
                            Tensor::zeros(
                                batch.pooled_indices.len(),
                                candle::DType::U32,
                                &self.device,
                            )?
                        } else {
                            Tensor::arange(0u32, batch_size as u32, &self.device)?
                        };

                        Some(outputs.i((&pooled_indices, &cls_indices))?)
                    } else {
                        Some(outputs.i((0, 0))?.unsqueeze(0)?)
                    }
                }
                Pool::LastToken => {
                    if batch_size > 1 {
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .iter()
                            .map(|&i| {
                                let i = i as usize;
                                // With left padding, the last token is always at max_length - 1
                                let last_token_idx = max_length - 1;
                                outputs.i((i, last_token_idx))?.unsqueeze(0)
                            })
                            .collect();

                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        // For single inference, use the actual last token position from cumulative_seq_lengths
                        let last_idx = batch.cumulative_seq_lengths[1] as usize - 1;
                        Some(outputs.i((0, last_idx))?.unsqueeze(0)?)
                    }
                }
                Pool::Mean => {
                    if batch_size > 1 {
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .iter()
                            .map(|&i| {
                                let i = i as usize;
                                let length = input_lengths[i];

                                // With left padding, actual tokens are at the end
                                let padding = max_length - length;
                                let embeddings = outputs.i((i, padding..))?;
                                let sum = embeddings.sum_keepdim(0)?;
                                sum / (length as f64)
                            })
                            .collect();

                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        let length = input_lengths[0] as f64;
                        let embeddings = outputs.i((0, ..input_lengths[0]))?;
                        Some((embeddings.sum_keepdim(0)? / length)?)
                    }
                }
                Pool::Splade => {
                    unreachable!("Splade is not supported for Qwen3");
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

impl Model for Qwen3Model {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
