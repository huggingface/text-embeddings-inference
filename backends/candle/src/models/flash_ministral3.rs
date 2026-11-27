use crate::flash_attn::flash_attn_varlen;
use crate::layers::{HiddenAct, Linear, RMSNorm};
use crate::models::ministral3::{get_inv_freqs, get_llama_4_query_scale, get_llama_4_scaling};
use crate::models::{Ministral3Config, Model};

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_rotary::apply_rotary_inplace;

use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct Ministral3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    attention_head_size: usize,

    softmax_scale: f32,

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

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

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
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        query_scale: Option<&Tensor>,
        max_s: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [tokens, heads, head_dim]
        let input_dims = hidden_states.dims();
        let input_shape = &input_dims[..input_dims.len() - 1];

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

        apply_rotary_inplace(&q, &k, cos, sin, true)?;

        // Llama 4 position-dependent attention scaling, applied to the queries
        let q = match query_scale {
            Some(query_scale) => q.broadcast_mul(query_scale)?,
            None => q,
        };

        let attention = flash_attn_varlen(
            &q,
            &k,
            &v,
            None,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            self.softmax_scale,
            // NOTE: Only bidirectional attention is supported i.e., `is_causal=false`
            false,
            None,
            None,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        self.o_proj.forward(&attention)
    }
}

struct Ministral3MLP {
    gate_up_proj: Linear,
    down_proj: Linear,

    act: HiddenAct,
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
            intermediate_size,
            act: config.hidden_act.clone(),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.gate_up_proj.forward(hidden_states)?;
        let gate_states = gate_up_states.narrow(1, 0, self.intermediate_size)?;
        let up_states = gate_up_states.narrow(1, self.intermediate_size, self.intermediate_size)?;

        let gate_states = self.act.forward(&gate_states)?;

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

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        query_scale: Option<&Tensor>,
        max_s: usize,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        let (normed_hidden_states, res) = self.input_layer_norm.forward(hidden_states, residual)?;

        let attn_output = self.attention.forward(
            &normed_hidden_states,
            cu_seqlens,
            cos,
            sin,
            query_scale,
            max_s,
        )?;

        let (normed_attn_res_output, attn_res) = self
            .post_attention_layer_norm
            .forward(&attn_output, Some(&res))?;

        let mlp_output = self.mlp.forward(&normed_attn_res_output)?;

        Ok((mlp_output, attn_res))
    }
}

pub struct FlashMinistral3Model {
    embeddings: Embedding,
    layers: Vec<Ministral3Layer>,
    norm: RMSNorm,
    inv_freqs: Tensor,
    rotary_scale: f32,
    // NOTE: `(beta, original_max_position_embeddings)`
    llama_4_scaling: Option<(f32, usize)>,
    pool: Pool,
    pub device: Device,

    span: tracing::Span,
}

impl FlashMinistral3Model {
    pub fn load(vb: VarBuilder, config: &Ministral3Config, model_type: ModelType) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashMinistral3 requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashMinistral3 requires DType::F16")
        }

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

        // NOTE: The cos/sin are computed per batch out of the position ids rather than cached, as
        // `max_position_embeddings=262144`
        let rotary_dim = layers[0].attention.attention_head_size;
        let (inv_freqs, rotary_scale) = get_inv_freqs(rotary_dim, config)?;
        let inv_freqs = Tensor::from_vec(inv_freqs, (1, rotary_dim / 2), vb.device())?;

        Ok(Self {
            embeddings,
            layers,
            norm,
            inv_freqs,
            rotary_scale,
            llama_4_scaling: get_llama_4_scaling(config)?,
            pool,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.cumulative_seq_lengths.len() - 1;
        let shape = batch.input_ids.len();

        let query_scale = get_llama_4_query_scale(&batch.position_ids, self.llama_4_scaling)
            .map(|scale| Tensor::from_vec(scale, (shape, 1, 1), &self.device)?.to_dtype(DType::F16))
            .transpose()?;

        let positions: Vec<f32> = batch.position_ids.iter().map(|&p| p as f32).collect();
        let freqs = Tensor::from_vec(positions, (shape, 1), &self.device)?
            .broadcast_mul(&self.inv_freqs)?;

        let (mut cos, mut sin) = (freqs.cos()?, freqs.sin()?);
        if self.rotary_scale != 1.0 {
            cos = (cos * self.rotary_scale as f64)?;
            sin = (sin * self.rotary_scale as f64)?;
        }
        let cos = cos.to_dtype(DType::F16)?;
        let sin = sin.to_dtype(DType::F16)?;

        // Create Cuda tensors
        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        let mut hidden_states = self.embeddings.forward(&input_ids)?;

        let mut residual = None;
        for layer in &self.layers {
            let (h, r) = layer.forward(
                &hidden_states,
                residual.as_ref(),
                &cu_seqlens,
                &cos,
                &sin,
                query_scale.as_ref(),
                batch.max_length as usize,
            )?;
            hidden_states = h;
            residual = Some(r);
        }

        let (outputs, _) = self.norm.forward(&hidden_states, residual.as_ref())?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            let results = batch
                .pooled_indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    let start = batch.cumulative_seq_lengths[i] as usize;
                    let end = batch.cumulative_seq_lengths[i + 1] as usize;

                    match self.pool {
                        Pool::Cls => outputs.i(start)?.unsqueeze(0),
                        Pool::LastToken => outputs.i(end - 1)?.unsqueeze(0),
                        Pool::Mean => {
                            // NOTE: Summed in fp32, as it overflows in fp16 for long sequences
                            let embeddings = outputs
                                .narrow(0, start, end - start)?
                                .to_dtype(DType::F32)?;
                            (embeddings.sum_keepdim(0)? / ((end - start) as f64))?
                                .to_dtype(DType::F16)
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
            if has_pooling_requests {
                let results = batch
                    .raw_indices
                    .iter()
                    .map(|&i| {
                        let i = i as usize;
                        let start = batch.cumulative_seq_lengths[i] as usize;
                        let end = batch.cumulative_seq_lengths[i + 1] as usize;
                        outputs.narrow(0, start, end - start)
                    })
                    .collect::<Result<Vec<Tensor>>>()?;

                Some(Tensor::cat(&results, 0)?)
            } else {
                Some(outputs)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for FlashMinistral3Model {
    fn is_padded(&self) -> bool {
        false
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
