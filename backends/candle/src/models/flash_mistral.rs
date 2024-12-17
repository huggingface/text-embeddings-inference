use crate::flash_attn::flash_attn_varlen;
use crate::layers::{get_cos_sin, get_inv_freqs, HiddenAct, Linear, RMSNorm};
use crate::models::{MistralConfig, Model};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_rotary::apply_rotary_inplace;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct MistralAttention {
    qkv_linear: Linear,
    o_proj: Linear,

    window_size_left: Option<usize>,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    attention_head_size: usize,

    softmax_scale: f32,

    span: tracing::Span,
}

impl MistralAttention {
    pub fn load(vb: VarBuilder, config: &MistralConfig) -> Result<Self> {
        let window_size_left = config.sliding_window;
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let hidden_size = config.hidden_size;

        let query_weight = vb.pp("q_proj").get((hidden_size, hidden_size), "weight")?;

        let key_weight = vb.pp("k_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;

        let value_weight = vb.pp("v_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;

        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;
        let qkv_linear = Linear::new(qkv_weight, None, None);

        let o_proj_weight = vb.pp("o_proj").get((hidden_size, hidden_size), "weight")?;

        let o_proj = Linear::new(o_proj_weight, None, None);

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

        Ok(Self {
            qkv_linear,
            o_proj,
            window_size_left,
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
        max_s: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let qkv = self.qkv_linear.forward(hidden_states)?;

        // Reshape to [tokens, heads, head_size]
        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads + 2 * self.num_key_value_heads);
        new_qkv_shape.push(self.attention_head_size);

        let qkv = qkv.reshape(new_qkv_shape)?;

        // Split qkv tensor
        let q = qkv.narrow(1, 0, self.num_attention_heads)?;
        let k = qkv.narrow(1, self.num_attention_heads, self.num_key_value_heads)?;
        let v = qkv.narrow(
            1,
            self.num_attention_heads + self.num_key_value_heads,
            self.num_key_value_heads,
        )?;

        apply_rotary_inplace(&q, &k, &cos, &sin, true)?;

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
            true,
            self.window_size_left,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        self.o_proj.forward(&attention)
    }
}

struct MistralMLP {
    gate_up_proj: Linear,
    down_proj: Linear,

    act: HiddenAct,
    intermediate_size: usize,

    span: tracing::Span,
}

impl MistralMLP {
    pub fn load(vb: VarBuilder, config: &MistralConfig) -> Result<Self> {
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

        let gate_states = match self.act {
            HiddenAct::Gelu => gate_states.gelu(),
            HiddenAct::Relu => gate_states.relu(),
            HiddenAct::Swiglu => gate_states.silu(),
        }?;
        let r = self.down_proj.forward(&(gate_states * up_states)?);
        r
    }
}

struct MistralLayer {
    attention: MistralAttention,
    mlp: MistralMLP,
    input_layer_norm: RMSNorm,
    post_attention_layer_norm: RMSNorm,

    span: tracing::Span,
}

impl MistralLayer {
    pub fn load(vb: VarBuilder, config: &MistralConfig) -> Result<Self> {
        let attention = MistralAttention::load(vb.pp("self_attn"), config)?;
        let mlp = MistralMLP::load(vb.pp("mlp"), config)?;

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
        residual: Option<&Tensor>,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        let (normed_hidden_states, res) = self.input_layer_norm.forward(hidden_states, residual)?;
        let attn_output =
            self.attention
                .forward(&normed_hidden_states, cu_seqlens, cos, sin, max_s)?;
        let (normed_attn_res_output, attn_res) = self
            .post_attention_layer_norm
            .forward(&attn_output, Some(&res))?;
        let mlp_output = self.mlp.forward(&normed_attn_res_output)?;

        Ok((mlp_output, attn_res))
    }
}

pub struct FlashMistralModel {
    embeddings: Embedding,
    layers: Vec<MistralLayer>,
    norm: RMSNorm,
    cos_cache: Tensor,
    sin_cache: Tensor,
    pool: Pool,
    pub device: Device,

    span: tracing::Span,
}

impl FlashMistralModel {
    pub fn load(vb: VarBuilder, config: &MistralConfig, model_type: ModelType) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashMistral requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashMistral requires DType::F16")
        }

        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Mistral")
            }
            ModelType::Embedding(pool) => pool,
        };

        let embeddings = Embedding::new(
            vb.pp("embed_tokens")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );

        let layers = (0..config.num_hidden_layers)
            .map(|index| MistralLayer::load(vb.pp(format!("layers.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        let inv_freqs = get_inv_freqs(
            layers[0].attention.attention_head_size,
            config.rope_theta,
            vb.device(),
            None,
        )?;
        let (cos_cache, sin_cache) = get_cos_sin(
            config.max_position_embeddings,
            &inv_freqs,
            vb.dtype(),
            false,
        )?;

        Ok(Self {
            embeddings,
            layers,
            norm,
            cos_cache,
            sin_cache,
            pool,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.cumulative_seq_lengths.len() - 1;
        let shape = batch.input_ids.len();

        // Create Cuda tensors
        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        let mut hidden_states = self.embeddings.forward(&input_ids)?;

        let cos = self.cos_cache.index_select(&position_ids, 0)?;
        let sin = self.sin_cache.index_select(&position_ids, 0)?;

        let mut residual = None;
        for layer in &self.layers {
            let (h, r) = layer.forward(
                &hidden_states,
                residual.as_ref(),
                &cu_seqlens,
                &cos,
                &sin,
                batch.max_length as usize,
            )?;
            hidden_states = h;
            residual = Some(r);
        }

        let (outputs, _) = self.norm.forward(&hidden_states, residual.as_ref())?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                // CLS and LastToken pooling
                Pool::Cls | Pool::LastToken => {
                    if batch_size > 1 {
                        // Get token indices form cu_seqlens
                        let mut indices = match self.pool {
                            Pool::Cls => cu_seqlens.narrow(0, 0, batch_size)?,
                            Pool::LastToken => {
                                let end = cu_seqlens.narrow(0, 1, batch_size)?;
                                (&end - &end.ones_like()?)?
                            }
                            _ => unreachable!(),
                        };

                        // If raw_indices is empty, we don't need to do anything with
                        // the pooled_indices
                        if has_raw_requests {
                            // We need the pooled indices to select the correct cls indices
                            let pooled_indices = Tensor::from_vec(
                                batch.pooled_indices.clone(),
                                batch.pooled_indices.len(),
                                &self.device,
                            )?;

                            // Only select indices that requires pooling
                            indices = indices.index_select(&pooled_indices, 0)?
                        }

                        // Select tokens
                        Some(outputs.index_select(&indices, 0)?)
                    } else {
                        Some(
                            match self.pool {
                                Pool::Cls => outputs.i(0)?,
                                Pool::LastToken => {
                                    outputs.i(batch.cumulative_seq_lengths[1] as usize - 1)?
                                }
                                _ => unreachable!(),
                            }
                            .unsqueeze(0)?,
                        )
                    }
                }
                // Mean pooling
                Pool::Mean => {
                    if batch_size > 1 {
                        // for each request that requires pooling
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .into_iter()
                            .map(|i| {
                                let i = i as usize;
                                let start = batch.cumulative_seq_lengths[i];
                                let len = batch.cumulative_seq_lengths[i + 1] - start;

                                // Mean
                                let embeddings = outputs.narrow(0, start as usize, len as usize)?;
                                embeddings.sum_keepdim(0)? / (len as f64)
                            })
                            .collect();

                        // Concatenate all results
                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        Some((outputs.sum_keepdim(0)? / (batch.max_length as f64))?)
                    }
                }
                Pool::Splade => {
                    unreachable!();
                }
            }
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                // Create indexing vector for the embeddings
                let mut final_indices: Vec<u32> = Vec::with_capacity(shape);
                for i in batch.raw_indices.into_iter() {
                    let i = i as usize;
                    // Get start/end token index of this specific member of the batch
                    let start = batch.cumulative_seq_lengths[i];
                    let end = batch.cumulative_seq_lengths[i + 1];

                    for j in start..end {
                        // Add indices for the tokens of this specific member of the batch
                        final_indices.push(j);
                    }
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

                // Select the tokens with final indices
                Some(outputs.index_select(&final_indices, 0)?)
            } else {
                Some(outputs)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for FlashMistralModel {
    fn is_padded(&self) -> bool {
        false
    }
    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
