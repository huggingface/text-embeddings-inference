use crate::flash_attn::flash_attn_varlen;
use crate::layers::{get_cos_sin, get_inv_freqs, LayerNorm, Linear};
use crate::models::{GTEClassificationHead, GTEConfig, Model, PositionEmbeddingType, GTEMLP};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_rotary::apply_rotary_inplace;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct GTEAttention {
    qkv_linear: Linear,
    o_proj: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,

    softmax_scale: f32,

    span: tracing::Span,
}

impl GTEAttention {
    pub fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let hidden_size = config.hidden_size;

        let qkv_weight = vb
            .pp("qkv_proj")
            .get((hidden_size * 3, hidden_size), "weight")?;
        let qkv_bias = vb.pp("qkv_proj").get(hidden_size * 3, "bias")?;

        let qkv_linear = Linear::new(qkv_weight, Some(qkv_bias), None);

        let o_proj_weight = vb.pp("o_proj").get((hidden_size, hidden_size), "weight")?;
        let o_proj_bias = vb.pp("o_proj").get(hidden_size, "bias")?;

        let o_proj = Linear::new(o_proj_weight, Some(o_proj_bias), None);

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

        Ok(Self {
            qkv_linear,
            o_proj,
            num_attention_heads,
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
        new_qkv_shape.push(self.num_attention_heads * 3);
        new_qkv_shape.push(self.attention_head_size);

        let qkv = qkv.reshape(new_qkv_shape)?;

        // Split qkv tensor
        let q = qkv.narrow(1, 0, self.num_attention_heads)?;
        let k = qkv.narrow(1, self.num_attention_heads, self.num_attention_heads)?;
        let v = qkv.narrow(1, self.num_attention_heads * 2, self.num_attention_heads)?;

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
            false,
            None,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        self.o_proj.forward(&attention)
    }
}

pub struct GTELayer {
    attention: GTEAttention,
    mlp: GTEMLP,
    attention_layer_norm: LayerNorm,
    mlp_layer_norm: LayerNorm,

    span: tracing::Span,
}

impl GTELayer {
    pub fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let attention = GTEAttention::load(vb.pp("attention"), config)?;
        let mlp = GTEMLP::load(vb.pp("mlp"), config)?;

        let attention_layer_norm =
            LayerNorm::load(vb.pp("attn_ln"), config.hidden_size, config.layer_norm_eps)?;
        let mlp_layer_norm =
            LayerNorm::load(vb.pp("mlp_ln"), config.hidden_size, config.layer_norm_eps)?;

        Ok(Self {
            attention,
            mlp,
            attention_layer_norm,
            mlp_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
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
        let attn_output = self
            .attention
            .forward(&hidden_states, cu_seqlens, cos, sin, max_s)?;
        let normed_attn_res_output = self
            .attention_layer_norm
            .forward(&attn_output, Some(hidden_states))?;

        let mlp_output = self.mlp.forward(&normed_attn_res_output)?;
        let normed_mlp_res_output = self
            .mlp_layer_norm
            .forward(&mlp_output, Some(&normed_attn_res_output))?;
        Ok(normed_mlp_res_output)
    }
}

pub struct FlashGTEModel {
    word_embeddings: Embedding,
    token_type_embeddings: Option<Embedding>,
    layers: Vec<GTELayer>,
    embeddings_norm: LayerNorm,
    cos_cache: Tensor,
    sin_cache: Tensor,
    classifier: Option<GTEClassificationHead>,
    pool: Pool,
    pub device: Device,

    span: tracing::Span,
}

impl FlashGTEModel {
    pub fn load(vb: VarBuilder, config: &GTEConfig, model_type: ModelType) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashGTE requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashGTE requires DType::F16")
        }

        if config.logn_attention_clip1 {
            candle::bail!("`logn_attention_clip1` is not supported");
        }
        if config.logn_attention_scale {
            candle::bail!("`logn_attention_scale` is not supported");
        }

        if config.position_embedding_type != PositionEmbeddingType::Rope {
            candle::bail!("Only `PositionEmbeddingType::Rope` is supported");
        }

        let (pool, classifier) = match model_type {
            ModelType::Classifier => {
                let pool = Pool::Cls;

                let classifier = GTEClassificationHead::load(vb.clone(), config)?;
                (pool, Some(classifier))
            }
            ModelType::Embedding(pool) => (pool, None),
        };

        let word_embeddings = Embedding::new(
            vb.pp("embeddings.word_embeddings")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );

        let token_type_embeddings = if config.type_vocab_size > 0 {
            Some(Embedding::new(
                vb.pp("embeddings.token_type_embeddings")
                    .get((config.type_vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ))
        } else {
            None
        };

        let layers = (0..config.num_hidden_layers)
            .map(|index| GTELayer::load(vb.pp(format!("encoder.layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let embeddings_norm = LayerNorm::load(
            vb.pp("embeddings.LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps,
        )?;

        let inv_freqs = get_inv_freqs(
            layers[0].attention.attention_head_size,
            config.rope_theta,
            vb.device(),
            config.rope_scaling.as_ref(),
        )?;

        let (cos_cache, sin_cache) = get_cos_sin(
            config.max_position_embeddings,
            &inv_freqs,
            vb.dtype(),
            false,
        )?;

        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            layers,
            embeddings_norm,
            cos_cache,
            sin_cache,
            classifier,
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
        let token_type_ids = Tensor::from_vec(batch.token_type_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        let word_embeddings = self.word_embeddings.forward(&input_ids)?;
        let token_type_embeddings = self
            .token_type_embeddings
            .as_ref()
            .map(|emb| emb.forward(&token_type_ids))
            .transpose()?;

        let mut hidden_states = self
            .embeddings_norm
            .forward(&word_embeddings, token_type_embeddings.as_ref())?;

        let cos = self.cos_cache.index_select(&position_ids, 0)?;
        let sin = self.sin_cache.index_select(&position_ids, 0)?;

        for layer in &self.layers {
            let h = layer.forward(
                &hidden_states,
                &cu_seqlens,
                &cos,
                &sin,
                batch.max_length as usize,
            )?;
            hidden_states = h;
        }

        let outputs = hidden_states;

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

impl Model for FlashGTEModel {
    fn is_padded(&self) -> bool {
        false
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        match &self.classifier {
            None => candle::bail!("`predict` is not implemented for this model"),
            Some(classifier) => {
                let (pooled_embeddings, _raw_embeddings) = self.forward(batch)?;
                let pooled_embeddings =
                    pooled_embeddings.expect("pooled_embeddings is empty. This is a bug.");
                classifier.forward(&pooled_embeddings)
            }
        }
    }
}
