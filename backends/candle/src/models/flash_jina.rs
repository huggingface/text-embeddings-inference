use crate::alibi::alibi_head_slopes;
use crate::flash_attn::flash_attn_varlen;
use crate::layers::{HiddenAct, LayerNorm, Linear};
use crate::models::bert::PositionEmbeddingType;
use crate::models::jina::JinaEmbeddings;
use crate::models::{BertConfig, Model};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct JinaAttention {
    qkv_linear: Linear,
    dense: Linear,
    layer_norm: LayerNorm,

    alibi_slopes: Option<Tensor>,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f32,

    span: tracing::Span,
}

impl JinaAttention {
    pub fn load(vb: VarBuilder, config: &BertConfig, alibi_slopes: Option<Tensor>) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let hidden_size = config.hidden_size;

        let query_weight = vb
            .pp("self.query")
            .get((all_head_size, hidden_size), "weight")?;
        let query_bias = vb.pp("self.query").get(all_head_size, "bias")?;
        let key_weight = vb
            .pp("self.key")
            .get((all_head_size, hidden_size), "weight")?;
        let key_bias = vb.pp("self.key").get(all_head_size, "bias")?;
        let value_weight = vb
            .pp("self.value")
            .get((all_head_size, hidden_size), "weight")?;
        let value_bias = vb.pp("self.value").get(all_head_size, "bias")?;

        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;
        let qkv_bias = Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?;

        let qkv_linear = Linear::new(qkv_weight, Some(qkv_bias), None);

        let dense_weight = vb
            .pp("output")
            .pp("dense")
            .get((hidden_size, hidden_size), "weight")?;
        let dense_bias = vb.pp("output").pp("dense").get(hidden_size, "bias")?;

        let dense = Linear::new(dense_weight, Some(dense_bias), None);

        let layer_norm = LayerNorm::load(
            vb.pp("output").pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

        Ok(Self {
            qkv_linear,
            dense,
            layer_norm,
            alibi_slopes,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            softmax_scale,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
        max_s: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let residual = hidden_states.clone();

        let qkv = self.qkv_linear.forward(hidden_states)?;

        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads * 3);
        new_qkv_shape.push(self.attention_head_size);

        let qkv = qkv.reshape(new_qkv_shape.as_slice())?;
        let qkv = qkv.chunk(3, 1)?;

        let attention = flash_attn_varlen(
            &qkv[0],
            &qkv[1],
            &qkv[2],
            self.alibi_slopes.as_ref(),
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            self.softmax_scale,
            false,
            None,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        let hidden_states = self.dense.forward(&attention)?;
        let hidden_states = self.layer_norm.forward(&hidden_states, Some(&residual))?;

        Ok(hidden_states)
    }
}

struct JinaBertLayer {
    attention: JinaAttention,
    gated: Linear,
    output: Linear,
    layer_norm: LayerNorm,
    act: HiddenAct,

    intermediate_size: usize,

    span: tracing::Span,
}

impl JinaBertLayer {
    pub fn load(vb: VarBuilder, config: &BertConfig, alibi: Option<Tensor>) -> Result<Self> {
        let attention = JinaAttention::load(vb.pp("attention"), config, alibi)?;

        let gated_weight = vb
            .pp("mlp")
            .pp("gated_layers")
            .get((config.intermediate_size * 2, config.hidden_size), "weight")?;
        let gated = Linear::new(gated_weight, None, None);

        let output_weight = vb
            .pp("mlp")
            .pp("wo")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let output_bias = vb.pp("mlp").pp("wo").get(config.hidden_size, "bias")?;
        let output = Linear::new(output_weight, Some(output_bias), None);

        let layer_norm = LayerNorm::load(
            vb.pp("mlp").pp("layernorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        Ok(Self {
            attention,
            gated,
            output,
            layer_norm,
            act: config.hidden_act.clone(),
            intermediate_size: config.intermediate_size,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
        max_s: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.attention.forward(hidden_states, cu_seqlens, max_s)?;
        let residual = hidden_states.clone();

        let hidden_states = self.gated.forward(&hidden_states)?;
        let gated = hidden_states.narrow(1, 0, self.intermediate_size)?;
        let gated = match self.act {
            HiddenAct::Gelu => gated.gelu(),
            HiddenAct::Relu => gated.relu(),
            HiddenAct::Swiglu => gated.silu(),
        }?;

        let non_gated = hidden_states.narrow(1, self.intermediate_size, self.intermediate_size)?;
        let hidden_states = (gated * non_gated)?;

        let hidden_states = self.output.forward(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(&hidden_states, Some(&residual))?;

        Ok(hidden_states)
    }
}

struct JinaBertEncoder {
    layers: Vec<JinaBertLayer>,
    span: tracing::Span,
}

impl JinaBertEncoder {
    pub fn load(vb: VarBuilder, config: &BertConfig, alibi: Option<Tensor>) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                JinaBertLayer::load(vb.pp(format!("layer.{index}")), config, alibi.clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(JinaBertEncoder { layers, span })
    }

    fn forward(&self, hidden_states: &Tensor, cu_seqlens: &Tensor, max_s: usize) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, cu_seqlens, max_s)?
        }

        Ok(hidden_states)
    }
}

pub struct FlashJinaBertModel {
    embeddings: JinaEmbeddings,
    encoder: JinaBertEncoder,
    pool: Pool,
    pub device: Device,

    span: tracing::Span,
}

impl FlashJinaBertModel {
    pub fn load(vb: VarBuilder, config: &BertConfig, model_type: ModelType) -> Result<Self> {
        let alibi = match config.position_embedding_type {
            PositionEmbeddingType::Alibi => {
                let alibi_slopes = alibi_head_slopes(config.num_attention_heads);
                Some(
                    Tensor::from_vec(alibi_slopes, config.num_attention_heads, vb.device())?
                        .to_dtype(DType::F32)?,
                )
            }
            PositionEmbeddingType::Absolute => None,
            _ => candle::bail!("not supported"),
        };

        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashJinaBertModel requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashJinaBertModel requires DType::F16")
        }

        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Jina")
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("`splade` is not supported for Jina")
                }
                pool
            }
        };

        let (embeddings, encoder) = match (
            JinaEmbeddings::load(vb.pp("embeddings"), config),
            JinaBertEncoder::load(vb.pp("encoder"), config, alibi.clone()),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let (Ok(embeddings), Ok(encoder)) = (
                    JinaEmbeddings::load(vb.pp("bert.embeddings"), config),
                    JinaBertEncoder::load(vb.pp("bert.encoder"), config, alibi.clone()),
                ) {
                    (embeddings, encoder)
                } else {
                    return Err(err);
                }
            }
        };

        Ok(Self {
            embeddings,
            encoder,
            pool,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let shape = batch.input_ids.len();

        // Create Cuda tensors
        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let type_ids = Tensor::from_vec(batch.token_type_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        let embedding_output = self
            .embeddings
            .forward(&input_ids, &type_ids, &position_ids)?;

        let outputs =
            self.encoder
                .forward(&embedding_output, &cu_seqlens, batch.max_length as usize)?;

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

impl Model for FlashJinaBertModel {
    fn is_padded(&self) -> bool {
        false
    }
    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
