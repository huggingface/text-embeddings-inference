use crate::flash_attn::flash_attn_varlen;
use crate::layers::{LayerNorm, Linear};
use crate::models::distilbert::{
    DistilBertConfig, DistilBertEmbeddings, DistilBertMLP, DistilBertSpladeHead,
};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug)]
struct DistilBertAttention {
    qkv_linear: Linear,
    dense: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f32,

    span: tracing::Span,
}

impl DistilBertAttention {
    pub fn load(vb: VarBuilder, config: &DistilBertConfig) -> Result<Self> {
        let attention_head_size = config.dim / config.n_heads;
        let all_head_size = config.n_heads * attention_head_size;
        let hidden_size = config.dim;

        let query_weight = vb.pp("q_lin").get((all_head_size, hidden_size), "weight")?;
        let query_bias = vb.pp("q_lin").get(all_head_size, "bias")?;
        let key_weight = vb.pp("k_lin").get((all_head_size, hidden_size), "weight")?;
        let key_bias = vb.pp("k_lin").get(all_head_size, "bias")?;
        let value_weight = vb.pp("v_lin").get((all_head_size, hidden_size), "weight")?;
        let value_bias = vb.pp("v_lin").get(all_head_size, "bias")?;

        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;
        let qkv_bias = Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?;

        let qkv_linear = Linear::new(qkv_weight, Some(qkv_bias), None);

        let dense_weight = vb.pp("out_lin").get((hidden_size, hidden_size), "weight")?;
        let dense_bias = vb.pp("out_lin").get(hidden_size, "bias")?;

        let dense = Linear::new(dense_weight, Some(dense_bias), None);

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

        Ok(Self {
            qkv_linear,
            dense,
            num_attention_heads: config.n_heads,
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

        let hidden_states = self.dense.forward(&attention)?;

        Ok(hidden_states)
    }
}

#[derive(Debug)]
struct DistilBertBlock {
    attention: DistilBertAttention,
    mlp: DistilBertMLP,
    post_attention_layer_norm: LayerNorm,
    output_layer_norm: LayerNorm,

    span: tracing::Span,
}

impl DistilBertBlock {
    pub fn load(vb: VarBuilder, config: &DistilBertConfig) -> Result<Self> {
        let attention = DistilBertAttention::load(vb.pp("attention"), config)?;
        let mlp = DistilBertMLP::load(vb.pp("ffn"), config)?;

        let post_attention_layer_norm =
            LayerNorm::load(vb.pp("sa_layer_norm"), config.dim, 1e-12f32)?;
        let output_layer_norm = LayerNorm::load(vb.pp("output_layer_norm"), config.dim, 1e-12f32)?;

        Ok(Self {
            attention,
            mlp,
            post_attention_layer_norm,
            output_layer_norm,
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

        let attn_output = self.attention.forward(hidden_states, cu_seqlens, max_s)?;
        let hidden_states = self
            .post_attention_layer_norm
            .forward(hidden_states, Some(&attn_output))?;

        let mlp_out = self.mlp.forward(&hidden_states)?;

        self.output_layer_norm
            .forward(&hidden_states, Some(&mlp_out))
    }
}

#[derive(Debug)]
struct DistilBertEncoder {
    layers: Vec<DistilBertBlock>,
    span: tracing::Span,
}

impl DistilBertEncoder {
    pub fn load(vb: VarBuilder, config: &DistilBertConfig) -> Result<Self> {
        let layers = (0..config.n_layers)
            .map(|index| DistilBertBlock::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(DistilBertEncoder { layers, span })
    }

    fn forward(&self, hidden_states: &Tensor, cu_seqlens: &Tensor, max_s: usize) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, cu_seqlens, max_s)?;
        }

        Ok(hidden_states)
    }
}

pub struct FlashDistilBertModel {
    embeddings: DistilBertEmbeddings,
    encoder: DistilBertEncoder,
    pool: Pool,
    splade: Option<DistilBertSpladeHead>,

    pub device: Device,

    span: tracing::Span,
}

impl FlashDistilBertModel {
    pub fn load(vb: VarBuilder, config: &DistilBertConfig, model_type: ModelType) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashDistilBert requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashDistilBert requires DType::F16")
        }

        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for DistilBert")
            }
            ModelType::Embedding(pool) => pool,
        };

        let (embeddings, encoder) = match (
            DistilBertEmbeddings::load(vb.pp("embeddings"), config),
            DistilBertEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let (Ok(embeddings), Ok(encoder)) = (
                    DistilBertEmbeddings::load(vb.pp("distilbert.embeddings"), config),
                    DistilBertEncoder::load(vb.pp("distilbert.transformer"), config),
                ) {
                    (embeddings, encoder)
                } else {
                    return Err(err);
                }
            }
        };

        let splade = if pool == Pool::Splade {
            Some(DistilBertSpladeHead::load(vb.clone(), config)?)
        } else {
            None
        };

        Ok(Self {
            embeddings,
            encoder,
            pool,
            splade,
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
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        let embedding_output = self.embeddings.forward(&input_ids, &position_ids)?;

        let outputs =
            self.encoder
                .forward(&embedding_output, &cu_seqlens, batch.max_length as usize)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            let pooled_embeddings = match self.pool {
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
                        outputs.index_select(&indices, 0)?
                    } else {
                        match self.pool {
                            Pool::Cls => outputs.i(0)?,
                            Pool::LastToken => {
                                outputs.i(batch.cumulative_seq_lengths[1] as usize - 1)?
                            }
                            _ => unreachable!(),
                        }
                        .unsqueeze(0)?
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
                        Tensor::cat(&results?, 0)?
                    } else {
                        (outputs.sum_keepdim(0)? / (batch.max_length as f64))?
                    }
                }
                Pool::Splade => {
                    // Unwrap is safe here
                    let splade_head = self.splade.as_ref().unwrap();
                    let relu_log = splade_head.forward(&outputs)?;

                    if batch_size > 1 {
                        // for each request that requires pooling
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .into_iter()
                            .map(|i| {
                                let i = i as usize;
                                let start = batch.cumulative_seq_lengths[i];
                                let len = batch.cumulative_seq_lengths[i + 1] - start;

                                relu_log
                                    .narrow(0, start as usize, len as usize)?
                                    .max_keepdim(0)
                            })
                            .collect();

                        // Concatenate all results
                        Tensor::cat(&results?, 0)?
                    } else {
                        relu_log.max_keepdim(0)?
                    }
                }
            };
            Some(pooled_embeddings)
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

impl Model for FlashDistilBertModel {
    fn is_padded(&self) -> bool {
        false
    }
    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
