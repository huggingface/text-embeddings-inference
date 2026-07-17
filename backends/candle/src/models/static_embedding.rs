use crate::models::Model;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct StaticEmbeddingConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
}

#[derive(Debug)]
pub struct StaticEmbedding {
    embedding: Embedding,

    span: tracing::Span,
}

impl StaticEmbedding {
    pub fn load(
        vb: VarBuilder,
        config: &StaticEmbeddingConfig,
        weight_name: String,
    ) -> Result<Self> {
        Ok(Self {
            embedding: Embedding::new(
                vb.get((config.vocab_size, config.hidden_size), &weight_name)?,
                config.hidden_size,
            ),
            span: tracing::span!(tracing::Level::TRACE, "embedding"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        self.embedding.forward(input_ids)
    }
}

pub struct StaticEmbeddingModel {
    pool: Pool,
    embedding: StaticEmbedding,

    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl StaticEmbeddingModel {
    pub fn load(
        vb: VarBuilder,
        config: &StaticEmbeddingConfig,
        model_type: ModelType,
    ) -> Result<Self> {
        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`Classifier` model type is not supported for Static models")
            }
            ModelType::Embedding(pool) => pool,
        };

        let embedding = StaticEmbedding::load(vb.pp("embedding"), config, "weight".to_string())
            .or_else(|_| StaticEmbedding::load(vb.clone(), config, "embeddings".to_string()))?;

        Ok(Self {
            pool,
            embedding,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let shape = (batch_size, max_length);

        let (input_ids, input_lengths, attention_mask) = if batch_size > 1 {
            // Prepare padded batch
            let elems = batch_size * max_length;

            let mut input_ids = Vec::with_capacity(elems);
            let mut attention_mask = Vec::with_capacity(elems);
            let mut input_lengths = Vec::with_capacity(batch_size);
            // Bool to know if we need to use the attention mask
            let mut masking = false;

            for i in 0..batch_size {
                let start = batch.cumulative_seq_lengths[i] as usize;
                let end = batch.cumulative_seq_lengths[i + 1] as usize;
                let seq_length = (end - start) as u32;
                input_lengths.push(seq_length as f32);

                // Copy values
                for j in start..end {
                    input_ids.push(batch.input_ids[j]);
                    attention_mask.push(1.0_f32);
                }

                // Add padding if needed
                let padding = batch.max_length - seq_length;
                if padding > 0 {
                    // Set bool to use attention mask
                    masking = true;
                    for _ in 0..padding {
                        input_ids.push(0);
                        attention_mask.push(0.0_f32);
                    }
                }
            }

            let attention_mask = match masking {
                true => {
                    // We only need the mask if we use mean pooling
                    // For CLS pooling, the bias is enough
                    if self.pool == Pool::Mean {
                        let attention_mask = Tensor::from_vec(
                            attention_mask,
                            (batch_size, max_length, 1),
                            &self.device,
                        )?
                        .to_dtype(self.dtype)?;

                        Some(attention_mask)
                    } else {
                        None
                    }
                }
                false => None,
            };

            (input_ids, input_lengths, attention_mask)
        } else {
            (batch.input_ids, vec![batch.max_length as f32], None)
        };

        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let mut input_lengths =
            Tensor::from_vec(input_lengths, (batch_size, 1), &self.device)?.to_dtype(self.dtype)?;

        let outputs = self.embedding.forward(&input_ids)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            let pooled_indices_length = batch.pooled_indices.len();
            let mut outputs = outputs.clone();

            // Only use pooled_indices if at least one member of the batch ask for raw embeddings
            let pooled_indices = if has_raw_requests {
                let pooled_indices =
                    Tensor::from_vec(batch.pooled_indices, pooled_indices_length, &self.device)?;

                // Select values in the batch
                outputs = outputs.index_select(&pooled_indices, 0)?;
                Some(pooled_indices)
            } else {
                None
            };

            let pooled_embeddings = match self.pool {
                // CLS pooling
                Pool::Cls => outputs.i((.., 0))?,
                // Mean pooling
                Pool::Mean => {
                    if let Some(ref attention_mask) = attention_mask {
                        let mut attention_mask = attention_mask.clone();

                        if let Some(pooled_indices) = pooled_indices {
                            // Select values in the batch
                            attention_mask = attention_mask.index_select(&pooled_indices, 0)?;
                            input_lengths = input_lengths.index_select(&pooled_indices, 0)?;
                        };

                        // Mask padded values
                        outputs = outputs.broadcast_mul(&attention_mask)?;
                    }

                    (outputs.sum(1)?.broadcast_div(&input_lengths))?
                }
                // Last token and splade pooling are not supported for this model
                Pool::LastToken | Pool::Splade => unreachable!(),
            };
            Some(pooled_embeddings)
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            // Reshape outputs
            let (b, l, h) = outputs.shape().dims3()?;
            let outputs = outputs.reshape((b * l, h))?;

            // We need to remove the padding tokens only if batch_size > 1 and there are some
            // member of the batch that require pooling
            // or if batch_size > 1 and the members of the batch have different lengths
            if (attention_mask.is_some() || has_pooling_requests) && batch_size > 1 {
                let mut final_indices: Vec<u32> = Vec::with_capacity(batch_size * max_length);

                for i in batch.raw_indices.into_iter() {
                    let start = i * batch.max_length;
                    let i = i as usize;
                    let length =
                        batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i];

                    for j in start..start + length {
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

impl Model for StaticEmbeddingModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }

    fn predict(&self, _batch: Batch) -> Result<Tensor> {
        candle::bail!("`predict` is not implemented for this model")
    }
}
