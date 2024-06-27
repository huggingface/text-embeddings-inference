use crate::layers::{get_cublas_lt_wrapper, HiddenAct, LayerNorm, Linear};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DistilBertConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub hidden_dim: usize,
    pub activation: HiddenAct,
    pub max_position_embeddings: usize,
    pub pad_token_id: usize,
    pub model_type: Option<String>,
}

#[derive(Debug)]
pub struct DistilBertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DistilBertEmbeddings {
    pub fn load(vb: VarBuilder, config: &DistilBertConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(
                vb.pp("word_embeddings")
                    .get((config.vocab_size, config.dim), "weight")?,
                config.dim,
            ),
            position_embeddings: Embedding::new(
                vb.pp("position_embeddings")
                    .get((config.max_position_embeddings, config.dim), "weight")?,
                config.dim,
            ),
            layer_norm: LayerNorm::load(vb.pp("LayerNorm"), config.dim, 1e-12f32)?,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(position_ids)?;

        let embeddings = self
            .layer_norm
            .forward(&input_embeddings, Some(&position_embeddings))?;

        Ok(embeddings)
    }
}

#[derive(Debug)]
struct DistilBertAttention {
    qkv_linear: Linear,
    dense: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f64,

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

        let softmax_scale = 1. / (attention_head_size as f64).sqrt();

        Ok(Self {
            qkv_linear,
            dense,
            num_attention_heads: config.n_heads,
            attention_head_size,
            softmax_scale,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_bias: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let device = hidden_states.device();

        let qkv = self.qkv_linear.forward(hidden_states)?;

        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads * 3);
        new_qkv_shape.push(self.attention_head_size);
        let qkv = qkv.reshape(new_qkv_shape.as_slice())?.transpose(1, 2)?;

        let qkv = qkv.chunk(3, 1)?;
        let query_layer = &qkv[0].contiguous()?;
        let key_layer = &qkv[1].contiguous()?;
        let value_layer = &qkv[2];

        #[allow(unused_variables)]
        let context_layer = if let (Device::Cuda(_), Some(cublaslt)) =
            (device, get_cublas_lt_wrapper())
        {
            #[cfg(feature = "cuda")]
            {
                // cuBLASLt batch matmul implementation requires inputs to be dims3
                let (batch_size, _, seq_len, _) = key_layer.shape().dims4()?;
                let key_layer = key_layer.flatten(0, 1)?;
                let query_layer = query_layer.flatten(0, 1)?;
                let value_layer = value_layer.flatten(0, 1)?;
                let attention_bias = attention_bias.map(|mask| mask.flatten(0, 1)).transpose()?;

                // If attention_bias is set, we fuse the add by giving it as the output matrix
                // and setting beta to 1.0
                let beta = match attention_bias.is_some() {
                    true => Some(1.0),
                    false => None,
                };

                // Batch matrix multiplication
                // Fuse softmax scale and attention_bias add
                let attention_scores = cublaslt.batch_matmul(
                    &key_layer,
                    &query_layer,
                    attention_bias.as_ref(),
                    Some(self.softmax_scale as f32),
                    beta,
                    None,
                    None,
                )?;
                let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

                let context_layer = cublaslt.batch_matmul(
                    &value_layer.t()?.contiguous()?,
                    &attention_probs,
                    // We save one allocation
                    Some(&query_layer),
                    None,
                    None,
                    None,
                    None,
                )?;

                // Reshape to dims4
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
            let attention_scores = query_layer.matmul(&key_layer.t()?)?;
            let mut attention_scores = (attention_scores * self.softmax_scale)?;

            if let Some(attention_bias) = attention_bias {
                attention_scores = attention_scores.add(attention_bias)?;
            }

            let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
            attention_probs.matmul(&value_layer.contiguous()?)
        }?;

        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;

        let hidden_states = self.dense.forward(&context_layer)?;

        Ok(hidden_states)
    }
}

#[derive(Debug)]
pub struct DistilBertMLP {
    lin1: Linear,
    lin2: Linear,

    span: tracing::Span,
}

impl DistilBertMLP {
    pub fn load(vb: VarBuilder, config: &DistilBertConfig) -> Result<Self> {
        let lin1_weight = vb
            .pp("lin1")
            .get((config.hidden_dim, config.dim), "weight")?;
        let lin1_bias = vb.pp("lin1").get(config.hidden_dim, "bias")?;
        let lin1 = Linear::new(
            lin1_weight,
            Some(lin1_bias),
            Some(config.activation.clone()),
        );

        let lin2_weight = vb
            .pp("lin2")
            .get((config.dim, config.hidden_dim), "weight")?;
        let lin2_bias = vb.pp("lin2").get(config.dim, "bias")?;
        let lin2 = Linear::new(lin2_weight, Some(lin2_bias), None);

        Ok(Self {
            lin1,
            lin2,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.lin1.forward(hidden_states)?;
        self.lin2.forward(&hidden_states)
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
        attention_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let attn_output = self.attention.forward(hidden_states, attention_bias)?;
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

    fn forward(&self, hidden_states: &Tensor, attention_bias: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_bias)?;
        }

        Ok(hidden_states)
    }
}

#[derive(Debug)]
pub struct DistilBertSpladeHead {
    vocab_transform: Linear,
    vocab_projector: Linear,
    vocab_layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DistilBertSpladeHead {
    pub(crate) fn load(vb: VarBuilder, config: &DistilBertConfig) -> Result<Self> {
        let vocab_transform_weight = vb
            .pp("vocab_transform")
            .get((config.dim, config.dim), "weight")?;
        let vocab_transform_bias = vb.pp("vocab_transform").get(config.dim, "bias")?;
        let vocab_transform = Linear::new(
            vocab_transform_weight,
            Some(vocab_transform_bias),
            Some(config.activation.clone()),
        );

        let vocab_projector_weight = vb
            .pp("vocab_projector")
            .get((config.vocab_size, config.dim), "weight")?;
        let vocab_projector_bias = vb.pp("vocab_projector").get(config.vocab_size, "bias")?;
        let vocab_projector = Linear::new(
            vocab_projector_weight,
            Some(vocab_projector_bias),
            Some(HiddenAct::Relu),
        );

        let vocab_layer_norm = LayerNorm::load(vb.pp("vocab_layer_norm"), config.dim, 1e-12f32)?;

        Ok(Self {
            vocab_transform,
            vocab_projector,
            vocab_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "splade"),
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.vocab_transform.forward(hidden_states)?;
        let hidden_states = self.vocab_layer_norm.forward(&hidden_states, None)?;
        let hidden_states = self.vocab_projector.forward(&hidden_states)?;
        (1.0 + hidden_states)?.log()
    }
}

#[derive(Debug)]
pub struct DistilBertModel {
    embeddings: DistilBertEmbeddings,
    encoder: DistilBertEncoder,
    pool: Pool,
    splade: Option<DistilBertSpladeHead>,

    num_attention_heads: usize,

    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl DistilBertModel {
    pub fn load(vb: VarBuilder, config: &DistilBertConfig, model_type: ModelType) -> Result<Self> {
        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for DistilBert")
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::LastToken {
                    candle::bail!("`last_token` is not supported for DistilBert");
                }
                pool
            }
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
            num_attention_heads: config.n_heads,
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

        let (input_ids, position_ids, input_lengths, attention_bias, attention_mask) =
            if batch_size > 1 {
                // Prepare padded batch
                let elems = batch_size * max_length;

                let mut input_ids = Vec::with_capacity(elems);
                let mut position_ids = Vec::with_capacity(elems);
                let mut attention_mask = Vec::with_capacity(elems);
                let mut attention_bias = Vec::with_capacity(elems);
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
                        position_ids.push(batch.position_ids[j]);
                        attention_mask.push(1.0_f32);
                        attention_bias.push(0.0);
                    }

                    // Add padding if needed
                    let padding = batch.max_length - seq_length;
                    if padding > 0 {
                        // Set bool to use attention mask
                        masking = true;
                        for _ in 0..padding {
                            input_ids.push(0);
                            position_ids.push(0);
                            attention_mask.push(0.0_f32);
                            attention_bias.push(f32::NEG_INFINITY);
                        }
                    }
                }

                let (attention_bias, attention_mask) = match masking {
                    true => {
                        // We only need the mask if we use mean pooling
                        // For CLS pooling, the bias is enough
                        let attention_mask = if self.pool == Pool::Mean {
                            let attention_mask = Tensor::from_vec(
                                attention_mask,
                                (batch_size, max_length, 1),
                                &self.device,
                            )?
                            .to_dtype(self.dtype)?;

                            Some(attention_mask)
                        } else {
                            None
                        };

                        let attention_bias = Tensor::from_vec(
                            attention_bias,
                            (batch_size, 1, 1, max_length),
                            &self.device,
                        )?
                        .to_dtype(self.dtype)?;
                        // Broadcast once instead of at every layer
                        let attention_bias = attention_bias
                            .broadcast_as((
                                batch_size,
                                self.num_attention_heads,
                                max_length,
                                max_length,
                            ))?
                            .contiguous()?;
                        (Some(attention_bias), attention_mask)
                    }
                    false => (None, None),
                };

                (
                    input_ids,
                    position_ids,
                    input_lengths,
                    attention_bias,
                    attention_mask,
                )
            } else {
                (
                    batch.input_ids,
                    batch.position_ids,
                    vec![batch.max_length as f32],
                    None,
                    None,
                )
            };

        // Create CPU tensors
        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;
        let input_lengths =
            Tensor::from_vec(input_lengths, (batch_size, 1), &self.device)?.to_dtype(self.dtype)?;

        let embedding_output = self.embeddings.forward(&input_ids, &position_ids)?;

        let outputs = self
            .encoder
            .forward(&embedding_output, attention_bias.as_ref())?;

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
                // Last token pooling is not supported for this model
                Pool::LastToken => unreachable!(),
                // Mean pooling
                Pool::Mean => {
                    if let Some(ref attention_mask) = attention_mask {
                        let mut attention_mask = attention_mask.clone();

                        if let Some(pooled_indices) = pooled_indices {
                            // Select values in the batch
                            attention_mask = attention_mask.index_select(&pooled_indices, 0)?;
                        };

                        // Mask padded values
                        outputs = outputs.broadcast_mul(&attention_mask)?;
                    }

                    (outputs.sum(1)?.broadcast_div(&input_lengths))?
                }
                Pool::Splade => {
                    // Unwrap is safe here
                    let splade_head = self.splade.as_ref().unwrap();
                    let mut relu_log = splade_head.forward(&outputs)?;

                    if let Some(ref attention_mask) = attention_mask {
                        let mut attention_mask = attention_mask.clone();

                        if let Some(pooled_indices) = pooled_indices {
                            // Select values in the batch
                            attention_mask = attention_mask.index_select(&pooled_indices, 0)?;
                        };

                        // Mask padded values
                        relu_log = relu_log.broadcast_mul(&attention_mask)?;
                    }

                    relu_log.max(1)?
                }
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

impl Model for DistilBertModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
