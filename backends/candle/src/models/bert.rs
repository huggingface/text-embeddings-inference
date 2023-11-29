use crate::layers::{get_cublas_lt_wrapper, HiddenAct, LayerNorm, Linear};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>,
    pub model_type: Option<String>,
    pub id2label: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
    Alibi,
}

#[derive(Debug)]
struct BertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertEmbeddings {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        if config.position_embedding_type != PositionEmbeddingType::Absolute {
            candle::bail!("Bert only supports absolute position embeddings");
        }

        Ok(Self {
            word_embeddings: Embedding::new(
                vb.pp("word_embeddings")
                    .get((config.vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ),
            token_type_embeddings: Embedding::new(
                vb.pp("token_type_embeddings")
                    .get((config.type_vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ),
            position_embeddings: Embedding::new(
                vb.pp("position_embeddings").get(
                    (config.max_position_embeddings, config.hidden_size),
                    "weight",
                )?,
                config.hidden_size,
            ),
            layer_norm: LayerNorm::load(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps as f32,
            )?,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let position_embeddings = self.position_embeddings.forward(position_ids)?;

        let embeddings = input_embeddings.add(&token_type_embeddings)?;
        let embeddings = self.layer_norm.forward(&embeddings, &position_embeddings)?;

        Ok(embeddings)
    }
}

struct BertAttention {
    qkv_linear: Linear,

    dense: Linear,
    layer_norm: LayerNorm,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f64,

    span: tracing::Span,
}

impl BertAttention {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
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

        let softmax_scale = 1. / (attention_head_size as f64).sqrt();

        Ok(Self {
            qkv_linear,
            dense,
            layer_norm,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            softmax_scale,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_bias: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let device = hidden_states.device();

        let residual = hidden_states.clone();

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
        let hidden_states = self.layer_norm.forward(&hidden_states, &residual)?;

        Ok(hidden_states)
    }
}

struct BertLayer {
    attention: BertAttention,
    intermediate: Linear,
    output: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertLayer {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = BertAttention::load(vb.pp("attention"), config)?;

        let intermediate_weight = vb
            .pp("intermediate")
            .pp("dense")
            .get((config.intermediate_size, config.hidden_size), "weight")?;
        let intermediate_bias = vb
            .pp("intermediate")
            .pp("dense")
            .get(config.intermediate_size, "bias")?;
        let intermediate = Linear::new(
            intermediate_weight,
            Some(intermediate_bias),
            Some(config.hidden_act.clone()),
        );

        let output_weight = vb
            .pp("output")
            .pp("dense")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let output_bias = vb
            .pp("output")
            .pp("dense")
            .get(config.hidden_size, "bias")?;
        let output = Linear::new(output_weight, Some(output_bias), None);

        let layer_norm = LayerNorm::load(
            vb.pp("output").pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        Ok(Self {
            attention,
            intermediate,
            output,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.attention.forward(hidden_states, attention_bias)?;
        let residual = hidden_states.clone();

        let hidden_states = self.intermediate.forward(&hidden_states)?;
        let hidden_states = self.output.forward(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(&hidden_states, &residual)?;

        Ok(hidden_states)
    }
}

struct BertEncoder {
    layers: Vec<BertLayer>,
    span: tracing::Span,
}

impl BertEncoder {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(BertEncoder { layers, span })
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

struct BertClassificationHead {
    intermediate: Linear,
    output: Linear,
    span: tracing::Span,
}

impl BertClassificationHead {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let n_classes = match &config.id2label {
            None => candle::bail!("`id2label` must be set for classifier models"),
            Some(id2label) => id2label.len(),
        };

        let intermediate_weight = vb
            .pp("dense")
            .get((config.hidden_size, config.hidden_size), "weight")?;
        let intermediate_bias = vb.pp("dense").get(config.hidden_size, "bias")?;
        let intermediate = Linear::new(intermediate_weight, Some(intermediate_bias), None);

        let output_weight = vb
            .pp("out_proj")
            .get((n_classes, config.hidden_size), "weight")?;
        let output_bias = vb.pp("out_proj").get(n_classes, "bias")?;
        let output = Linear::new(output_weight, Some(output_bias), None);

        Ok(Self {
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "classifier"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.intermediate.forward(hidden_states)?;
        let hidden_states = hidden_states.tanh()?;
        let hidden_states = self.output.forward(&hidden_states)?;

        Ok(hidden_states)
    }
}

pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pool: Pool,
    classifier: Option<BertClassificationHead>,

    num_attention_heads: usize,

    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl BertModel {
    pub fn load(vb: VarBuilder, config: &Config, model_type: ModelType) -> Result<Self> {
        // Check position embedding type
        if config.position_embedding_type != PositionEmbeddingType::Absolute {
            candle::bail!("Bert only supports absolute position embeddings")
        }

        let (pool, classifier) = match model_type {
            // Classifier models always use CLS pooling
            ModelType::Classifier => {
                if config.model_type == Some("bert".to_string()) {
                    candle::bail!("`classifier` model type is not supported for Bert");
                }
                (
                    Pool::Cls,
                    Some(BertClassificationHead::load(vb.pp("classifier"), config)?),
                )
            }
            ModelType::Embedding(pool) => (pool, None),
        };

        // Check pool type
        if pool != Pool::Mean && pool != Pool::Cls {
            candle::bail!("Pool type {pool:?} is not supported");
        }

        let (embeddings, encoder) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config),
            BertEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                let model_type = config.model_type.clone().unwrap_or("bert".to_string());

                if let (Ok(embeddings), Ok(encoder)) = (
                    BertEmbeddings::load(vb.pp(format!("{model_type}.embeddings")), config),
                    BertEncoder::load(vb.pp(format!("{model_type}.encoder")), config),
                ) {
                    (embeddings, encoder)
                } else if let (Ok(embeddings), Ok(encoder)) = (
                    BertEmbeddings::load(vb.pp("roberta.embeddings"), config),
                    BertEncoder::load(vb.pp("roberta.encoder"), config),
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
            classifier,
            num_attention_heads: config.num_attention_heads,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<Tensor> {
        let _enter = self.span.enter();

        let batch_size = batch.cumulative_seq_lengths.len() - 1;
        let max_length = batch.max_length as usize;

        let shape = (batch_size, max_length);

        let (input_ids, type_ids, position_ids, input_lengths, attention_bias, attention_mask) =
            if batch_size > 1 {
                // Prepare padded batch
                let elems = batch_size * max_length;

                let mut input_ids = Vec::with_capacity(elems);
                let mut type_ids = Vec::with_capacity(elems);
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
                        type_ids.push(batch.token_type_ids[j]);
                        position_ids.push(batch.position_ids[j]);
                        attention_mask.push(1.0);
                        attention_bias.push(0.0);
                    }

                    // Add padding if needed
                    let padding = batch.max_length - seq_length;
                    if padding > 0 {
                        // Set bool to use attention mask
                        masking = true;
                        for _ in 0..padding {
                            input_ids.push(0);
                            type_ids.push(0);
                            position_ids.push(0);
                            attention_mask.push(0.0);
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
                    type_ids,
                    position_ids,
                    input_lengths,
                    attention_bias,
                    attention_mask,
                )
            } else {
                (
                    batch.input_ids,
                    batch.token_type_ids,
                    batch.position_ids,
                    vec![batch.max_length as f32],
                    None,
                    None,
                )
            };

        // Create CPU tensors
        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let type_ids = Tensor::from_vec(type_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;
        let input_lengths =
            Tensor::from_vec(input_lengths, (batch_size, 1), &self.device)?.to_dtype(self.dtype)?;

        let embedding_output = self
            .embeddings
            .forward(&input_ids, &type_ids, &position_ids)?;

        let mut outputs = self
            .encoder
            .forward(&embedding_output, attention_bias.as_ref())?;

        let results = match self.pool {
            // CLS pooling
            Pool::Cls => outputs.i((.., 0))?,
            // Mean pooling
            Pool::Mean => {
                if let Some(attention_mask) = attention_mask {
                    // Mask padded values
                    outputs = outputs.broadcast_mul(&attention_mask)?;
                }

                (outputs.sum(1)?.broadcast_div(&input_lengths))?
            }
        };

        Ok(results)
    }
}

impl Model for BertModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<Tensor> {
        self.forward(batch)
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        match &self.classifier {
            None => candle::bail!("`predict` is not implemented for this model"),
            Some(classifier) => {
                let hidden_states = self.forward(batch)?;
                classifier.forward(&hidden_states)
            }
        }
    }
}
