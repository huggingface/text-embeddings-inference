use crate::models::EmbeddingModel;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use text_embeddings_backend_core::{Batch, Pool};

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

#[derive(Debug, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    Relu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
}

#[derive(Debug)]
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    epsilon: f64,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get(config.hidden_size, "weight")
                .or_else(|_| vb.get(config.hidden_size, "gamma"))?,
            bias: vb
                .get(config.hidden_size, "bias")
                .or_else(|_| vb.get(config.hidden_size, "beta"))?,
            epsilon: config.layer_norm_eps,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.epsilon)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}

#[derive(Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    act: Option<HiddenAct>,
    span: tracing::Span,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>, act: Option<HiddenAct>) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "linear");

        Self {
            weight,
            bias,
            act,
            span,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let x = x.matmul(&self.weight)?;
        let x = match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }?;
        if let Some(act) = &self.act {
            match act {
                HiddenAct::Gelu => x.gelu(),
                HiddenAct::Relu => x.relu(),
            }
        } else {
            Ok(x)
        }
    }
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
            candle::bail!("FlashBert only supports absolute position embeddings");
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
            layer_norm: LayerNorm::load(vb.pp("LayerNorm"), config)?,
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

        let embeddings = input_embeddings
            .add(&token_type_embeddings)?
            .add(&position_embeddings)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;

        Ok(embeddings)
    }
}

struct BertAttention {
    q_linear: Linear,
    k_linear: Linear,
    v_linear: Linear,

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
        let q_linear = Linear::new(query_weight.t()?, Some(query_bias), None);

        let key_weight = vb
            .pp("self.key")
            .get((all_head_size, hidden_size), "weight")?;
        let key_bias = vb.pp("self.key").get(all_head_size, "bias")?;
        let k_linear = Linear::new(key_weight.t()?, Some(key_bias), None);

        let value_weight = vb
            .pp("self.value")
            .get((all_head_size, hidden_size), "weight")?;
        let value_bias = vb.pp("self.value").get(all_head_size, "bias")?;
        let v_linear = Linear::new(value_weight.t()?, Some(value_bias), None);

        let dense_weight = vb
            .pp("output")
            .pp("dense")
            .get((hidden_size, hidden_size), "weight")?;
        let dense_bias = vb.pp("output").pp("dense").get(hidden_size, "bias")?;

        let dense = Linear::new(dense_weight.t()?, Some(dense_bias), None);

        let layer_norm = LayerNorm::load(vb.pp("output").pp("LayerNorm"), config)?;

        let softmax_scale = 1. / (attention_head_size as f64).sqrt();

        Ok(Self {
            q_linear,
            k_linear,
            v_linear,
            dense,
            layer_norm,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            softmax_scale,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs
            .reshape(new_x_shape.as_slice())?
            .unsqueeze(0)?
            .transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let residual = hidden_states.clone();

        let query_layer = self.q_linear.forward(hidden_states)?;
        let key_layer = self.k_linear.forward(hidden_states)?;
        let value_layer = self.v_linear.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores * self.softmax_scale)?;
        let attention_probs = softmax(&attention_scores, D::Minus1)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(D::Minus2)?.squeeze(0)?;

        let hidden_states = self.dense.forward(&context_layer)?.add(&residual)?;
        let hidden_states = self.layer_norm.forward(&hidden_states)?;

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
            intermediate_weight.t()?,
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
        let output = Linear::new(output_weight.t()?, Some(output_bias), None);

        let layer_norm = LayerNorm::load(vb.pp("output").pp("LayerNorm"), config)?;

        Ok(Self {
            attention,
            intermediate,
            output,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.attention.forward(hidden_states)?;
        let residual = hidden_states.clone();

        let hidden_states = self.intermediate.forward(&hidden_states)?;
        let hidden_states = self.output.forward(&hidden_states)?.add(&residual)?;
        let hidden_states = self.layer_norm.forward(&hidden_states)?;

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

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states)?
        }

        Ok(hidden_states)
    }
}

pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pool: Pool,
    pub device: Device,

    span: tracing::Span,
}

impl BertModel {
    pub fn load(vb: VarBuilder, config: &Config, pool: Pool) -> Result<Self> {
        match vb.device() {
            Device::Cpu => {}
            _ => candle::bail!("Bert requires CPU"),
        }

        // Check position embedding type
        if config.position_embedding_type != PositionEmbeddingType::Absolute {
            candle::bail!("FlashBert only supports absolute position embeddings")
        }

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
                    BertEmbeddings::load(vb.pp("bert.embeddings"), config),
                    BertEncoder::load(vb.pp("bert.encoder"), config),
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

    pub fn forward(&self, batch: Batch) -> Result<Tensor> {
        let _enter = self.span.enter();

        let batch_size = batch.cumulative_seq_lengths.len() - 1;
        if batch_size > 1 {
            candle::bail!("Bert CPU only support batch_size == 1");
        }
        let shape = batch.input_ids.len();

        // Create Cuda tensors
        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let type_ids = Tensor::from_vec(batch.token_type_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;

        let embedding_output = self
            .embeddings
            .forward(&input_ids, &type_ids, &position_ids)?;

        let outputs = self.encoder.forward(&embedding_output)?;

        let results = match self.pool {
            // CLS pooling
            Pool::Cls => outputs.i(0..1)?,
            // Mean pooling
            Pool::Mean => (outputs.sum_keepdim(0)? / (batch.max_length as f64))?,
        };

        // Normalize
        let normalized_results = results.broadcast_div(&results.sqr()?.sum_keepdim(1)?.sqrt()?)?;

        Ok(normalized_results)
    }
}

impl EmbeddingModel for BertModel {
    fn embed(&self, batch: Batch) -> Result<Tensor> {
        self.forward(batch)
    }
}
