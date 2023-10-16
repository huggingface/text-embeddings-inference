use crate::flash_attn::flash_attn_varlen;
use crate::models::bert::{Config, HiddenAct, PositionEmbeddingType};
use crate::models::EmbeddingModel;
use candle::{DType, Device, Result, Tensor};
use candle_cublaslt::{fused_matmul, Activation, CublasLt};
use candle_layer_norm::fused_add_layer_norm;
use candle_nn::{Embedding, Module, VarBuilder};
use text_embeddings_backend_core::{Batch, Pool};

#[derive(Debug)]
struct FastLayerNorm {
    weight: Tensor,
    bias: Tensor,
    epsilon: f32,
    span: tracing::Span,
}

impl FastLayerNorm {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get(config.hidden_size, "weight")
                .or_else(|_| vb.get(config.hidden_size, "gamma"))?,
            bias: vb
                .get(config.hidden_size, "bias")
                .or_else(|_| vb.get(config.hidden_size, "beta"))?,
            epsilon: config.layer_norm_eps as f32,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        fused_add_layer_norm(
            hidden_states,
            residual,
            &self.weight,
            &self.bias,
            self.epsilon,
        )
    }
}

#[derive(Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    act: Option<Activation>,
    cublaslt: CublasLt,
    span: tracing::Span,
}

impl Linear {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        act: Option<HiddenAct>,
        cublaslt: CublasLt,
    ) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "linear");

        let act = act.map(|a| match a {
            HiddenAct::Gelu => Activation::Gelu,
            HiddenAct::Relu => Activation::Relu,
        });

        Self {
            weight,
            bias,
            act,
            cublaslt,
            span,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        fused_matmul(
            &self.weight,
            x,
            self.bias.as_ref(),
            self.act.clone(),
            self.cublaslt.clone(),
        )
    }
}

#[derive(Debug)]
struct BertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: FastLayerNorm,
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
            layer_norm: FastLayerNorm::load(vb.pp("LayerNorm"), config)?,
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
        let embeddings = input_embeddings.add(&token_type_embeddings)?;

        let position_embeddings = self.position_embeddings.forward(position_ids)?;

        let embeddings = self.layer_norm.forward(&embeddings, &position_embeddings)?;

        Ok(embeddings)
    }
}

struct BertAttention {
    qkv_linear: Linear,
    dense: Linear,
    layer_norm: FastLayerNorm,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f32,

    span: tracing::Span,
}

impl BertAttention {
    pub fn load(vb: VarBuilder, config: &Config, cublaslt: CublasLt) -> Result<Self> {
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

        let qkv_linear = Linear::new(qkv_weight, Some(qkv_bias), None, cublaslt.clone());

        let dense_weight = vb
            .pp("output")
            .pp("dense")
            .get((hidden_size, hidden_size), "weight")?;
        let dense_bias = vb.pp("output").pp("dense").get(hidden_size, "bias")?;

        let dense = Linear::new(dense_weight, Some(dense_bias), None, cublaslt.clone());

        let layer_norm = FastLayerNorm::load(vb.pp("output").pp("LayerNorm"), config)?;

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

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
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            self.softmax_scale,
            false,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        let hidden_states = self.dense.forward(&attention)?;
        let hidden_states = self.layer_norm.forward(&hidden_states, &residual)?;

        Ok(hidden_states)
    }
}

struct BertLayer {
    attention: BertAttention,
    intermediate: Linear,
    output: Linear,
    layer_norm: FastLayerNorm,
    span: tracing::Span,
}

impl BertLayer {
    pub fn load(vb: VarBuilder, config: &Config, cublaslt: CublasLt) -> Result<Self> {
        let attention = BertAttention::load(vb.pp("attention"), config, cublaslt.clone())?;

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
            cublaslt.clone(),
        );

        let output_weight = vb
            .pp("output")
            .pp("dense")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let output_bias = vb
            .pp("output")
            .pp("dense")
            .get(config.hidden_size, "bias")?;
        let output = Linear::new(output_weight, Some(output_bias), None, cublaslt.clone());

        let layer_norm = FastLayerNorm::load(vb.pp("output").pp("LayerNorm"), config)?;

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
        cu_seqlens: &Tensor,
        max_s: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.attention.forward(hidden_states, cu_seqlens, max_s)?;
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
    pub fn load(vb: VarBuilder, config: &Config, cublaslt: CublasLt) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(vb.pp(format!("layer.{index}")), config, cublaslt.clone()))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(BertEncoder { layers, span })
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

pub struct FlashBertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pool: Pool,
    pub device: Device,

    span: tracing::Span,
}

impl FlashBertModel {
    pub fn load(vb: VarBuilder, config: &Config, pool: Pool) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashBert requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashBert requires DType::F16")
        }

        // Check position embedding type
        if config.position_embedding_type != PositionEmbeddingType::Absolute {
            candle::bail!("FlashBert only supports absolute position embeddings")
        }

        // Check pool type
        if pool != Pool::Mean && pool != Pool::Cls {
            candle::bail!("Pool type {pool:?} is not supported");
        }

        let cublaslt = CublasLt::new(vb.device()).unwrap();

        let (embeddings, encoder) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config),
            BertEncoder::load(vb.pp("encoder"), config, cublaslt.clone()),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                let model_type = config.model_type.clone().unwrap_or("bert".to_string());

                if let (Ok(embeddings), Ok(encoder)) = (
                    BertEmbeddings::load(vb.pp(format!("{model_type}.embeddings")), config),
                    BertEncoder::load(
                        vb.pp(format!("{model_type}.encoder")),
                        config,
                        cublaslt.clone(),
                    ),
                ) {
                    (embeddings, encoder)
                } else if let (Ok(embeddings), Ok(encoder)) = (
                    BertEmbeddings::load(vb.pp("bert.embeddings"), config),
                    BertEncoder::load(vb.pp("bert.encoder"), config, cublaslt.clone()),
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

        let results = match self.pool {
            // CLS pooling
            Pool::Cls => outputs.index_select(&cu_seqlens.narrow(0, 0, batch_size)?, 0)?,
            // Mean pooling
            Pool::Mean => {
                if batch_size > 1 {
                    // for each request
                    let results: Result<Vec<Tensor>> = (0..batch.cumulative_seq_lengths.len() - 1)
                        .map(|i| {
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
        };

        // Normalize
        let normalized_results = results.broadcast_div(&results.sqr()?.sum_keepdim(1)?.sqrt()?)?;

        Ok(normalized_results)
    }
}

impl EmbeddingModel for FlashBertModel {
    fn embed(&self, batch: Batch) -> Result<Tensor> {
        self.forward(batch)
    }
}
