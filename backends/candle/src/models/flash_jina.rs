use crate::alibi::alibi_head_slopes;
use crate::flash_attn::flash_attn_varlen;
use crate::layers::{HiddenAct, LayerNorm, Linear};
use crate::models::bert::{Config, PositionEmbeddingType};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug)]
struct BertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertEmbeddings {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let position_embeddings =
            if config.position_embedding_type == PositionEmbeddingType::Absolute {
                Some(Embedding::new(
                    vb.pp("position_embeddings").get(
                        (config.max_position_embeddings, config.hidden_size),
                        "weight",
                    )?,
                    config.hidden_size,
                ))
            } else {
                None
            };

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
            position_embeddings,
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

        if let Some(position_embeddings) = &self.position_embeddings {
            let position_embeddings = position_embeddings.forward(position_ids)?;
            let embeddings = input_embeddings.add(&token_type_embeddings)?;
            self.layer_norm.forward(&embeddings, &position_embeddings)
        } else {
            self.layer_norm
                .forward(&input_embeddings, &token_type_embeddings)
        }
    }
}

struct AlibiBertAttention {
    qkv_linear: Linear,
    dense: Linear,
    layer_norm: LayerNorm,

    alibi_slopes: Option<Tensor>,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f32,

    span: tracing::Span,
}

impl AlibiBertAttention {
    pub fn load(vb: VarBuilder, config: &Config, alibi_slopes: Option<Tensor>) -> Result<Self> {
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
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        let hidden_states = self.dense.forward(&attention)?;
        let hidden_states = self.layer_norm.forward(&hidden_states, &residual)?;

        Ok(hidden_states)
    }
}

struct JinaBertLayer {
    attention: AlibiBertAttention,
    gated: Linear,
    output: Linear,
    layer_norm: LayerNorm,
    act: HiddenAct,

    intermediate_size: usize,

    span: tracing::Span,
}

impl JinaBertLayer {
    pub fn load(vb: VarBuilder, config: &Config, alibi: Option<Tensor>) -> Result<Self> {
        let attention = AlibiBertAttention::load(vb.pp("attention"), config, alibi)?;

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
        let gated = hidden_states.i((.., 0..self.intermediate_size))?;
        let gated = match self.act {
            HiddenAct::Gelu => gated.gelu(),
            HiddenAct::Relu => gated.relu(),
        }?;

        let non_gated = hidden_states.i((.., self.intermediate_size..))?;
        let hidden_states = (gated * non_gated)?;

        let hidden_states = self.output.forward(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(&hidden_states, &residual)?;

        Ok(hidden_states)
    }
}

struct BertEncoder {
    layers: Vec<JinaBertLayer>,
    span: tracing::Span,
}

impl BertEncoder {
    pub fn load(vb: VarBuilder, config: &Config, alibi: Option<Tensor>) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                JinaBertLayer::load(vb.pp(format!("layer.{index}")), config, alibi.clone())
            })
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

pub struct FlashJinaBertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pool: Pool,
    pub device: Device,

    span: tracing::Span,
}

impl FlashJinaBertModel {
    pub fn load(vb: VarBuilder, config: &Config, model_type: ModelType) -> Result<Self> {
        let alibi = match config.position_embedding_type {
            PositionEmbeddingType::Alibi => {
                let alibi_slopes = alibi_head_slopes(config.num_attention_heads);
                Some(
                    Tensor::from_vec(alibi_slopes, config.num_attention_heads, vb.device())?
                        .to_dtype(DType::F32)?,
                )
            }
            PositionEmbeddingType::Absolute => None,
        };

        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashJinaBertModel requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashJinaBertModel requires DType::F16")
        }

        let pool = match model_type {
            // Classifier models always use CLS pooling
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Jina")
            }
            ModelType::Embedding(pool) => pool,
        };

        let (embeddings, encoder) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config),
            BertEncoder::load(vb.pp("encoder"), config, alibi.clone()),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                let model_type = config.model_type.clone().unwrap_or("bert".to_string());

                if let (Ok(embeddings), Ok(encoder)) = (
                    BertEmbeddings::load(vb.pp(format!("{model_type}.embeddings")), config),
                    BertEncoder::load(
                        vb.pp(format!("{model_type}.encoder")),
                        config,
                        alibi.clone(),
                    ),
                ) {
                    (embeddings, encoder)
                } else if let (Ok(embeddings), Ok(encoder)) = (
                    BertEmbeddings::load(vb.pp("bert.embeddings"), config),
                    BertEncoder::load(vb.pp("bert.encoder"), config, alibi.clone()),
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

        Ok(results)
    }
}

impl Model for FlashJinaBertModel {
    fn is_padded(&self) -> bool {
        false
    }
    fn embed(&self, batch: Batch) -> Result<Tensor> {
        self.forward(batch)
    }
}
