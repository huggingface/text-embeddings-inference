use crate::layers::{get_cublas_lt_wrapper, HiddenAct, LayerNorm, Linear};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Module, Result, Shape, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpnet/configuration_mpnet.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MPNetConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub relative_attention_num_buckets: usize,
}

#[derive(Debug)]
pub struct MPNetEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl MPNetEmbeddings {
    pub fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(
                vb.pp("word_embeddings")
                    .get((config.vocab_size, config.hidden_size), "weight")?,
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

struct MPNetAttention {
    qkv_linear: Linear,
    dense: Linear,
    layer_norm: LayerNorm,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f64,

    span: tracing::Span,
}

impl MPNetAttention {
    pub fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let hidden_size = config.hidden_size;

        let query_weight = vb
            .pp("attn.q")
            .get((all_head_size, hidden_size), "weight")?;
        let query_bias = vb.pp("attn.q").get(all_head_size, "bias")?;

        let key_weight = vb
            .pp("attn.k")
            .get((all_head_size, hidden_size), "weight")?;
        let key_bias = vb.pp("attn.k").get(all_head_size, "bias")?;

        let value_weight = vb
            .pp("attn.v")
            .get((all_head_size, hidden_size), "weight")?;
        let value_bias = vb.pp("attn.v").get(all_head_size, "bias")?;

        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;
        let qkv_bias = Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?;

        let qkv_linear = Linear::new(qkv_weight, Some(qkv_bias), None);

        let dense_weight = vb.pp("attn.o").get((hidden_size, hidden_size), "weight")?;
        let dense_bias = vb.pp("attn.o").get(hidden_size, "bias")?;

        let dense = Linear::new(dense_weight, Some(dense_bias), None);

        let layer_norm = LayerNorm::load(
            vb.pp("LayerNorm"),
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

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        attention_bias: &Tensor,
    ) -> Result<Tensor> {
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
        let value_layer = &qkv[2].contiguous()?;

        #[allow(unused_variables)]
        let context_layer =
            if let (Device::Cuda(_), Some(cublaslt)) = (device, get_cublas_lt_wrapper()) {
                #[cfg(feature = "cuda")]
                {
                    // cuBLASLt batch matmul implementation requires inputs to be dims3
                    let (batch_size, _, seq_len, _) = key_layer.shape().dims4()?;
                    let key_layer = key_layer.flatten(0, 1)?;
                    let query_layer = query_layer.flatten(0, 1)?;
                    let value_layer = value_layer.flatten(0, 1)?;
                    let bias = (attention_bias + attention_mask)?.flatten(0, 1)?;

                    // Batch matrix multiplication
                    // Fuse softmax scale and attention_bias add
                    let attention_scores = cublaslt.batch_matmul(
                        &key_layer,
                        &query_layer,
                        Some(bias.as_ref()),
                        Some(self.softmax_scale as f32),
                        Some(1.0),
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
                let attention_scores = (attention_scores * self.softmax_scale)?;

                let attention_scores = attention_scores.add(attention_bias)?;
                let attention_scores = attention_scores.add(attention_mask)?;

                let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
                attention_probs.matmul(value_layer)
            }?;

        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(D::Minus2)?;

        let hidden_states = self.dense.forward(&context_layer)?;
        let hidden_states = self.layer_norm.forward(&hidden_states, Some(&residual))?;

        Ok(hidden_states)
    }
}

struct MPNetLayer {
    attention: MPNetAttention,
    intermediate: Linear,
    output: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl MPNetLayer {
    pub fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        let attention = MPNetAttention::load(vb.pp("attention"), config)?;

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
        attention_mask: &Tensor,
        attention_bias: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states =
            self.attention
                .forward(hidden_states, attention_mask, attention_bias)?;

        let residual = hidden_states.clone();

        let hidden_states = self.intermediate.forward(&hidden_states)?;
        let hidden_states = self.output.forward(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(&hidden_states, Some(&residual))?;

        Ok(hidden_states)
    }
}

struct MPNetAttentionBias {
    relative_attention_bias: Embedding,

    relative_attention_num_buckets: usize,

    span: tracing::Span,
}

impl MPNetAttentionBias {
    pub fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        let relative_attention_bias = Embedding::new(
            vb.pp("relative_attention_bias").get(
                (
                    config.relative_attention_num_buckets,
                    config.num_attention_heads,
                ),
                "weight",
            )?,
            config.num_attention_heads,
        );

        let span = tracing::span!(tracing::Level::TRACE, "relative_attention_bias");

        Ok(MPNetAttentionBias {
            relative_attention_bias,
            relative_attention_num_buckets: config.relative_attention_num_buckets,
            span,
        })
    }

    fn relative_position_bucket(
        &self,
        relative_position: &Tensor,
        max_distance: i64,
    ) -> Result<Tensor> {
        let device = relative_position.device();

        let num_buckets = (self.relative_attention_num_buckets / 2) as f64;
        let max_exact = num_buckets / 2.0;
        let max_distance_log = (max_distance as f64 / max_exact).ln();
        let scale = (num_buckets - max_exact) / max_distance_log;

        let mut ret = Tensor::zeros_like(relative_position)?;
        let n = relative_position.to_dtype(DType::F32)?.neg()?;

        ret = ret.add(&(&n.lt(0.0)?.to_dtype(DType::F32)? * num_buckets)?.to_dtype(DType::I64)?)?;
        let n = n.abs()?;

        let is_small = n.lt(max_exact)?;

        let log_val = (n.clone() / max_exact)?.log()?;
        let val_if_large = (max_exact + (log_val * scale)?)?;

        let val_if_large = val_if_large
            .minimum(&Tensor::full(
                (num_buckets - 1.0) as f32,
                val_if_large.shape(),
                device,
            )?)?
            .to_dtype(DType::I64)?;
        ret.add(&is_small.where_cond(&n.clone().to_dtype(DType::I64)?, &val_if_large)?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let bsz = x.dim(0)?;
        let qlen = x.dim(1)?;
        let klen = x.dim(1)?;

        let context_position = Tensor::arange(0_i64, qlen as i64, x.device())?.unsqueeze(1)?;
        let memory_position = Tensor::arange(0_i64, klen as i64, x.device())?.unsqueeze(0)?;

        let context_position = context_position.broadcast_as((qlen, klen))?;
        let memory_position = memory_position.broadcast_as((qlen, klen))?;

        let relative_position = memory_position.sub(&context_position)?;

        let rp_bucket = self.relative_position_bucket(&relative_position, 128)?;

        let values = self.relative_attention_bias.forward(&rp_bucket)?;
        let values = values.permute([2, 0, 1])?.unsqueeze(0)?;
        let values = values
            .expand(&[bsz, values.dim(1)?, qlen, klen])?
            .contiguous()?;
        Ok(values)
    }
}

struct MPNetEncoder {
    layers: Vec<MPNetLayer>,
    attention_bias: MPNetAttentionBias,
    span: tracing::Span,
}

impl MPNetEncoder {
    pub fn load(vb: VarBuilder, config: &MPNetConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| MPNetLayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let attention_bias = MPNetAttentionBias::load(vb, config)?;

        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(MPNetEncoder {
            layers,
            attention_bias,
            span,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        let attention_bias = self.attention_bias.forward(&hidden_states)?;

        let attention_mask = attention_mask.broadcast_as(attention_bias.shape())?;

        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, &attention_mask, &attention_bias)?;
        }

        Ok(hidden_states)
    }
}

pub struct MPNetModel {
    embeddings: MPNetEmbeddings,
    encoder: MPNetEncoder,
    pool: Pool,

    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl MPNetModel {
    pub fn load(vb: VarBuilder, config: &MPNetConfig, model_type: ModelType) -> Result<Self> {
        let pool = match model_type {
            // Classifier models always use CLS pooling
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for MPNet")
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("`splade` is not supported for MPNet")
                }
                pool
            }
        };

        let (embeddings, encoder) = match (
            MPNetEmbeddings::load(vb.pp("embeddings"), config),
            MPNetEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let (Ok(embeddings), Ok(encoder)) = (
                    MPNetEmbeddings::load(vb.pp("mpnet.embeddings".to_string()), config),
                    MPNetEncoder::load(vb.pp("mpnet.encoder".to_string()), config),
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
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    fn get_extended_attention_mask(
        &self,
        attention_mask: Option<&Tensor>,
        input_shape: &Shape,
    ) -> Result<Tensor> {
        let extended_attention_mask = if let Some(attention_mask) = attention_mask {
            attention_mask.squeeze(2)?
        } else {
            Tensor::ones(input_shape, DType::F32, &self.device)?
        }
        .unsqueeze(1)?
        .unsqueeze(1)?
        .to_dtype(self.dtype)?;

        let min_value = match self.dtype {
            DType::F32 => f32::MIN as f64,
            _ => -65504.0_f64, // f16 minumum value
        };

        let extended_attention_mask = ((1.0 - extended_attention_mask)? * min_value)?;

        Ok(extended_attention_mask)
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let shape = (batch_size, max_length);

        let (input_ids, position_ids, input_lengths, attention_mask) = if batch_size > 1 {
            // Prepare padded batch
            let elems = batch_size * max_length;

            let mut input_ids = Vec::with_capacity(elems);
            let mut position_ids = Vec::with_capacity(elems);
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
                    // `position_id` starts from `padding_idx` + 1, which is 2.
                    position_ids.push(batch.position_ids[j] + 2);
                    attention_mask.push(1.0_f32);
                }

                // Add padding if needed
                let padding = batch.max_length - seq_length;
                if padding > 0 {
                    // Set bool to use attention mask
                    masking = true;
                    for _ in 0..padding {
                        input_ids.push(1);
                        position_ids.push(1);
                        attention_mask.push(0.0_f32);
                    }
                }
            }

            let attention_mask = match masking {
                true => {
                    let attention_mask = Tensor::from_vec(
                        attention_mask,
                        (batch_size, max_length, 1),
                        &self.device,
                    )?
                    .to_dtype(self.dtype)?;

                    Some(attention_mask)
                }
                false => None,
            };

            (input_ids, position_ids, input_lengths, attention_mask)
        } else {
            // `position_id` starts from `padding_idx` + 1. So, we need to add 2 to the every position ids.
            let mut position_ids = batch.position_ids;
            for position_id in position_ids.iter_mut() {
                *position_id += 2;
            }

            (
                batch.input_ids,
                position_ids,
                vec![batch.max_length as f32],
                None,
            )
        };

        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;
        let mut input_lengths =
            Tensor::from_vec(input_lengths, (batch_size, 1), &self.device)?.to_dtype(self.dtype)?;

        let embedding_output = self.embeddings.forward(&input_ids, &position_ids)?;

        let extended_attention_mask =
            self.get_extended_attention_mask(attention_mask.as_ref(), input_ids.shape())?;

        let outputs = self
            .encoder
            .forward(&embedding_output, &extended_attention_mask)?;

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
                            input_lengths = input_lengths.index_select(&pooled_indices, 0)?;
                        };

                        // Mask padded values
                        outputs = outputs.broadcast_mul(&attention_mask)?;
                    }

                    (outputs.sum(1)?.broadcast_div(&input_lengths))?
                }
                Pool::Splade => unreachable!(),
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

impl Model for MPNetModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
