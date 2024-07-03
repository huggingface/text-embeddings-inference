use crate::alibi::build_alibi_tensor;
use crate::layers::{get_cublas_lt_wrapper, HiddenAct, LayerNorm, Linear};
use crate::models::jina::JinaEmbeddings;
use crate::models::PositionEmbeddingType;
use crate::models::{BertConfig, Model};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct JinaCodeAttention {
    qkv_linear: Linear,

    dense: Linear,
    layer_norm_q: LayerNorm,
    layer_norm_k: LayerNorm,
    layer_norm_out: LayerNorm,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f64,

    span: tracing::Span,
}

impl JinaCodeAttention {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
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

        let layer_norm_q = LayerNorm::load(
            vb.pp("self").pp("layer_norm_q"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;
        let layer_norm_k = LayerNorm::load(
            vb.pp("self").pp("layer_norm_k"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        let dense_weight = vb
            .pp("output")
            .pp("dense")
            .get((hidden_size, hidden_size), "weight")?;
        let dense_bias = vb.pp("output").pp("dense").get(hidden_size, "bias")?;

        let dense = Linear::new(dense_weight, Some(dense_bias), None);

        let layer_norm_out = LayerNorm::load(
            vb.pp("output").pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        let softmax_scale = 1. / (attention_head_size as f64).sqrt();

        Ok(Self {
            qkv_linear,
            dense,
            layer_norm_q,
            layer_norm_k,
            layer_norm_out,
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

        let qkv = qkv.reshape(new_qkv_shape.as_slice())?;

        // Split heads
        let qkv = qkv.chunk(3, 2)?;

        // Flatten last dims again to go through the layer norm
        let query_layer = &qkv[0].flatten_from(D::Minus2)?;
        let key_layer = &qkv[1].flatten_from(D::Minus2)?;

        // Layer norm on q and k
        let query_layer = self.layer_norm_q.forward(query_layer, None)?;
        let key_layer = self.layer_norm_k.forward(key_layer, None)?;

        let mut new_qk_shape = query_layer.dims().to_vec();
        new_qk_shape.pop();
        new_qk_shape.push(self.num_attention_heads);
        new_qk_shape.push(self.attention_head_size);

        let query_layer = query_layer
            .reshape(new_qk_shape.as_slice())?
            .transpose(1, 2)?
            .contiguous()?;
        let key_layer = key_layer
            .reshape(new_qk_shape.as_slice())?
            .transpose(1, 2)?
            .contiguous()?;
        let value_layer = &qkv[2].transpose(1, 2)?.contiguous()?;

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
        let hidden_states = self
            .layer_norm_out
            .forward(&hidden_states, Some(&residual))?;

        Ok(hidden_states)
    }
}

struct JinaCodeBertLayer {
    attention: JinaCodeAttention,
    up_gated_layer: Linear,
    down_layer: Linear,
    layer_norm_1: LayerNorm,
    layer_norm_2: LayerNorm,
    act: HiddenAct,

    intermediate_size: usize,

    span: tracing::Span,
}

impl JinaCodeBertLayer {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let attention = JinaCodeAttention::load(vb.pp("attention"), config)?;

        let up_gated_weight = vb
            .pp("mlp")
            .pp("up_gated_layer")
            .get((config.intermediate_size * 2, config.hidden_size), "weight")?;
        let up_gated_layer = Linear::new(up_gated_weight, None, None);

        let down_weight = vb
            .pp("mlp")
            .pp("down_layer")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let down_bias = vb
            .pp("mlp")
            .pp("down_layer")
            .get(config.hidden_size, "bias")?;
        let down_layer = Linear::new(down_weight, Some(down_bias), None);

        let layer_norm_1 = LayerNorm::load(
            vb.pp("layer_norm_1"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;
        let layer_norm_2 = LayerNorm::load(
            vb.pp("layer_norm_2"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        Ok(Self {
            attention,
            up_gated_layer,
            down_layer,
            layer_norm_1,
            layer_norm_2,
            act: config.hidden_act.clone(),
            intermediate_size: config.intermediate_size,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        // Pre-Norm
        let residual = hidden_states.clone();

        // Self-Attention block
        let hidden_states = self.attention.forward(hidden_states, attention_bias)?;

        // Pre-MLP LayerNorm
        let hidden_states = self.layer_norm_1.forward(&hidden_states, Some(&residual))?;

        // MLP block
        let residual = hidden_states.clone();
        let hidden_states = self.up_gated_layer.forward(&hidden_states)?;
        let non_gated = hidden_states.i((.., .., 0..self.intermediate_size))?;
        let gated = hidden_states.i((.., .., self.intermediate_size..))?;
        let gated = match self.act {
            HiddenAct::Gelu => gated.gelu(),
            HiddenAct::Relu => gated.relu(),
            HiddenAct::Swiglu => gated.silu(),
        }?;
        let hidden_states = (non_gated * gated)?;
        let hidden_states = self.down_layer.forward(&hidden_states)?;

        // Post-MLP LayerNorm
        let hidden_states = self.layer_norm_2.forward(&hidden_states, Some(&residual))?;

        Ok(hidden_states)
    }
}

struct JinaCodeBertEncoder {
    layers: Vec<JinaCodeBertLayer>,
    span: tracing::Span,
}

impl JinaCodeBertEncoder {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| JinaCodeBertLayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(JinaCodeBertEncoder { layers, span })
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

pub struct JinaCodeBertModel {
    embeddings: JinaEmbeddings,
    encoder: JinaCodeBertEncoder,
    pool: Pool,
    alibi: Option<Tensor>,

    num_attention_heads: usize,

    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl JinaCodeBertModel {
    pub fn load(vb: VarBuilder, config: &BertConfig, model_type: ModelType) -> Result<Self> {
        let alibi = match config.position_embedding_type {
            PositionEmbeddingType::Alibi => Some(build_alibi_tensor(
                config.max_position_embeddings,
                config.num_attention_heads,
                vb.device(),
                vb.dtype(),
            )?),
            PositionEmbeddingType::Absolute => None,
            _ => candle::bail!("not supported"),
        };

        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for JinaCode")
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("`splade` is not supported for JinaCode")
                }
                if pool == Pool::LastToken {
                    candle::bail!("`last_token` is not supported for JinaCode");
                }
                pool
            }
        };

        let (embeddings, encoder) = match (
            JinaEmbeddings::load(vb.pp("embeddings"), config),
            JinaCodeBertEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let (Ok(embeddings), Ok(encoder)) = (
                    JinaEmbeddings::load(vb.pp("bert.embeddings"), config),
                    JinaCodeBertEncoder::load(vb.pp("bert.encoder"), config),
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
            alibi,
            num_attention_heads: config.num_attention_heads,
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
                            type_ids.push(0);
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
                        let mut attention_bias = attention_bias.broadcast_as((
                            batch_size,
                            self.num_attention_heads,
                            max_length,
                            max_length,
                        ))?;

                        // Add alibi tensor
                        if let Some(alibi) = &self.alibi {
                            let alibi = alibi
                                .i((.., .., 0..max_length, 0..max_length))?
                                .broadcast_as((
                                    batch_size,
                                    self.num_attention_heads,
                                    max_length,
                                    max_length,
                                ))?;

                            attention_bias = attention_bias.add(&alibi)?;
                        }

                        (Some(attention_bias.contiguous()?), attention_mask)
                    }
                    false => {
                        if let Some(alibi) = &self.alibi {
                            (
                                Some(
                                    alibi
                                        .i((.., .., 0..max_length, 0..max_length))?
                                        .broadcast_as((
                                            batch_size,
                                            self.num_attention_heads,
                                            max_length,
                                            max_length,
                                        ))?
                                        .contiguous()?,
                                ),
                                None,
                            )
                        } else {
                            (None, None)
                        }
                    }
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
                let attention_bias = if let Some(alibi) = &self.alibi {
                    Some(
                        alibi
                            .i((.., .., 0..max_length, 0..max_length))?
                            .contiguous()?,
                    )
                } else {
                    None
                };

                (
                    batch.input_ids,
                    batch.token_type_ids,
                    batch.position_ids,
                    vec![batch.max_length as f32],
                    attention_bias,
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

impl Model for JinaCodeBertModel {
    fn is_padded(&self) -> bool {
        true
    }
    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
