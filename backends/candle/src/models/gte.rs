use crate::layers::{
    apply_rotary, get_cos_sin, get_cublas_lt_wrapper, get_inv_freqs, HiddenAct, LayerNorm, Linear,
    RopeScaling,
};
use crate::models::{Model, PositionEmbeddingType};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GTEConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_type: String,
    pub layer_norm_eps: f32,
    pub position_embedding_type: PositionEmbeddingType,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub logn_attention_scale: bool,
    #[serde(default)]
    pub logn_attention_clip1: bool,
    pub id2label: Option<HashMap<String, String>>,
}

struct GTEAttention {
    qkv_linear: Linear,
    o_proj: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,

    softmax_scale: f64,

    span: tracing::Span,
}

impl GTEAttention {
    pub fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let hidden_size = config.hidden_size;

        let qkv_weight = vb
            .pp("qkv_proj")
            .get((hidden_size * 3, hidden_size), "weight")?;
        let qkv_bias = vb.pp("qkv_proj").get(hidden_size * 3, "bias")?;

        let qkv_linear = Linear::new(qkv_weight, Some(qkv_bias), None);

        let o_proj_weight = vb.pp("o_proj").get((hidden_size, hidden_size), "weight")?;
        let o_proj_bias = vb.pp("o_proj").get(hidden_size, "bias")?;

        let o_proj = Linear::new(o_proj_weight, Some(o_proj_bias), None);

        let softmax_scale = 1. / (attention_head_size as f64).sqrt();

        Ok(Self {
            qkv_linear,
            o_proj,
            num_attention_heads,
            attention_head_size,
            softmax_scale,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
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

        let query_layer = apply_rotary(query_layer, cos, sin, self.attention_head_size)?;
        let key_layer = apply_rotary(key_layer, cos, sin, self.attention_head_size)?;

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

        let hidden_states = self.o_proj.forward(&context_layer)?;

        Ok(hidden_states)
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct GTEMLP {
    up_gate_proj: Linear,
    down_proj: Linear,

    act: HiddenAct,
    intermediate_size: usize,

    span: tracing::Span,
}

impl GTEMLP {
    pub fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let intermediate_size = config.intermediate_size;

        let up_gate_proj_weight = vb
            .pp("up_gate_proj")
            .get((intermediate_size * 2, config.hidden_size), "weight")?;

        let up_gate_proj = Linear::new(up_gate_proj_weight, None, None);

        let down_proj_weight = vb
            .pp("down_proj")
            .get((config.hidden_size, intermediate_size), "weight")?;
        let down_proj_bias = vb.pp("down_proj").get(config.hidden_size, "bias")?;
        let down_proj = Linear::new(down_proj_weight, Some(down_proj_bias), None);

        Ok(Self {
            up_gate_proj,
            down_proj,
            intermediate_size,
            act: config.hidden_act.clone(),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let up_gate_states = self.up_gate_proj.forward(hidden_states)?;
        let up_states = up_gate_states.narrow(D::Minus1, 0, self.intermediate_size)?;
        let gate_states =
            up_gate_states.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;

        let gate_states = match self.act {
            HiddenAct::Gelu => gate_states.gelu(),
            HiddenAct::Relu => gate_states.relu(),
            HiddenAct::Swiglu => gate_states.silu(),
        }?;

        self.down_proj.forward(&(gate_states * up_states)?)
    }
}

pub struct GTELayer {
    attention: GTEAttention,
    mlp: GTEMLP,
    attention_layer_norm: LayerNorm,
    mlp_layer_norm: LayerNorm,

    span: tracing::Span,
}

impl GTELayer {
    pub fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let attention = GTEAttention::load(vb.pp("attention"), config)?;
        let mlp = GTEMLP::load(vb.pp("mlp"), config)?;

        let attention_layer_norm =
            LayerNorm::load(vb.pp("attn_ln"), config.hidden_size, config.layer_norm_eps)?;
        let mlp_layer_norm =
            LayerNorm::load(vb.pp("mlp_ln"), config.hidden_size, config.layer_norm_eps)?;

        Ok(Self {
            attention,
            mlp,
            attention_layer_norm,
            mlp_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attn_output = self
            .attention
            .forward(hidden_states, attention_bias, cos, sin)?;

        let normed_attn_res_output = self
            .attention_layer_norm
            .forward(&attn_output, Some(hidden_states))?;

        let mlp_output = self.mlp.forward(&normed_attn_res_output)?;
        let normed_mlp_res_output = self
            .mlp_layer_norm
            .forward(&mlp_output, Some(&normed_attn_res_output))?;
        Ok(normed_mlp_res_output)
    }
}

pub struct GTEClassificationHead {
    pooler: Option<Linear>,
    classifier: Linear,
    span: tracing::Span,
}

impl GTEClassificationHead {
    #[allow(dead_code)]
    pub(crate) fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let n_classes = match &config.id2label {
            None => candle::bail!("`id2label` must be set for classifier models"),
            Some(id2label) => id2label.len(),
        };

        let pooler = if let Ok(pooler_weight) = vb
            .pp("pooler.dense")
            .get((config.hidden_size, config.hidden_size), "weight")
        {
            let pooler_bias = vb.pp("pooler.dense").get(config.hidden_size, "bias")?;
            Some(Linear::new(pooler_weight, Some(pooler_bias), None))
        } else {
            None
        };

        let classifier_weight = vb
            .pp("classifier")
            .get((n_classes, config.hidden_size), "weight")?;
        let classifier_bias = vb.pp("classifier").get(n_classes, "bias")?;
        let classifier = Linear::new(classifier_weight, Some(classifier_bias), None);

        Ok(Self {
            classifier,
            pooler,
            span: tracing::span!(tracing::Level::TRACE, "classifier"),
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.unsqueeze(1)?;
        if let Some(pooler) = self.pooler.as_ref() {
            hidden_states = pooler.forward(&hidden_states)?;
            hidden_states = hidden_states.tanh()?;
        }

        let hidden_states = self.classifier.forward(&hidden_states)?;
        let hidden_states = hidden_states.squeeze(1)?;
        Ok(hidden_states)
    }
}

struct GTEEncoder {
    layers: Vec<GTELayer>,
    span: tracing::Span,
}

impl GTEEncoder {
    pub fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| GTELayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(GTEEncoder { layers, span })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_bias, cos, sin)?
        }

        Ok(hidden_states)
    }
}

pub struct GTEModel {
    word_embeddings: Embedding,
    token_type_embeddings: Option<Embedding>,
    embeddings_norm: LayerNorm,
    encoder: GTEEncoder,
    dtype: DType,
    rotary_cache: (Tensor, Tensor),
    rotary_dim: usize,
    classifier: Option<GTEClassificationHead>,
    pool: Pool,
    pub device: Device,

    num_attention_heads: usize,

    span: tracing::Span,
}

impl GTEModel {
    pub fn load(vb: VarBuilder, config: &GTEConfig, model_type: ModelType) -> Result<Self> {
        if config.logn_attention_clip1 {
            candle::bail!("`logn_attention_clip1` is not supported");
        }
        if config.logn_attention_scale {
            candle::bail!("`logn_attention_scale` is not supported");
        }

        if config.position_embedding_type != PositionEmbeddingType::Rope {
            candle::bail!("Only `PositionEmbeddingType::Rope` is supported");
        }

        let (pool, classifier) = match model_type {
            ModelType::Classifier => {
                let pool = Pool::Cls;

                let classifier = GTEClassificationHead::load(vb.clone(), config)?;
                (pool, Some(classifier))
            }
            ModelType::Embedding(pool) => (pool, None),
        };

        let word_embeddings = Embedding::new(
            vb.pp("embeddings.word_embeddings")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );

        let token_type_embeddings = if config.type_vocab_size > 0 {
            Some(Embedding::new(
                vb.pp("embeddings.token_type_embeddings")
                    .get((config.type_vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ))
        } else {
            None
        };

        let encoder = GTEEncoder::load(vb.pp("encoder"), config)?;

        let embeddings_norm = LayerNorm::load(
            vb.pp("embeddings.LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps,
        )?;

        let rotary_dim = encoder.layers[0].attention.attention_head_size;
        let inv_freqs = get_inv_freqs(
            rotary_dim,
            config.rope_theta,
            vb.device(),
            config.rope_scaling.as_ref(),
        )?;

        let rotary_cache =
            get_cos_sin(config.max_position_embeddings, &inv_freqs, vb.dtype(), true)?;

        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            encoder,
            embeddings_norm,
            rotary_cache,
            classifier,
            pool,
            num_attention_heads: config.num_attention_heads,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            rotary_dim,
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
        let position_ids = Tensor::from_vec(position_ids, batch_size * max_length, &self.device)?;
        let input_lengths =
            Tensor::from_vec(input_lengths, (batch_size, 1), &self.device)?.to_dtype(self.dtype)?;

        let cos = self.rotary_cache.0.index_select(&position_ids, 0)?;
        let sin = self.rotary_cache.1.index_select(&position_ids, 0)?;

        let cos = cos.reshape((batch_size, 1, max_length, self.rotary_dim))?;
        let sin = sin.reshape((batch_size, 1, max_length, self.rotary_dim))?;

        let word_embeddings = self.word_embeddings.forward(&input_ids)?;
        let token_type_embeddings = self
            .token_type_embeddings
            .as_ref()
            .map(|emb| emb.forward(&type_ids))
            .transpose()?;

        let embedding_output = self
            .embeddings_norm
            .forward(&word_embeddings, token_type_embeddings.as_ref())?;

        let outputs =
            self.encoder
                .forward(&embedding_output, attention_bias.as_ref(), &cos, &sin)?;

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

impl Model for GTEModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        match &self.classifier {
            None => candle::bail!("`predict` is not implemented for this model"),
            Some(classifier) => {
                let (pooled_embeddings, _raw_embeddings) = self.forward(batch)?;
                let pooled_embeddings =
                    pooled_embeddings.expect("pooled_embeddings is empty. This is a bug.");
                classifier.forward(&pooled_embeddings)
            }
        }
    }
}
