use crate::flash_attn::flash_attn_varlen;
use crate::layers::{LayerNorm, Linear};
use crate::models::modernbert::{
    ClassificationHead, ModernBertClassificationHead, ModernBertConfig, ModernBertEmbeddings,
    ModernBertMLP,
};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct ModernBertAttention {
    wqkv: Linear,
    wo: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f64,

    span: tracing::Span,
}

impl ModernBertAttention {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let hidden_size = config.hidden_size;

        let wqkv_weight = vb
            .pp("Wqkv")
            .get((hidden_size * 3, hidden_size), "weight")?;
        let wqkv_bias = if config.attention_bias {
            vb.pp("Wqkv").get(hidden_size * 3, "bias").ok()
        } else {
            None
        };
        let wqkv: Linear = Linear::new(wqkv_weight, wqkv_bias, None);

        let wo_weight = vb.pp("Wo").get((hidden_size, hidden_size), "weight")?;
        let wo_bias = if config.attention_bias {
            vb.pp("Wo").get(hidden_size, "bias").ok()
        } else {
            None
        };
        let wo = Linear::new(wo_weight, wo_bias, None);

        let softmax_scale = 1. / (attention_head_size as f64).sqrt();

        Ok(Self {
            wqkv,
            wo,
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
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let qkv = self.wqkv.forward(hidden_states)?;

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

        let attention = flash_attn_varlen(
            &query_layer,
            &key_layer,
            &value_layer,
            None,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            self.softmax_scale,
            false,
            self.local_attention,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        let hidden_states = self.wo.forward(&attention)?;

        Ok(hidden_states)
    }
}

struct ModernBertEncoderLayer {
    attn_norm: Option<LayerNorm>,
    attn: ModernBertAttention,
    mlp_norm: LayerNorm,
    mlp: ModernBertMLP,

    span: tracing::Span,
}

impl ModernBertEncoderLayer {
    pub fn load(vb: VarBuilder, index: usize, config: &ModernBertConfig) -> Result<Self> {
        let attn_norm = if index != 0 {
            Some(LayerNorm::load(
                vb.pp("attn_norm"),
                config.hidden_size,
                config.norm_eps as f32,
            )?)
        } else {
            None
        };

        let attn = ModernBertAttention::load(vb.pp("attn"), config)?;

        let mlp_norm = LayerNorm::load(
            vb.pp("mlp_norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )?;
        let mlp = ModernBertMLP::load(vb.pp("mlp"), config)?;

        let span = tracing::span!(tracing::Level::TRACE, "layer");

        Ok(ModernBertEncoderLayer {
            attn_norm,
            attn,
            mlp_norm,
            mlp,
            span,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let residual = hidden_states.clone();

        let attn_norm = if let Some(attn_norm) = &self.attn_norm {
            attn_norm.forward(hidden_states, None)?
        } else {
            hidden_states.clone()
        };

        let attn_outputs = self.attn.forward(&attn_norm, cu_seqlens, cos, sin, max_s)?;

        let hidden_states = residual.add(&attn_outputs)?;

        let mlp_output = self
            .mlp
            .forward(&self.mlp_norm.forward(&hidden_states, None)?)?;

        hidden_states.add(&mlp_output)
    }
}

struct ModernBertEncoder {
    layers: Vec<ModernBertEncoderLayer>,

    global_attn_every_n_layers: usize,

    span: tracing::Span,
}

impl ModernBertEncoder {
    pub fn load(vb: VarBuilder, config: &ModernBertConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| ModernBertEncoderLayer::load(vb.pp(format!("{index}")), index, config))
            .collect::<Result<Vec<_>>>()?;

        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(ModernBertEncoder {
            layers,
            global_attn_every_n_layers: config.global_attn_every_n_layers,
            span,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
        rotary_cache: &HashMap<bool, (Tensor, Tensor)>,
        max_s: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        for (index, layer) in self.layers.iter().enumerate() {
            let use_local_attention = index % self.global_attn_every_n_layers != 0;
            let (cos, sin) = &rotary_cache[&use_local_attention];

            hidden_states = layer.forward(&hidden_states, cu_seqlens, cos, sin, max_s)?;
        }

        Ok(hidden_states)
    }
}

pub struct FlashModernBertModel {
    embeddings: ModernBertEmbeddings,
    encoder: ModernBertEncoder,
    final_norm: LayerNorm,
    pool: Pool,
    classifier: Option<Box<dyn ClassificationHead + Send>>,

    local_attention: usize,
    rotary_dim: usize,
    rotary_cache: HashMap<bool, (Tensor, Tensor)>,
    pad_token_id: u32,
    num_attention_heads: usize,

    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl FlashModernBertModel {
    pub fn load(vb: VarBuilder, config: &BertConfig, model_type: ModelType) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashModernBert requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashModernBert requires DType::F16")
        }

        let (pool, classifier) = match model_type {
            ModelType::Classifier => {
                let pool = Pool::Cls;

                let classifier: Box<dyn ClassificationHead + Send> =
                    Box::new(ModernBertClassificationHead::load(vb.clone(), config)?);

                (pool, Some(classifier))
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("`splade` is not supported for ModernBert")
                }

                if pool == Pool::LastToken {
                    candle::bail!("`last_token` is not supported for ModernBert");
                }

                (pool, None)
            }
        };

        let embeddings = ModernBertEmbeddings::load(vb.pp("model.embeddings"), config)?;
        let encoder = ModernBertEncoder::load(vb.pp("model.layers"), config)?;
        let final_norm = LayerNorm::load(
            vb.pp("model.final_norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )?;

        let rotary_dim = config.hidden_size / config.num_attention_heads;
        let mut rotary_cache: HashMap<bool, (Tensor, Tensor)> = HashMap::new();

        for use_local_attention in [true, false] {
            let rope_theta = if use_local_attention {
                config.local_rope_theta
            } else {
                config.global_rope_theta
            };

            let max_position_embeddings = if use_local_attention {
                config.max_position_embeddings
            } else {
                config.local_attention
            };

            let inv_freqs = get_inv_freqs(rotary_dim, rope_theta as f32, vb.device(), None)?;

            let (cos, sin) = get_cos_sin(max_position_embeddings, &inv_freqs, vb.dtype(), true)?;

            rotary_cache.insert(use_local_attention, (cos, sin));
        }

        Ok(Self {
            embeddings,
            encoder,
            final_norm,
            pool,
            classifier,
            local_attention: config.local_attention,
            rotary_dim,
            rotary_cache,
            pad_token_id: config.pad_token_id as u32,
            num_attention_heads: config.num_attention_heads,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    fn get_global_attention_mask(
        &self,
        attention_mask: Option<&Tensor>,
        input_shape: &Shape,
        num_attention_heads: usize,
    ) -> Result<Tensor> {
        let extended_attention_mask = if let Some(attention_mask) = attention_mask {
            attention_mask.squeeze(2)?
        } else {
            Tensor::ones(input_shape, DType::F32, &self.device)?
        }
        .unsqueeze(1)?
        .unsqueeze(1)?
        .to_dtype(self.dtype)?;

        let (bs, seq_len) = input_shape.dims2()?;
        let extended_attention_mask =
            extended_attention_mask.broadcast_as((bs, num_attention_heads, seq_len, seq_len))?;

        Ok(extended_attention_mask)
    }

    fn get_silding_window_mask(
        &self,
        attention_mask: &Tensor,
        local_attention: usize,
    ) -> Result<Tensor> {
        let attention_mask = attention_mask.to_dtype(DType::U8)?;
        let mask_shape = attention_mask.shape();
        let (_, _, seq_len, _) = mask_shape.dims4()?;

        let rows = Tensor::arange(0, seq_len as i64, attention_mask.device())?.unsqueeze(0)?;
        let rows = rows.broadcast_as((seq_len, seq_len))?;

        let distance = (&rows - &rows.t()?)?.abs()?;

        let window_size = local_attention / 2;
        let window_mask = distance
            .le(window_size as i64)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(mask_shape)?;

        let zero_tensor = Tensor::zeros_like(&attention_mask)?;
        let sliding_window_mask = attention_mask.where_cond(&window_mask, &zero_tensor)?;

        Ok(sliding_window_mask)
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let shape = batch.input_ids.len();

        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        let mut rotary_cache: HashMap<bool, (Tensor, Tensor)> = HashMap::new();
        for use_local_attention in [true, false] {
            let (cos, sin) = &self.rotary_cache[&use_local_attention];

            let cos = cos.index_select(&position_ids, 0)?;
            let sin = sin.index_select(&position_ids, 0)?;

            let cos = cos.reshape((batch_size, 1, max_length, self.rotary_dim))?;
            let sin = sin.reshape((batch_size, 1, max_length, self.rotary_dim))?;

            rotary_cache.insert(use_local_attention, (cos, sin));
        }

        let hidden_states = self.embeddings.forward(&input_ids)?;
        let hidden_states = self.encoder.forward(
            &hidden_states,
            &cu_seqlens,
            &rotary_cache,
            batch.max_length as usize,
        )?;
        let outputs = self.final_norm.forward(&hidden_states, None)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                Pool::Cls | Pool::LastToken => {
                    if batch_size > 1 {
                        let mut indices = match self.pool {
                            Pool::Cls => cu_seqlens.narrow(0, 0, batch_size)?,
                            Pool::LastToken => {
                                let end = cu_seqlens.narrow(0, 1, batch_size)?;
                                (&end - &end.ones_like()?)?
                            }
                            _ => unreachable!(),
                        };

                        if has_raw_requests {
                            let pooled_indices = Tensor::from_vec(
                                batch.pooled_indices.clone(),
                                batch.pooled_indices.len(),
                                &self.device,
                            )?;

                            indices = indices.index_select(&pooled_indices, 0)?
                        }

                        Some(outputs.index_select(&indices, 0)?)
                    } else {
                        Some(
                            match self.pool {
                                Pool::Cls => outputs.i(0)?,
                                Pool::LastToken => {
                                    outputs.i(batch.cumulative_seq_lengths[1] as usize - 1)?
                                }
                                _ => unreachable!(),
                            }
                            .unsqueeze(0)?,
                        )
                    }
                }
                Pool::Mean => {
                    if batch_size > 1 {
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

                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        Some((outputs.sum_keepdim(0)? / (batch.max_length as f64))?)
                    }
                }
                Pool::Splade => {
                    let splade_head = self.splade.as_ref().unwrap();
                    let relu_log = splade_head.forward(&outputs)?;

                    if batch_size > 1 {
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

                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        Some(relu_log.max_keepdim(0)?)
                    }
                }
            }
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                let mut final_indices: Vec<u32> = Vec::with_capacity(shape);
                for i in batch.raw_indices.into_iter() {
                    let i = i as usize;
                    let start = batch.cumulative_seq_lengths[i];
                    let end = batch.cumulative_seq_lengths[i + 1];

                    for j in start..end {
                        final_indices.push(j);
                    }
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

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

impl Model for FlashModernBertModel {
    fn is_padded(&self) -> bool {
        false
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
