use crate::flash_attn::flash_attn_varlen;
use crate::layers::{LayerNorm, Linear};
use crate::models::modernbert::{
    ClassificationHead, ModernBertClassificationHead, ModernBertConfig, ModernBertEmbeddings,
};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct ModernBertAttention {
    wqkv: Linear,
    wo: Linear,

    local_attention: (i64, i64),
    cos: Tensor,
    sin: Tensor,

    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f64,

    span: tracing::Span,
}

impl ModernBertAttention {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let wi_weight = vb
            .pp("Wi")
            .get((config.hidden_size, config.intermediate_size * 2), "weight")?;
        let wi_bias = vb
            .pp("Wi")
            .get((config.intermediate_size * 2,), "bias")
            .ok();
        let wi = Linear::new(wi_weight, wi_bias, None);

        let wo_weight = vb
            .pp("Wo")
            .get((config.intermediate_size * 2, config.hidden_size), "weight")?;
        let wo_bias = vb.pp("Wo").get((config.hidden_size,), "bias").ok();

        let wo = Linear::new(wo_weight, wo_bias, None);

        let activation = Some(config.hidden_activation.clone());

        Ok(Self {
            wi,
            wo,
            activation,
            intermediate_size: config.intermediate_size,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
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

        let query_layer =
            apply_rotary(query_layer, &self.cos, &self.sin, self.attention_head_size)?;
        let key_layer = apply_rotary(key_layer, &self.cos, &self.sin, self.attention_head_size)?;

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
            self.local_attention[0],
            self.local_attention[1],
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
        let attn_norm = if index > 0 {
            Some(LayerNorm::load(
                vb.pp("attn_norm"),
                config.hidden_size,
                config.norm_eps as f32,
            )?)
        } else {
            None
        };

        let attn = ModernBertAttention::load(vb.pp("attn"), index, config)?;

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
        attention_mask: &Tensor,
        silding_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        if let Some(attn_norm) = &self.attn_norm {
            hidden_states = attn_norm.forward(&hidden_states, None)?;
        }

        let hidden_states =
            self.attn
                .forward(&hidden_states, attention_mask, silding_attention_mask)?;
        let mlp_output = self
            .mlp
            .forward(&self.mlp_norm.forward(&hidden_states, None)?)?;

        hidden_states.broadcast_add(&mlp_output)
    }
}

struct ModernBertEncoder {
    layers: Vec<ModernBertEncoderLayer>,
    span: tracing::Span,
}

impl ModernBertEncoder {
    pub fn load(vb: VarBuilder, config: &ModernBertConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| ModernBertEncoderLayer::load(vb.pp(format!("{index}")), index, config))
            .collect::<Result<Vec<_>>>()?;

        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(ModernBertEncoder { layers, span })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        silding_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.clone();

        for layer in self.layers.iter() {
            hidden_states =
                layer.forward(&hidden_states, attention_mask, silding_attention_mask)?;
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
            vb.pp("final_norm"),
            config.hidden_size,
            config.norm_eps as f32,
        )?;

        Ok(Self {
            embeddings,
            encoder,
            final_norm,
            pool,
            classifier,
            local_attention: config.local_attention,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    fn get_global_attention_mask(
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

        let (bs, seq_len) = input_shape.dims2()?;
        let extended_attention_mask =
            extended_attention_mask.broadcast_as((bs, 1, seq_len, seq_len))?;

        Ok(extended_attention_mask)
    }

    fn get_silding_window_mask(
        &self,
        attention_mask: &Tensor,
        local_attention: usize,
    ) -> Result<Tensor> {
        let mask_shape = attention_mask.shape();
        let (_, _, seq_len, _) = mask_shape.dims4()?;

        let rows = Tensor::arange(0, seq_len as i64, attention_mask.device())?.unsqueeze(0)?;
        let distance = (&rows - &rows.t()?)?.abs()?;

        let window_size = local_attention / 2;
        let window_mask = distance
            .le(window_size as i64)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        let dtype = attention_mask.dtype();
        let min_value = match dtype {
            DType::F32 => f32::MIN as f64,
            _ => -65504.0, // f16 minimum value
        };

        let inverted_window_mask = window_mask.eq(0_i64)?;
        let min_value_tensor = Tensor::full(min_value, mask_shape, attention_mask.device())?;
        let sliding_window_mask =
            attention_mask.where_cond(&inverted_window_mask, &min_value_tensor)?;

        Ok(sliding_window_mask)
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let shape = batch.input_ids.len();

        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        let global_attention_mask =
            self.get_global_attention_mask(attention_mask.as_ref(), input_ids.shape())?;
        let silding_attention_mask =
            self.get_silding_window_mask(&global_attention_mask, self.local_attention)?;

        let hidden_states = self.embeddings.forward(&input_ids)?;
        let hidden_states = self.encoder.forward(
            &hidden_states,
            &global_attention_mask,
            &silding_attention_mask,
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
