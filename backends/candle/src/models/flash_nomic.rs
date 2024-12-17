use crate::flash_attn::flash_attn_varlen;
use crate::layers::{get_cos_sin, get_inv_freqs, LayerNorm, Linear};
use crate::models::nomic::{NomicBertEmbeddings, NomicBertGatedMLP};
use crate::models::{Model, NomicConfig};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;
use candle_rotary::apply_rotary_inplace;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

struct NomicAttention {
    qkv_linear: Linear,
    out_proj: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,

    softmax_scale: f32,

    span: tracing::Span,
}

impl NomicAttention {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let num_attention_heads = config.n_head;
        let attention_head_size = config.n_embd / config.n_head;
        let hidden_size = config.n_embd;

        let qkv_weight = vb.pp("Wqkv").get(
            (3 * num_attention_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let qkv_linear = Linear::new(qkv_weight, None, None);

        let out_proj_weight = vb
            .pp("out_proj")
            .get((hidden_size, hidden_size), "weight")?;
        let out_proj = Linear::new(out_proj_weight, None, None);

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

        Ok(Self {
            qkv_linear,
            out_proj,
            num_attention_heads,
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

        let qkv = self.qkv_linear.forward(hidden_states)?;

        // Reshape to [tokens, heads, head_size]
        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads * 3);
        new_qkv_shape.push(self.attention_head_size);

        let qkv = qkv.reshape(new_qkv_shape.as_slice())?;
        let qkv = qkv.chunk(3, 1)?;

        apply_rotary_inplace(&qkv[0], &qkv[1], &cos, &sin, true)?;

        let attention = flash_attn_varlen(
            &qkv[0],
            &qkv[1],
            &qkv[2],
            None,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            self.softmax_scale,
            false,
            None,
        )?;
        let attention = attention.flatten_from(D::Minus2)?;

        self.out_proj.forward(&attention)
    }
}

struct NomicBertBlock {
    attention: NomicAttention,
    mlp: NomicBertGatedMLP,
    post_attention_layer_norm: LayerNorm,
    output_layer_norm: LayerNorm,

    span: tracing::Span,
}

impl NomicBertBlock {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let attention = NomicAttention::load(vb.pp("attn"), config)?;
        let mlp = NomicBertGatedMLP::load(vb.pp("mlp"), config)?;

        let post_attention_layer_norm =
            LayerNorm::load(vb.pp("norm1"), config.n_embd, config.layer_norm_epsilon)?;
        let output_layer_norm =
            LayerNorm::load(vb.pp("norm2"), config.n_embd, config.layer_norm_epsilon)?;

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
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let attn_output = self
            .attention
            .forward(&hidden_states, cu_seqlens, cos, sin, max_s)?;
        let hidden_states = self
            .post_attention_layer_norm
            .forward(&hidden_states, Some(&attn_output))?;

        let mlp_out = self.mlp.forward(&hidden_states)?;

        self.output_layer_norm
            .forward(&hidden_states, Some(&mlp_out))
    }
}

struct NomicBertEncoder {
    layers: Vec<NomicBertBlock>,
    span: tracing::Span,
}

impl NomicBertEncoder {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let layers = (0..config.n_layer)
            .map(|index| NomicBertBlock::load(vb.pp(format!("layers.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(NomicBertEncoder { layers, span })
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

        let mut hidden_states = hidden_states.clone();

        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, cu_seqlens, cos, sin, max_s)?
        }

        Ok(hidden_states)
    }
}

pub struct FlashNomicBertModel {
    embeddings: NomicBertEmbeddings,
    encoder: NomicBertEncoder,
    pool: Pool,
    pub device: Device,

    max_trained_positions: u32,
    rotary_cache: (Tensor, Tensor),
    scaled_rotary_cache: Option<(Tensor, Tensor)>,

    span: tracing::Span,
}

impl FlashNomicBertModel {
    pub fn load(vb: VarBuilder, config: &NomicConfig, model_type: ModelType) -> Result<Self> {
        if !config.valid() {
            candle::bail!("config is not supported")
        }

        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashNomicBertModel requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashNomicBertModel requires DType::F16")
        }

        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Nomic")
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("`splade` is not supported for Nomic")
                }
                pool
            }
        };

        let embeddings = NomicBertEmbeddings::load(vb.clone(), config)?;
        let encoder = NomicBertEncoder::load(vb.pp("encoder"), config)?;

        let rotary_dim = encoder.layers[0].attention.attention_head_size;
        let inv_freqs = get_inv_freqs(rotary_dim, config.rotary_emb_base, vb.device(), None)?;
        let rotary_cache = get_cos_sin(config.n_positions, &inv_freqs, vb.dtype(), false)?;

        let scaled_rotary_cache = if let Some(scaling_factor) = config.rotary_scaling_factor {
            let new_base = (config.rotary_emb_base
                * ((scaling_factor * config.n_positions as f32
                    / config.max_trained_positions as f32)
                    - (scaling_factor - 1.0)))
                .powi((rotary_dim as f32 / (rotary_dim as f32 - 2.0)) as i32);
            let inv_freqs = get_inv_freqs(rotary_dim, new_base, vb.device(), None)?;
            Some(get_cos_sin(
                config.n_positions,
                &inv_freqs,
                vb.dtype(),
                false,
            )?)
        } else {
            None
        };

        Ok(Self {
            embeddings,
            encoder,
            pool,
            max_trained_positions: config.max_trained_positions as u32,
            rotary_cache,
            scaled_rotary_cache,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
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

        let (cos, sin) = if self.scaled_rotary_cache.is_some()
            && batch.max_length > self.max_trained_positions
        {
            let cos = self
                .scaled_rotary_cache
                .as_ref()
                .unwrap()
                .0
                .index_select(&position_ids, 0)?;
            let sin = self
                .scaled_rotary_cache
                .as_ref()
                .unwrap()
                .1
                .index_select(&position_ids, 0)?;
            (cos, sin)
        } else {
            let cos = self.rotary_cache.0.index_select(&position_ids, 0)?;
            let sin = self.rotary_cache.1.index_select(&position_ids, 0)?;
            (cos, sin)
        };

        let embedding_output = self.embeddings.forward(&input_ids, &type_ids)?;

        let outputs = self.encoder.forward(
            &embedding_output,
            &cu_seqlens,
            &cos,
            &sin,
            batch.max_length as usize,
        )?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                // CLS and LastToken pooling
                Pool::Cls | Pool::LastToken => {
                    if batch_size > 1 {
                        // Get token indices form cu_seqlens
                        let mut indices = match self.pool {
                            Pool::Cls => cu_seqlens.narrow(0, 0, batch_size)?,
                            Pool::LastToken => {
                                let end = cu_seqlens.narrow(0, 1, batch_size)?;
                                (&end - &end.ones_like()?)?
                            }
                            _ => unreachable!(),
                        };

                        // If raw_indices is empty, we don't need to do anything with
                        // the pooled_indices
                        if has_raw_requests {
                            // We need the pooled indices to select the correct cls indices
                            let pooled_indices = Tensor::from_vec(
                                batch.pooled_indices.clone(),
                                batch.pooled_indices.len(),
                                &self.device,
                            )?;

                            // Only select indices that requires pooling
                            indices = indices.index_select(&pooled_indices, 0)?
                        }

                        // Select tokens
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
                // Mean pooling
                Pool::Mean => {
                    if batch_size > 1 {
                        // for each request that requires pooling
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

                        // Concatenate all results
                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        Some((outputs.sum_keepdim(0)? / (batch.max_length as f64))?)
                    }
                }
                Pool::Splade => {
                    unreachable!();
                }
            }
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                // Create indexing vector for the embeddings
                let mut final_indices: Vec<u32> = Vec::with_capacity(shape);
                for i in batch.raw_indices.into_iter() {
                    let i = i as usize;
                    // Get start/end token index of this specific member of the batch
                    let start = batch.cumulative_seq_lengths[i];
                    let end = batch.cumulative_seq_lengths[i + 1];

                    for j in start..end {
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

impl Model for FlashNomicBertModel {
    fn is_padded(&self) -> bool {
        false
    }
    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
