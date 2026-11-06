use crate::flash_attn::flash_attn_varlen;
use crate::layers::{index_select, Linear, RMSNorm};
use crate::models::embedding_gemma2::{
    chunked_attention, padding_bias, AttentionPlan, AttentionType, EmbeddingGemma2Attention,
    EmbeddingGemma2MLP, EmbeddingGemma2PLE, EmbeddingGemma2PLEBlock, EmbeddingGemma2TextConfig,
    RotaryEmbedding,
};
use crate::models::{EmbeddingGemma2Config, Model};

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_rotary::apply_rotary_inplace;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

// NOTE: Flash Attention v2 supports head dimensions up to 256, so the full attention layers of
// EmbeddingGemma2 (with a head dimension of 512) fall back to the padded chunked attention
const MAX_FLASH_ATTN_HEAD_DIM: usize = 256;

// Everything a layer needs to attend over the unpadded batch, shared by all the layers
struct AttentionInputs {
    cu_seqlens: Tensor,
    max_s: usize,
    cos_full: Tensor,
    sin_full: Tensor,
    cos_sliding: Tensor,
    sin_sliding: Tensor,
    // NOTE: Only needed by the layers that fall back to the padded attention, see `Padding`
    padding: Option<Padding>,
}

// Maps the unpadded tokens to a padded `(batch_size, max_s)` layout and back
struct Padding {
    batch_size: usize,
    // Index of the token in each padded position (padded positions point to any valid token, as
    // those are masked out as keys and discarded as queries)
    pad_indices: Tensor,
    // Index of the padded position of each token
    unpad_indices: Tensor,
    plan: AttentionPlan,
}

impl Padding {
    fn new(
        cumulative_seq_lengths: &[u32],
        max_s: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let batch_size = cumulative_seq_lengths.len() - 1;
        let input_lengths: Vec<usize> = cumulative_seq_lengths
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .collect();

        let mut pad_indices = Vec::with_capacity(batch_size * max_s);
        let mut unpad_indices = Vec::with_capacity(cumulative_seq_lengths[batch_size] as usize);
        for (i, &length) in input_lengths.iter().enumerate() {
            let start = cumulative_seq_lengths[i];
            for j in 0..max_s {
                if j < length {
                    pad_indices.push(start + j as u32);
                    unpad_indices.push((i * max_s + j) as u32);
                } else {
                    pad_indices.push(start);
                }
            }
        }

        let num_pad_indices = pad_indices.len();
        let num_unpad_indices = unpad_indices.len();
        let plan = AttentionPlan::new(
            padding_bias(&input_lengths, max_s, dtype, device)?,
            max_s,
            None,
            dtype,
            device,
        )?;

        Ok(Self {
            batch_size,
            pad_indices: Tensor::from_vec(pad_indices, num_pad_indices, device)?,
            unpad_indices: Tensor::from_vec(unpad_indices, num_unpad_indices, device)?,
            plan,
        })
    }

    // `(num_tokens, num_heads, head_dim)` -> `(batch_size, num_heads, max_s, head_dim)`
    fn pad(&self, x: &Tensor, max_s: usize) -> Result<Tensor> {
        let (_, num_heads, head_dim) = x.dims3()?;
        index_select(&x.flatten_from(1)?, &self.pad_indices, 0)?
            .reshape((self.batch_size, max_s, num_heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    // `(batch_size, num_heads, max_s, head_dim)` -> `(num_tokens, num_heads * head_dim)`
    fn unpad(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?.flatten_to(1)?.flatten_from(1)?;
        index_select(&x, &self.unpad_indices, 0)
    }
}

struct FlashEmbeddingGemma2Layer {
    input_layernorm: RMSNorm,
    self_attn: EmbeddingGemma2Attention,
    post_attention_layernorm: RMSNorm,

    pre_feedforward_layernorm: RMSNorm,
    mlp: EmbeddingGemma2MLP,
    post_feedforward_layernorm: RMSNorm,

    ple_block: EmbeddingGemma2PLEBlock,
    layer_scalar: Tensor,

    attention_type: AttentionType,
    sliding_window: usize,

    span: tracing::Span,
}

impl FlashEmbeddingGemma2Layer {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let (head_dim, num_key_value_heads) = config.layer_dims(layer_idx);
        let hidden_size = config.hidden_size;
        let eps = config.rms_norm_eps;

        Ok(Self {
            input_layernorm: RMSNorm::load(vb.pp("input_layernorm"), hidden_size, eps)?,
            self_attn: EmbeddingGemma2Attention::load(
                vb.pp("self_attn"),
                config,
                head_dim,
                num_key_value_heads,
            )?,
            post_attention_layernorm: RMSNorm::load(
                vb.pp("post_attention_layernorm"),
                hidden_size,
                eps,
            )?,
            pre_feedforward_layernorm: RMSNorm::load(
                vb.pp("pre_feedforward_layernorm"),
                hidden_size,
                eps,
            )?,
            mlp: EmbeddingGemma2MLP::load(vb.pp("mlp"), config)?,
            post_feedforward_layernorm: RMSNorm::load(
                vb.pp("post_feedforward_layernorm"),
                hidden_size,
                eps,
            )?,
            ple_block: EmbeddingGemma2PLEBlock::load(vb.pp("ple_block"), config)?,
            layer_scalar: vb.get(1, "layer_scalar")?,
            attention_type: config.layer_types[layer_idx],
            sliding_window: config.sliding_window,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn uses_flash_attn(&self) -> bool {
        self.self_attn.attention_head_size <= MAX_FLASH_ATTN_HEAD_DIM
    }

    fn attention(&self, hidden_states: &Tensor, inputs: &AttentionInputs) -> Result<Tensor> {
        let (q, k, v) = self.self_attn.project(hidden_states)?;

        let (cos, sin) = match self.attention_type {
            AttentionType::FullAttention => (&inputs.cos_full, &inputs.sin_full),
            AttentionType::SlidingAttention => (&inputs.cos_sliding, &inputs.sin_sliding),
        };
        apply_rotary_inplace(&q, &k, cos, sin, true)?;

        let context_layer = if self.uses_flash_attn() {
            // NOTE: The window is a bidirectional radius, i.e. `abs(i - j) <= sliding_window`
            let window_size = match self.attention_type {
                AttentionType::FullAttention => None,
                AttentionType::SlidingAttention => Some(self.sliding_window),
            };
            // NOTE: The attention scaling is 1.0, as both queries and keys are already
            // RMS-normalized
            flash_attn_varlen(
                &q,
                &k,
                &v,
                None,
                &inputs.cu_seqlens,
                &inputs.cu_seqlens,
                inputs.max_s,
                inputs.max_s,
                1.0,
                false,
                window_size,
                window_size,
            )?
            .flatten_from(D::Minus2)?
        } else {
            let padding = match &inputs.padding {
                Some(padding) => padding,
                None => candle::bail!("`padding` is required for the padded attention"),
            };
            if self.attention_type == AttentionType::SlidingAttention {
                candle::bail!(
                    "Sliding attention with a head dimension over {MAX_FLASH_ATTN_HEAD_DIM} is not supported for FlashEmbeddingGemma2"
                );
            }
            let context_layer = chunked_attention(
                &padding.pad(&q, inputs.max_s)?,
                &padding.pad(&k, inputs.max_s)?,
                &padding.pad(&v, inputs.max_s)?,
                &padding.plan,
            )?;
            padding.unpad(&context_layer)?
        };

        self.self_attn.output(&context_layer)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        per_layer_input: &Tensor,
        inputs: &AttentionInputs,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (attn_states, _) = self.input_layernorm.forward(hidden_states, None)?;
        let attn_states = self.attention(&attn_states, inputs)?;
        let (attn_states, _) = self.post_attention_layernorm.forward(&attn_states, None)?;

        // NOTE: Adds the attention output to the residual and normalizes it in a single step
        let (mlp_states, hidden_states) = self
            .pre_feedforward_layernorm
            .forward(&attn_states, Some(hidden_states))?;
        let mlp_states = self.mlp.forward(&mlp_states)?;
        let (mlp_states, _) = self.post_feedforward_layernorm.forward(&mlp_states, None)?;
        let hidden_states = (hidden_states + mlp_states)?;

        let hidden_states = self.ple_block.forward(&hidden_states, per_layer_input)?;
        hidden_states.broadcast_mul(&self.layer_scalar)
    }
}

pub struct FlashEmbeddingGemma2Model {
    embed_tokens: Embedding,
    embed_scale: f64,
    ple: EmbeddingGemma2PLE,
    layers: Vec<FlashEmbeddingGemma2Layer>,
    norm: RMSNorm,
    embedding_projection: Linear,

    rotary_full: RotaryEmbedding,
    rotary_sliding: RotaryEmbedding,

    pool: Pool,

    dtype: DType,
    device: Device,

    span: tracing::Span,
}

impl FlashEmbeddingGemma2Model {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2Config,
        model_type: ModelType,
    ) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashEmbeddingGemma2 requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashEmbeddingGemma2 requires DType::F16")
        }

        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for EmbeddingGemma2")
            }
            ModelType::Embedding(pool) => pool,
        };
        if matches!(pool, Pool::Cls | Pool::Splade) {
            candle::bail!(
                "EmbeddingGemma2 only supports mean and last-token pooling, got {pool:?}"
            );
        }

        let config = &config.text_config;
        if config.layer_types.len() != config.num_hidden_layers {
            candle::bail!(
                "`layer_types` has {} entries but `num_hidden_layers` is {}",
                config.layer_types.len(),
                config.num_hidden_layers
            );
        }

        // NOTE: Only the text tower is loaded, the vision and audio towers are ignored
        let vb = vb.pp("language_model");

        let embed_tokens = Embedding::new(
            vb.pp("embed_tokens")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );
        let embed_scale = (config.hidden_size as f64).sqrt();

        let ple = EmbeddingGemma2PLE::load(vb.pp("ple"), config)?;

        let layers = (0..config.num_hidden_layers)
            .map(|layer_idx| {
                FlashEmbeddingGemma2Layer::load(
                    vb.pp(format!("layers.{layer_idx}")),
                    config,
                    layer_idx,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        let embedding_projection = Linear::new(
            vb.pp("embedding_projection")
                .get((config.embedding_dim, config.hidden_size), "weight")?,
            None,
            None,
        );

        let rotary_full = RotaryEmbedding::load(config, AttentionType::FullAttention, vb.device())?;
        let rotary_sliding =
            RotaryEmbedding::load(config, AttentionType::SlidingAttention, vb.device())?;

        Ok(Self {
            embed_tokens,
            embed_scale,
            ple,
            layers,
            norm,
            embedding_projection,
            rotary_full,
            rotary_sliding,
            pool,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    // Computes the `cos` and `sin` of shape `(num_tokens, dim / 2)` for the given positions, as
    // expected by `apply_rotary_inplace`
    fn cos_sin(&self, rotary: &RotaryEmbedding, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let positions = position_ids.to_dtype(DType::F32)?.unsqueeze(1)?;
        // NOTE: `inv_freqs` is `(1, dim / 2)`
        let freqs = positions.broadcast_mul(&rotary.inv_freqs)?;
        Ok((
            freqs.cos()?.to_dtype(self.dtype)?,
            freqs.sin()?.to_dtype(self.dtype)?,
        ))
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.cumulative_seq_lengths.len() - 1;
        let shape = batch.input_ids.len();
        let max_s = batch.max_length as usize;

        // Create Cuda tensors
        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        let (cos_full, sin_full) = self.cos_sin(&self.rotary_full, &position_ids)?;
        let (cos_sliding, sin_sliding) = self.cos_sin(&self.rotary_sliding, &position_ids)?;

        let padding = if self.layers.iter().all(|layer| layer.uses_flash_attn()) {
            None
        } else {
            Some(Padding::new(
                &batch.cumulative_seq_lengths,
                max_s,
                self.dtype,
                &self.device,
            )?)
        };

        let inputs = AttentionInputs {
            cu_seqlens: cu_seqlens.clone(),
            max_s,
            cos_full,
            sin_full,
            cos_sliding,
            sin_sliding,
            padding,
        };

        let inputs_embeds = (self.embed_tokens.forward(&input_ids)? * self.embed_scale)?;
        let mut hidden_states = inputs_embeds.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let per_layer_input = self.ple.forward(&inputs_embeds, layer_idx)?;
            hidden_states = layer.forward(&hidden_states, &per_layer_input, &inputs)?;
        }

        let (outputs, _) = self.norm.forward(&hidden_states, None)?;
        // NOTE: Projecting per token is equivalent to projecting after mean pooling, and pooling
        // is done in fp32, as summing up to `max_input_length` token embeddings would overflow in
        // fp16
        let outputs = self
            .embedding_projection
            .forward(&outputs)?
            .to_dtype(DType::F32)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            let results = batch
                .pooled_indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    let start = batch.cumulative_seq_lengths[i] as usize;
                    let length = batch.cumulative_seq_lengths[i + 1] as usize - start;
                    match self.pool {
                        Pool::Mean => outputs.narrow(0, start, length)?.sum_keepdim(0)? / (length as f64),
                        Pool::LastToken => outputs.i(start + length - 1)?.unsqueeze(0),
                        Pool::Cls | Pool::Splade => unreachable!(
                            "Only Mean and Last-Token pooling are supported for EmbeddingGemma2, no CLS or SPLADE"
                        ),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Some(Tensor::cat(&results, 0)?)
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                let mut final_indices: Vec<u32> = Vec::with_capacity(shape);
                for &i in &batch.raw_indices {
                    let i = i as usize;
                    let start = batch.cumulative_seq_lengths[i];
                    let end = batch.cumulative_seq_lengths[i + 1];
                    final_indices.extend(start..end);
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;
                Some(index_select(&outputs, &final_indices, 0)?)
            } else {
                Some(outputs)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for FlashEmbeddingGemma2Model {
    fn is_padded(&self) -> bool {
        false
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
