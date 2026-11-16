use std::collections::HashMap;

use crate::layers::{apply_rotary, get_inv_freqs, HiddenAct, Linear, RMSNorm, RopeParameters};
use crate::models::Model;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

// NOTE: Defaults applied by `transformers` to the full attention layers when `per_layer_config` is
// not explicitly set, see `EmbeddingGemma2TextConfig.__post_init__`
const DEFAULT_GLOBAL_HEAD_DIM: usize = 512;
const DEFAULT_NUM_GLOBAL_KEY_VALUE_HEADS: usize = 1;

// NOTE: Large enough to keep the matmuls efficient, small enough that the attention scores of a
// chunk stay in the hundreds of MiB even for a batch of 8192-token inputs
const QUERY_CHUNK_SIZE: usize = 1024;

// NOTE: A large finite value is used rather than `-inf` (or the dtype minimum, which would
// overflow when the padding and the window biases add up) so that fully masked rows (e.g. a padded
// query far from every valid key) don't produce NaNs that would leak into the valid tokens
fn mask_value(dtype: DType) -> f32 {
    match dtype {
        DType::F16 => -30000.0,
        _ => -1e30,
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2LayerConfig {
    pub head_dim: Option<usize>,
    pub num_key_value_heads: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2TextConfig {
    #[serde(default)]
    pub attention_bias: bool,
    pub pad_token_id: u32,
    pub head_dim: usize,
    pub hidden_activation: HiddenAct,
    pub hidden_size: usize,
    pub hidden_size_per_layer_input: usize,
    pub embedding_dim: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: HashMap<AttentionType, RopeParameters>,
    // NOTE: Unlike Gemma3, this is the inclusive radius of the bidirectional window i.e. a token
    // attends to every position with `abs(i - j) <= sliding_window`
    pub sliding_window: usize,
    pub layer_types: Vec<AttentionType>,
    pub per_layer_config: Option<HashMap<String, EmbeddingGemma2LayerConfig>>,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2Config {
    pub text_config: EmbeddingGemma2TextConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionType {
    FullAttention,
    SlidingAttention,
}

impl EmbeddingGemma2TextConfig {
    // NOTE: Resolves the `(head_dim, num_key_value_heads)` of a given layer, as those differ
    // between sliding and full attention layers
    pub fn layer_dims(&self, layer_idx: usize) -> (usize, usize) {
        let layer_config = match &self.per_layer_config {
            Some(per_layer_config) => per_layer_config
                .iter()
                .find(|(key, _)| key.parse::<usize>().ok() == Some(layer_idx))
                .map(|(_, value)| value.clone()),
            None if self.layer_types[layer_idx] == AttentionType::FullAttention => {
                Some(EmbeddingGemma2LayerConfig {
                    head_dim: Some(DEFAULT_GLOBAL_HEAD_DIM),
                    num_key_value_heads: Some(DEFAULT_NUM_GLOBAL_KEY_VALUE_HEADS),
                })
            }
            None => None,
        };

        match layer_config {
            Some(layer_config) => (
                layer_config.head_dim.unwrap_or(self.head_dim),
                layer_config
                    .num_key_value_heads
                    .unwrap_or(self.num_key_value_heads),
            ),
            None => (self.head_dim, self.num_key_value_heads),
        }
    }
}

pub struct EmbeddingGemma2Attention {
    qkv_proj: Linear,
    o_proj: Linear,

    q_norm: RMSNorm,
    k_norm: RMSNorm,
    v_norm: RMSNorm,

    pub attention_head_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    span: tracing::Span,
}

impl EmbeddingGemma2Attention {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2TextConfig,
        attention_head_size: usize,
        num_key_value_heads: usize,
    ) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let hidden_size = config.hidden_size;

        let query_weight = vb.pp("q_proj").get(
            (num_attention_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let key_weight = vb.pp("k_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let value_weight = vb.pp("v_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;

        let qkv_bias = if config.attention_bias {
            let query_bias = vb
                .pp("q_proj")
                .get(num_attention_heads * attention_head_size, "bias")?;
            let key_bias = vb
                .pp("k_proj")
                .get(num_key_value_heads * attention_head_size, "bias")?;
            let value_bias = vb
                .pp("v_proj")
                .get(num_key_value_heads * attention_head_size, "bias")?;
            Some(Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?)
        } else {
            None
        };
        let qkv_proj = Linear::new(qkv_weight, qkv_bias, None);

        let output_weight = vb.pp("o_proj").get(
            (hidden_size, num_attention_heads * attention_head_size),
            "weight",
        )?;
        let output_bias = if config.attention_bias {
            Some(vb.pp("o_proj").get(hidden_size, "bias")?)
        } else {
            None
        };
        let o_proj = Linear::new(output_weight, output_bias, None);

        let q_norm = RMSNorm::load(vb.pp("q_norm"), attention_head_size, config.rms_norm_eps)?;
        let k_norm = RMSNorm::load(vb.pp("k_norm"), attention_head_size, config.rms_norm_eps)?;
        // NOTE: The norm over the value states has no weight
        let v_norm = RMSNorm::new(
            Tensor::ones(attention_head_size, vb.dtype(), vb.device())?,
            config.rms_norm_eps,
        );

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            attention_head_size,
            num_attention_heads,
            num_key_value_heads,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    // NOTE: Returns the normalized queries, keys and values (before RoPE), with shape
    // `(..., num_heads, head_dim)` for any leading dimensions, so that it's shared with the
    // unpadded layout of `FlashEmbeddingGemma2Model`
    pub fn project(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = self.qkv_proj.forward(hidden_states)?;

        let input_dims = hidden_states.dims();
        let input_shape = &input_dims[..input_dims.len() - 1];

        let q_size = self.num_attention_heads * self.attention_head_size;
        let kv_size = self.num_key_value_heads * self.attention_head_size;

        let q = qkv.narrow(D::Minus1, 0, q_size)?.reshape(
            [
                input_shape,
                &[self.num_attention_heads, self.attention_head_size],
            ]
            .concat(),
        )?;
        let k = qkv.narrow(D::Minus1, q_size, kv_size)?.reshape(
            [
                input_shape,
                &[self.num_key_value_heads, self.attention_head_size],
            ]
            .concat(),
        )?;
        let v = qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?.reshape(
            [
                input_shape,
                &[self.num_key_value_heads, self.attention_head_size],
            ]
            .concat(),
        )?;

        let (q, _) = self.q_norm.forward(&q, None)?;
        let (k, _) = self.k_norm.forward(&k, None)?;
        let (v, _) = self.v_norm.forward(&v, None)?;
        Ok((q, k, v))
    }

    pub fn output(&self, context_layer: &Tensor) -> Result<Tensor> {
        self.o_proj.forward(context_layer)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        plan: &AttentionPlan,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (q, k, v) = self.project(hidden_states)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = apply_rotary(&q.contiguous()?, cos, sin, self.attention_head_size)?;
        let k = apply_rotary(&k.contiguous()?, cos, sin, self.attention_head_size)?;

        let context_layer = chunked_attention(&q, &k, &v, plan)?;

        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;
        self.output(&context_layer)
    }
}

// NOTE: Bidirectional attention over padded inputs, where `q` is `(batch_size, num_heads, seq_len,
// head_dim)` and `k` and `v` are `(batch_size, num_kv_heads, seq_len, head_dim)`
pub fn chunked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    plan: &AttentionPlan,
) -> Result<Tensor> {
    let (batch_size, num_attention_heads, _, attention_head_size) = q.dims4()?;
    let num_key_value_heads = k.dim(1)?;

    // NOTE: Rather than repeating the keys and values for each query head (GQA), the query heads
    // that share a key-value head are stacked along the sequence dimension, i.e.
    // `(batch_size, num_kv_heads, num_groups * q_len, head_dim)`, so no copies are needed
    let num_groups = num_attention_heads / num_key_value_heads;

    // NOTE: The queries are processed in chunks so that the attention scores never materialize as
    // a full `(batch_size, num_heads, seq_len, seq_len)` tensor, which would run out of memory for
    // long inputs; each chunk only attends to the keys in its window (if any)
    let mut context_chunks = Vec::with_capacity(plan.chunks.len());
    for chunk in &plan.chunks {
        let q = q.narrow(2, chunk.q_start, chunk.q_len)?;
        let k = k.narrow(2, chunk.k_start, chunk.k_len)?;
        let v = v.narrow(2, chunk.k_start, chunk.k_len)?;

        let q = q.contiguous()?.reshape((
            batch_size,
            num_key_value_heads,
            num_groups * chunk.q_len,
            attention_head_size,
        ))?;

        // NOTE: The attention scaling is 1.0, as both queries and keys are already RMS-normalized
        let mut attn_weights = q.matmul(&k.t()?.contiguous()?)?.reshape((
            batch_size,
            num_attention_heads,
            chunk.q_len,
            chunk.k_len,
        ))?;
        if let Some(padding_bias) = &plan.padding_bias {
            let padding_bias = padding_bias.narrow(3, chunk.k_start, chunk.k_len)?;
            attn_weights = attn_weights.broadcast_add(&padding_bias)?;
        }
        if let Some(window_bias) = &chunk.window_bias {
            attn_weights = attn_weights.broadcast_add(window_bias)?;
        }
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?.reshape((
            batch_size,
            num_key_value_heads,
            num_groups * chunk.q_len,
            chunk.k_len,
        ))?;
        let context_layer = attn_weights.matmul(&v.contiguous()?)?;
        context_chunks.push(context_layer.reshape((
            batch_size,
            num_attention_heads,
            chunk.q_len,
            attention_head_size,
        ))?);
    }
    Tensor::cat(&context_chunks, 2)
}

pub struct EmbeddingGemma2MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    hidden_activation: HiddenAct,
    intermediate_size: usize,

    span: tracing::Span,
}

impl EmbeddingGemma2MLP {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2TextConfig) -> Result<Self> {
        let gate_proj_weight = vb
            .pp("gate_proj")
            .get((config.intermediate_size, config.hidden_size), "weight")?;
        let up_proj_weight = vb
            .pp("up_proj")
            .get((config.intermediate_size, config.hidden_size), "weight")?;
        let gate_up_proj_weight = Tensor::cat(&[&gate_proj_weight, &up_proj_weight], 0)?;
        let gate_up_proj = Linear::new(gate_up_proj_weight, None, None);

        let down_proj_weight = vb
            .pp("down_proj")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let down_proj = Linear::new(down_proj_weight, None, None);

        Ok(Self {
            gate_up_proj,
            down_proj,
            hidden_activation: config.hidden_activation.clone(),
            intermediate_size: config.intermediate_size,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.gate_up_proj.forward(hidden_states)?;
        let gate_states = gate_up_states.narrow(D::Minus1, 0, self.intermediate_size)?;
        let gate_states = self.hidden_activation.forward(&gate_states)?;
        let up_states =
            gate_up_states.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;

        self.down_proj.forward(&(gate_states * up_states)?)
    }
}

// Per-layer embeddings (PLE), derived from the input embeddings via a projection only
pub struct EmbeddingGemma2PLE {
    // NOTE: `per_layer_model_projection` is split into one projection per layer, so that each
    // layer's input is computed right before it's needed, instead of holding a
    // `(batch_size, seq_len, num_hidden_layers, hidden_size_per_layer_input)` tensor for the
    // whole forward pass; the norm is applied per layer slice either way, so it's equivalent
    per_layer_model_projections: Vec<Linear>,
    per_layer_projection_norm: RMSNorm,
    scale: f64,

    span: tracing::Span,
}

impl EmbeddingGemma2PLE {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2TextConfig) -> Result<Self> {
        let weight = vb.pp("per_layer_model_projection").get(
            (
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                config.hidden_size,
            ),
            "weight",
        )?;
        let per_layer_model_projections = (0..config.num_hidden_layers)
            .map(|layer_idx| {
                let weight = weight.narrow(
                    0,
                    layer_idx * config.hidden_size_per_layer_input,
                    config.hidden_size_per_layer_input,
                )?;
                Ok(Linear::new(weight.contiguous()?, None, None))
            })
            .collect::<Result<Vec<_>>>()?;
        let per_layer_projection_norm = RMSNorm::load(
            vb.pp("per_layer_projection_norm"),
            config.hidden_size_per_layer_input,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            per_layer_model_projections,
            per_layer_projection_norm,
            scale: (config.hidden_size as f64).powf(-0.5),
            span: tracing::span!(tracing::Level::TRACE, "ple"),
        })
    }

    // Returns the input of the given layer, of shape `(batch_size, seq_len, hidden_size_per_layer_input)`
    pub fn forward(&self, inputs_embeds: &Tensor, layer_idx: usize) -> Result<Tensor> {
        let _enter = self.span.enter();

        let per_layer_projection =
            (self.per_layer_model_projections[layer_idx].forward(inputs_embeds)? * self.scale)?;
        let (per_layer_input, _) = self
            .per_layer_projection_norm
            .forward(&per_layer_projection, None)?;
        Ok(per_layer_input)
    }
}

// Gates the residual stream with the layer's slice of the per-layer embeddings
pub struct EmbeddingGemma2PLEBlock {
    per_layer_input_gate: Linear,
    per_layer_projection: Linear,
    post_per_layer_input_norm: RMSNorm,
    hidden_activation: HiddenAct,

    span: tracing::Span,
}

impl EmbeddingGemma2PLEBlock {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2TextConfig) -> Result<Self> {
        let gate_weight = vb.pp("per_layer_input_gate").get(
            (config.hidden_size_per_layer_input, config.hidden_size),
            "weight",
        )?;
        let projection_weight = vb.pp("per_layer_projection").get(
            (config.hidden_size, config.hidden_size_per_layer_input),
            "weight",
        )?;
        let post_per_layer_input_norm = RMSNorm::load(
            vb.pp("post_per_layer_input_norm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            per_layer_input_gate: Linear::new(gate_weight, None, None),
            per_layer_projection: Linear::new(projection_weight, None, None),
            post_per_layer_input_norm,
            hidden_activation: config.hidden_activation.clone(),
            span: tracing::span!(tracing::Level::TRACE, "ple_block"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gated = self.per_layer_input_gate.forward(hidden_states)?;
        let gated = (self.hidden_activation.forward(&gated)? * per_layer_input)?;
        let projected = self.per_layer_projection.forward(&gated)?;
        let (projected, _) = self.post_per_layer_input_norm.forward(&projected, None)?;
        hidden_states + projected
    }
}

struct EmbeddingGemma2Layer {
    input_layernorm: RMSNorm,
    self_attn: EmbeddingGemma2Attention,
    post_attention_layernorm: RMSNorm,

    pre_feedforward_layernorm: RMSNorm,
    mlp: EmbeddingGemma2MLP,
    post_feedforward_layernorm: RMSNorm,

    ple_block: EmbeddingGemma2PLEBlock,
    layer_scalar: Tensor,

    attention_type: AttentionType,

    span: tracing::Span,
}

impl EmbeddingGemma2Layer {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let attention_type = config.layer_types[layer_idx];
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
            attention_type,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        per_layer_input: &Tensor,
        plan: &AttentionPlan,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (attn_states, _) = self.input_layernorm.forward(hidden_states, None)?;
        let attn_states = self.self_attn.forward(&attn_states, plan, cos, sin)?;
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

struct AttentionChunk {
    q_start: usize,
    q_len: usize,
    k_start: usize,
    k_len: usize,
    // Additive bias of shape `(1, 1, q_len, k_len)` that masks the keys outside the window
    window_bias: Option<Tensor>,
}

// Describes how the attention of a given layer type is computed for a batch: the query chunks
// with the key range each of them attends to, and the padding bias shared by all of them
pub struct AttentionPlan {
    // Additive bias of shape `(batch_size, 1, 1, seq_len)` that masks the padded keys
    padding_bias: Option<Tensor>,
    chunks: Vec<AttentionChunk>,
}

impl AttentionPlan {
    pub fn new(
        padding_bias: Option<Tensor>,
        seq_len: usize,
        sliding_window: Option<usize>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let chunks = (0..seq_len)
            .step_by(QUERY_CHUNK_SIZE)
            .map(|q_start| {
                let q_len = QUERY_CHUNK_SIZE.min(seq_len - q_start);
                let (k_start, k_end) = match sliding_window {
                    Some(window) => (
                        q_start.saturating_sub(window),
                        (q_start + q_len + window).min(seq_len),
                    ),
                    None => (0, seq_len),
                };
                let k_len = k_end - k_start;

                // NOTE: Only needed if any query-key pair in the chunk is further apart than the
                // window, i.e. for short inputs the sliding layers attend to every key
                let max_distance = (q_start + q_len - 1 - k_start).max(k_end - 1 - q_start);
                let window_bias = match sliding_window {
                    Some(window) if max_distance > window => {
                        let bias: Vec<f32> = (q_start..q_start + q_len)
                            .flat_map(|i| {
                                (k_start..k_end).map(move |j| {
                                    if i.abs_diff(j) <= window {
                                        0.0
                                    } else {
                                        mask_value(dtype)
                                    }
                                })
                            })
                            .collect();
                        Some(
                            Tensor::from_vec(bias, (1, 1, q_len, k_len), device)?
                                .to_dtype(dtype)?,
                        )
                    }
                    _ => None,
                };

                Ok(AttentionChunk {
                    q_start,
                    q_len,
                    k_start,
                    k_len,
                    window_bias,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            padding_bias,
            chunks,
        })
    }
}

// Builds the additive bias of shape `(batch_size, 1, 1, seq_len)` that masks the padded keys, or
// `None` when no sequence in the batch is padded
pub fn padding_bias(
    input_lengths: &[usize],
    max_length: usize,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    if input_lengths.iter().all(|&length| length == max_length) {
        return Ok(None);
    }

    let bias: Vec<f32> = input_lengths
        .iter()
        .flat_map(|&length| {
            (0..max_length).map(move |j| if j < length { 0.0 } else { mask_value(dtype) })
        })
        .collect();

    Tensor::from_vec(bias, (input_lengths.len(), 1, 1, max_length), device)?
        .to_dtype(dtype)
        .map(Some)
}

pub struct RotaryEmbedding {
    pub inv_freqs: Tensor,
    pub dim: usize,
}

impl RotaryEmbedding {
    // NOTE: The RoPE dimension follows the head dimension of each layer type, which differs
    // between the sliding (local) and the full (global) attention layers
    pub fn load(
        config: &EmbeddingGemma2TextConfig,
        attention_type: AttentionType,
        device: &Device,
    ) -> Result<Self> {
        let dim = match config
            .layer_types
            .iter()
            .position(|layer_type| *layer_type == attention_type)
        {
            Some(layer_idx) => config.layer_dims(layer_idx).0,
            None => config.head_dim,
        };
        let rope_theta = match config.rope_parameters.get(&attention_type) {
            Some(rope_parameters) => rope_parameters.rope_theta,
            None => candle::bail!(
                "`rope_parameters` for `{attention_type:?}` is not defined in the `config.json`"
            ),
        };
        Ok(Self {
            inv_freqs: get_inv_freqs(dim, rope_theta, device, None)?,
            dim,
        })
    }

    // Computes the `cos` and `sin` of shape `(batch_size, 1, seq_len, dim)` for the given
    // positions, on the fly to avoid caching the (very large) `max_position_embeddings`
    pub fn cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len) = position_ids.dims2()?;
        let positions = position_ids
            .flatten_all()?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?;
        let freqs = positions.broadcast_mul(&self.inv_freqs.unsqueeze(0)?)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let shape = (batch_size, 1, seq_len, self.dim);
        Ok((
            freqs.cos()?.to_dtype(dtype)?.reshape(shape)?,
            freqs.sin()?.to_dtype(dtype)?.reshape(shape)?,
        ))
    }
}

pub struct EmbeddingGemma2Model {
    embed_tokens: Embedding,
    embed_scale: f64,
    ple: EmbeddingGemma2PLE,
    layers: Vec<EmbeddingGemma2Layer>,
    norm: RMSNorm,
    embedding_projection: Linear,

    rotary_full: RotaryEmbedding,
    rotary_sliding: RotaryEmbedding,
    sliding_window: usize,

    pad_token_id: u32,
    pool: Pool,

    dtype: DType,
    device: Device,

    span: tracing::Span,
}

impl EmbeddingGemma2Model {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2Config,
        model_type: ModelType,
    ) -> Result<Self> {
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
                EmbeddingGemma2Layer::load(vb.pp(format!("layers.{layer_idx}")), config, layer_idx)
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
            sliding_window: config.sliding_window,
            pad_token_id: config.pad_token_id,
            pool,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;
        let shape = (batch_size, max_length);

        let elems = batch_size * max_length;
        let mut input_ids = Vec::with_capacity(elems);
        let mut position_ids = Vec::with_capacity(elems);
        let mut input_lengths = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let start = batch.cumulative_seq_lengths[i] as usize;
            let end = batch.cumulative_seq_lengths[i + 1] as usize;
            input_lengths.push(end - start);

            input_ids.extend_from_slice(&batch.input_ids[start..end]);
            position_ids.extend_from_slice(&batch.position_ids[start..end]);
            for _ in 0..(max_length - (end - start)) {
                input_ids.push(self.pad_token_id);
                position_ids.push(0);
            }
        }

        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;

        let padding_bias = padding_bias(&input_lengths, max_length, self.dtype, &self.device)?;
        let full_plan = AttentionPlan::new(
            padding_bias.clone(),
            max_length,
            None,
            self.dtype,
            &self.device,
        )?;
        let sliding_plan = AttentionPlan::new(
            padding_bias,
            max_length,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;

        let (cos_full, sin_full) = self.rotary_full.cos_sin(&position_ids, self.dtype)?;
        let (cos_sliding, sin_sliding) = self.rotary_sliding.cos_sin(&position_ids, self.dtype)?;

        let inputs_embeds = (self.embed_tokens.forward(&input_ids)? * self.embed_scale)?;
        let mut hidden_states = inputs_embeds.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let per_layer_input = self.ple.forward(&inputs_embeds, layer_idx)?;
            let (plan, cos, sin) = match layer.attention_type {
                AttentionType::FullAttention => (&full_plan, &cos_full, &sin_full),
                AttentionType::SlidingAttention => (&sliding_plan, &cos_sliding, &sin_sliding),
            };
            hidden_states = layer.forward(&hidden_states, &per_layer_input, plan, cos, sin)?;
        }

        let (outputs, _) = self.norm.forward(&hidden_states, None)?;
        // NOTE: Projecting per token is equivalent to projecting after mean pooling
        // NOTE: Pooling is done in fp32, as summing up to `max_input_length` token embeddings
        // would overflow in fp16
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
                    let length = input_lengths[i];
                    match self.pool {
                        Pool::Mean => outputs.i((i, ..length))?.sum_keepdim(0)? / (length as f64),
                        Pool::LastToken => outputs.i((i..i + 1, length - 1)),
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
            let results = batch
                .raw_indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    outputs.i((i, ..input_lengths[i]))
                })
                .collect::<Result<Vec<_>>>()?;
            Some(Tensor::cat(&results, 0)?)
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for EmbeddingGemma2Model {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
