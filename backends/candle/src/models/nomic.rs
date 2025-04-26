use crate::layers::{
    apply_rotary, get_cos_sin, get_cublas_lt_wrapper, get_inv_freqs, HiddenAct, LayerNorm, Linear,
};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use candle_transformers::models::deepseek2::{BincountOp, NonZeroOp, TopKLastDimOp, TopKOutput};
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NomicConfig {
    pub prenorm: bool,
    pub rotary_emb_fraction: f32,
    pub qkv_proj_bias: bool,
    pub rotary_emb_base: f32,
    pub rotary_emb_interleaved: bool,
    pub mlp_fc1_bias: bool,
    pub mlp_fc2_bias: bool,
    pub rotary_scaling_factor: Option<f32>,
    #[serde(default = "default_max_trained_positions")]
    pub max_trained_positions: usize,

    pub moe_every_n_layers: Option<usize>,
    pub moe_normalize_expert_weights: Option<bool>,
    pub moe_top_k: Option<usize>,
    pub num_experts: Option<usize>,

    pub n_embd: usize,
    pub n_head: usize,
    pub n_inner: usize,
    pub n_layer: usize,
    pub n_positions: usize,

    pub activation_function: HiddenAct,

    pub vocab_size: usize,
    pub type_vocab_size: usize,
    pub layer_norm_epsilon: f32,
}

fn default_max_trained_positions() -> usize {
    2048
}

impl NomicConfig {
    // For now, we only support these parameters
    pub fn valid(&self) -> bool {
        !self.prenorm
            && self.rotary_emb_fraction == 1.0
            && !self.rotary_emb_interleaved
            && self.type_vocab_size > 0
    }
}

#[derive(Debug)]
pub struct NomicBertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,

    span: tracing::Span,
}

impl NomicBertEmbeddings {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(
                vb.pp("embeddings.word_embeddings")
                    .get((config.vocab_size, config.n_embd), "weight")?,
                config.n_embd,
            ),
            token_type_embeddings: Embedding::new(
                vb.pp("embeddings.token_type_embeddings")
                    .get((config.type_vocab_size, config.n_embd), "weight")?,
                config.n_embd,
            ),
            layer_norm: LayerNorm::load(vb.pp("emb_ln"), config.n_embd, config.layer_norm_epsilon)?,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;

        let embeddings = self
            .layer_norm
            .forward(&input_embeddings, Some(&token_type_embeddings))?;

        Ok(embeddings)
    }
}

pub struct NomicBertGatedMLP {
    fc1: Linear,
    fc2: Linear,

    span: tracing::Span,
}

impl NomicBertGatedMLP {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let intermediate_size = config.n_inner;

        let fc11_weight = vb
            .pp("fc11")
            .get((intermediate_size, config.n_embd), "weight")?;
        let fc12_weight = vb
            .pp("fc12")
            .get((intermediate_size, config.n_embd), "weight")?;
        let fc1_weight = Tensor::cat(&[fc11_weight, fc12_weight], 0)?;

        let fc1_bias = if config.mlp_fc1_bias {
            let fc11_bias = vb.pp("fc11").get((intermediate_size,), "bias")?;
            let fc12_bias = vb.pp("fc12").get((intermediate_size,), "bias")?;
            Some(Tensor::cat(&[fc11_bias, fc12_bias], 0)?)
        } else {
            None
        };

        let fc1 = Linear::new(
            fc1_weight,
            fc1_bias,
            Some(config.activation_function.clone()),
        );

        let fc2_weight = vb
            .pp("fc2")
            .get((config.n_embd, intermediate_size), "weight")?;
        let fc2_bias = if config.mlp_fc2_bias {
            Some(vb.pp("fc2").get((config.n_embd,), "bias")?)
        } else {
            None
        };
        let fc2 = Linear::new(fc2_weight, fc2_bias, None);

        Ok(Self {
            fc1,
            fc2,
            span: tracing::span!(tracing::Level::TRACE, "gated_mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.fc1.forward(hidden_states)?;
        self.fc2.forward(&gate_up_states)
    }
}

pub struct NomicBertMLP {
    fc1: Linear,
    fc2: Linear,

    span: tracing::Span,
}

impl NomicBertMLP {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let intermediate_size = config.n_inner;

        let fc1_weight = vb
            .pp("fc1")
            .get((intermediate_size, config.n_embd), "weight")?;
        let fc1_bias = if config.mlp_fc1_bias {
            Some(vb.pp("fc1").get((intermediate_size,), "bias")?)
        } else {
            None
        };
        let fc1 = Linear::new(
            fc1_weight,
            fc1_bias,
            Some(config.activation_function.clone()),
        );

        let fc2_weight = vb
            .pp("fc2")
            .get((config.n_embd, intermediate_size), "weight")?;
        let fc2_bias = if config.mlp_fc2_bias {
            Some(vb.pp("fc2").get((config.n_embd,), "bias")?)
        } else {
            None
        };
        let fc2 = Linear::new(fc2_weight, fc2_bias, None);

        Ok(Self {
            fc1,
            fc2,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let y = self.fc1.forward(hidden_states)?;
        self.fc2.forward(&y)
    }
}

pub struct NomicRouter {
    layer: Linear,
    moe_top_k: usize,

    span: tracing::Span,
}

impl NomicRouter {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let num_experts = config.num_experts.unwrap();
        let moe_top_k = config.moe_top_k.unwrap();

        let layer_weight = vb.pp("layer").get((num_experts, config.n_embd), "weight")?;
        let layer = Linear::new(layer_weight, None, None);

        Ok(Self {
            layer,
            moe_top_k,
            span: tracing::span!(tracing::Level::TRACE, "router"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        let weights = hidden_states.reshape(((), hidden_states.dim(D::Minus1)?))?;
        let weights = self.layer.forward(&weights)?.to_dtype(DType::F32)?;
        let weights = candle_nn::ops::softmax_last_dim(&weights)?;

        let TopKOutput { values, indices } = weights.topk(self.moe_top_k)?;

        let values = values.to_dtype(hidden_states.dtype())?;

        Ok((values, indices))
    }
}

pub struct NomicExpertMLP {
    w1: Tensor,
    w2: Tensor,
    activation: HiddenAct,

    span: tracing::Span,
}

impl NomicExpertMLP {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let hidden_size = config.n_embd;
        let ffn_hidden_size = config.n_inner;
        let moe_num_experts = config.num_experts.unwrap();
        let activation = config.activation_function.clone();

        let w1 = vb
            .get((moe_num_experts * ffn_hidden_size, hidden_size), "w1")?
            .reshape((moe_num_experts, ffn_hidden_size, hidden_size))?;
        let w2 = vb
            .get((moe_num_experts * ffn_hidden_size, hidden_size), "w2")?
            .reshape((moe_num_experts, ffn_hidden_size, hidden_size))?;

        Ok(Self {
            w1,
            w2,
            activation,
            span: tracing::span!(tracing::Level::TRACE, "expert_mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, expert_idx: usize) -> Result<Tensor> {
        let _enter = self.span.enter();

        let expert_w1 = self.w1.narrow(0, expert_idx, 1)?.squeeze(0)?.t()?;
        let expert_w2 = self.w2.narrow(0, expert_idx, 1)?.squeeze(0)?;

        let hidden_states = hidden_states.broadcast_matmul(&expert_w1)?;
        let hidden_states = self.activation.forward(&hidden_states)?;

        hidden_states.broadcast_matmul(&expert_w2)
    }
}

pub struct NomicExperts {
    moe_num_experts: usize,
    mlp: NomicExpertMLP,
    bias: Tensor,

    span: tracing::Span,
}

impl NomicExperts {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let moe_num_experts = config.num_experts.unwrap();

        let mlp = NomicExpertMLP::load(vb.pp("mlp"), config)?;

        let bias = vb.get((config.n_embd,), "bias")?;

        Ok(Self {
            moe_num_experts,
            mlp,
            bias,
            span: tracing::span!(tracing::Level::TRACE, "experts"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        top_weights: &Tensor,
        top_experts: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let dims = hidden_states.dims();
        let ndim = dims.len();

        let (bs, seq_len, hidden_size) = match ndim {
            3 => (dims[0], dims[1], dims[2]),
            2 => (1, dims[0], dims[1]),
            _ => unreachable!(),
        };

        let hidden_states = hidden_states.reshape(((), hidden_size))?;

        let mut out = Tensor::zeros_like(&hidden_states)?;

        let counts = top_experts
            .flatten_all()?
            .bincount(self.moe_num_experts as u32)?;

        for (expert_idx, &count) in counts.iter().enumerate().take(self.moe_num_experts) {
            if count == 0u32 {
                continue;
            }

            let idx_top = top_experts.eq(expert_idx as f64)?.nonzero()?.t()?;
            let idx = &idx_top.i(0)?.contiguous()?;
            let top = &idx_top.i(1)?.contiguous()?;

            let expert_out = self
                .mlp
                .forward(&hidden_states.index_select(idx, 0)?, expert_idx)?
                .broadcast_mul(
                    &top_weights
                        .index_select(idx, 0)?
                        .gather(&top.unsqueeze(1)?, 1)?
                        .squeeze(1)?
                        .unsqueeze(D::Minus1)?
                        .to_dtype(hidden_states.dtype())?,
                )?;

            out = out.index_add(idx, &expert_out, 0)?;
        }

        if ndim == 3 {
            out = out.reshape((bs, seq_len, hidden_size))?;
        }

        out.broadcast_add(&self.bias)
    }
}

pub struct NomicMoELayer {
    router: NomicRouter,
    experts: NomicExperts,

    span: tracing::Span,
}

impl NomicMoELayer {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let router = NomicRouter::load(vb.pp("router"), config)?;
        let experts = NomicExperts::load(vb.pp("experts"), config)?;

        Ok(Self {
            router,
            experts,
            span: tracing::span!(tracing::Level::TRACE, "moe"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (top_weights, top_experts) = self.router.forward(hidden_states)?;

        self.experts
            .forward(hidden_states, &top_weights, &top_experts)
    }
}

pub enum NomicMLP {
    MoE(NomicMoELayer),
    GatedMLP(NomicBertGatedMLP),
    Mlp(NomicBertMLP),
}

impl NomicMLP {
    pub fn load(vb: VarBuilder, index: usize, config: &NomicConfig) -> Result<Self> {
        let use_moe = matches!(config.moe_every_n_layers, Some(n) if n > 0 && index % n == 1);

        if use_moe {
            Ok(Self::MoE(NomicMoELayer::load(vb, config)?))
        } else if config.activation_function == HiddenAct::Gelu {
            Ok(Self::Mlp(NomicBertMLP::load(vb, config)?))
        } else {
            Ok(Self::GatedMLP(NomicBertGatedMLP::load(vb, config)?))
        }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE(layer) => layer.forward(hidden_states),
            Self::GatedMLP(layer) => layer.forward(hidden_states),
            Self::Mlp(layer) => layer.forward(hidden_states),
        }
    }
}

struct NomicAttention {
    qkv_linear: Linear,
    out_proj: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,

    softmax_scale: f64,

    span: tracing::Span,
}

impl NomicAttention {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let num_attention_heads = config.n_head;
        let attention_head_size = config.n_embd / config.n_head;
        let hidden_size = config.n_embd;

        let qkv_dim = 3 * num_attention_heads * attention_head_size;

        let qkv_weight = vb.pp("Wqkv").get((qkv_dim, hidden_size), "weight")?;
        let qkv_bias = if config.qkv_proj_bias {
            Some(vb.pp("Wqkv").get((qkv_dim,), "bias")?)
        } else {
            None
        };
        let qkv_linear = Linear::new(qkv_weight, qkv_bias, None);

        let out_proj_weight = vb
            .pp("out_proj")
            .get((hidden_size, hidden_size), "weight")?;
        let out_proj_bias = if config.qkv_proj_bias {
            Some(vb.pp("out_proj").get((hidden_size,), "bias")?)
        } else {
            None
        };
        let out_proj = Linear::new(out_proj_weight, out_proj_bias, None);

        let softmax_scale = 1. / (attention_head_size as f64).sqrt();

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

        let hidden_states = self.out_proj.forward(&context_layer)?;

        Ok(hidden_states)
    }
}

struct NomicBertBlock {
    attention: NomicAttention,
    mlp: NomicMLP,
    post_attention_layer_norm: LayerNorm,
    output_layer_norm: LayerNorm,

    span: tracing::Span,
}

impl NomicBertBlock {
    pub fn load(vb: VarBuilder, index: usize, config: &NomicConfig) -> Result<Self> {
        let attention = NomicAttention::load(vb.pp("attn"), config)?;

        let mlp = NomicMLP::load(vb.pp("mlp"), index, config)?;

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
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let attn_output = self
            .attention
            .forward(hidden_states, attention_bias, cos, sin)?;

        let hidden_states = self
            .post_attention_layer_norm
            .forward(hidden_states, Some(&attn_output))?;

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
            .map(|index| NomicBertBlock::load(vb.pp(format!("layers.{index}")), index, config))
            .collect::<Result<Vec<_>>>()?;

        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(NomicBertEncoder { layers, span })
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

        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_bias, cos, sin)?
        }

        Ok(hidden_states)
    }
}

pub struct NomicBertModel {
    embeddings: NomicBertEmbeddings,
    encoder: NomicBertEncoder,
    pool: Pool,
    pub device: Device,
    dtype: DType,

    rotary_dim: usize,
    max_trained_positions: u32,
    rotary_cache: (Tensor, Tensor),
    scaled_rotary_cache: Option<(Tensor, Tensor)>,

    num_attention_heads: usize,

    span: tracing::Span,
}

impl NomicBertModel {
    pub fn load(vb: VarBuilder, config: &NomicConfig, model_type: ModelType) -> Result<Self> {
        if !config.valid() {
            candle::bail!("config is not supported")
        }

        let pool = match model_type {
            // Classifier models always use CLS pooling
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Nomic")
            }
            ModelType::Embedding(pool) => {
                if pool == Pool::Splade {
                    candle::bail!("`splade` is not supported for Nomic")
                }
                if pool == Pool::LastToken {
                    candle::bail!("`last_token` is not supported for Nomic");
                }
                pool
            }
        };

        let embeddings = NomicBertEmbeddings::load(vb.clone(), config)?;
        let encoder = NomicBertEncoder::load(vb.pp("encoder"), config)?;

        let rotary_dim = encoder.layers[0].attention.attention_head_size;
        let inv_freqs_tensor =
            get_inv_freqs(rotary_dim, config.rotary_emb_base, vb.device(), None)?;
        let rotary_cache = get_cos_sin(config.n_positions, &inv_freqs_tensor, vb.dtype(), true)?;

        let scaled_rotary_cache = if let Some(scaling_factor) = config.rotary_scaling_factor {
            let new_base = (config.rotary_emb_base
                * ((scaling_factor * config.n_positions as f32
                    / config.max_trained_positions as f32)
                    - (scaling_factor - 1.0)))
                .powi((rotary_dim as f32 / (rotary_dim as f32 - 2.0)) as i32);
            let inv_freqs_tensor = get_inv_freqs(rotary_dim, new_base, vb.device(), None)?;
            Some(get_cos_sin(
                config.n_positions,
                &inv_freqs_tensor,
                vb.dtype(),
                true,
            )?)
        } else {
            None
        };

        Ok(Self {
            embeddings,
            encoder,
            pool,
            rotary_dim,
            max_trained_positions: config.max_trained_positions as u32,
            rotary_cache,
            scaled_rotary_cache,
            num_attention_heads: config.n_head,
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

        let cos = cos.reshape((batch_size, 1, max_length, self.rotary_dim))?;
        let sin = sin.reshape((batch_size, 1, max_length, self.rotary_dim))?;

        let embedding_output = self.embeddings.forward(&input_ids, &type_ids)?;

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

impl Model for NomicBertModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
