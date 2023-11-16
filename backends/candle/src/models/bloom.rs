use crate::layers::LayerNorm;
use candle::{DType, Device, Result, Tensor};
use candle_nn::{Dropout, Embedding, Linear, VarBuilder};
use serde::Deserialize;
use text_embeddings_backend_core::Embedding;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub layer_norm_epsilon: f64,
    pub initializer_range: f64,
    pub use_cache: bool,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub apply_residual_connection_post_layernorm: bool,
    pub hidden_dropout: f64,
    pub attention_dropout: f64,
    pub pretraining_tp: usize,
    pub slow_but_exact: bool,
    pub architectures: Vec<String>,
    pub attention_softmax_in_fp32: bool,
    pub seq_length: usize,
    pub pad_token_id: usize,
    pub masked_softmax_fusion: bool,
    pub model_type: Option<String>,
    pub n_embed: usize,
    pub num_attention_heads: usize,
}

fn gelu_forward(x: &Tensor) -> Result<Tensor> {
    (x * 0.5)? * (1.0 + ((0.79788456 * x)? * (1.0 + ((0.044715 * x)? * x)?))?.tanh()?)?
}

pub struct BloomAttention {
    pretraining_tp: usize,
    slow_but_exact: bool,

    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    split_size: usize,
    hidden_dropout: f64,

    inv_norm_factor: f64,
    beta: f64,

    query_key_value: Linear,
    dense: Linear,
    attention_dropout: Dropout,
}

impl BloomAttention {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let pretraining_tp = config.pretraining_tp;
        let slow_but_exact = config.slow_but_exact;

        let hidden_size = config.hidden_size;
        let num_heads = config.n_head;
        let head_dim = config.hidden_size / config.n_head;
        let split_size = config.hidden_size;
        let hidden_dropout = config.hidden_dropout;

        if head_dim * num_heads != hidden_size {
            candle::bail!(
                format!("hidden_size must be divisible by num_heads (got hidden_size: {hidden_size} and num_heads: {num_heads}"),
            )
        }

        let inv_norm_factor = 1. / (head_dim as f64).sqrt();
        let beta = 1.;

        let qkv_weight = vb
            .pp("query_key_value")
            .get((hidden_size * 3, hidden_size), "weight")?;
        let qkv_bias = vb.pp("query_key_value").get(hidden_size * 3, "bias")?;
        let query_key_value = Linear::new(qkv_weight, Some(qkv_bias));

        let dense_weight = vb
            .pp("query_key_value")
            .get((hidden_size, hidden_size), "weight")?;
        let dense_bias = vb.pp("dense").get(hidden_size, "bias")?;
        let dense = Linear::new(dense_weight, Some(dense_bias));

        let attention_dropout = Dropout::new(config.attention_dropout);

        Ok(Self {
            pretraining_tp,
            slow_but_exact,
            hidden_size,
            num_heads,
            head_dim,
            split_size,
            hidden_dropout,
            inv_norm_factor,
            beta,
            query_key_value,
            dense,
            attention_dropout,
        })
    }
}

pub struct BloomMlp {
    hidden_size: usize,

    pretraining_tp: usize,
    slow_but_exact: bool,
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    hidden_dropout: f64,
}

impl BloomMlp {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let hidden_size = config.hidden_size;

        let pretraining_tp = config.pretraining_tp;
        let slow_but_exact = config.slow_but_exact;

        let dense_h_to_4h_weight = vb
            .pp("dense_h_to_4h")
            .get((hidden_size * 4, hidden_size), "weight")?;
        let dense_h_to_4h_bias = vb.pp("dense_h_to_4h").get(hidden_size * 4, "bias")?;
        let dense_h_to_4h = Linear::new(dense_h_to_4h_weight, Some(dense_h_to_4h_bias));

        let dense_4h_to_h_weight = vb
            .pp("dense_4h_to_h")
            .get((hidden_size, hidden_size * 4), "weight")?;
        let dense_4h_to_h_bias = vb.pp("dense_4h_to_h").get(hidden_size, "bias")?;
        let dense_4h_to_h = Linear::new(dense_4h_to_h_weight, Some(dense_4h_to_h_bias));

        let hidden_dropout = config.hidden_dropout;

        Ok(Self {
            hidden_size,
            pretraining_tp,
            slow_but_exact,
            dense_h_to_4h,
            dense_4h_to_h,
            hidden_dropout,
        })
    }
}

pub struct BloomBlock {
    hidden_size: usize,
    input_layernorm: LayerNorm,
    num_heads: usize,
    self_attention: BloomAttention,
    post_attention_layernorm: LayerNorm,
    mlp: BloomMlp,
    apply_residual_connection_post_layernorm: bool,
    hidden_dropout: f64,
}

impl BloomBlock {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let hidden_size = config.hidden_size;

        let input_layernorm = LayerNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.layer_norm_epsilon,
        );
        let num_heads = config.n_head;
        let self_attention = BloomAttention::load(vb.pp("self_attention"), config)?;
        let post_attention_layernorm = LayerNorm::new(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.layer_norm_epsilon,
        );

        let mlp = BloomMlp::load(vb.pp("mlp"), config)?;

        let apply_residual_connection_post_layernorm =
            config.apply_residual_connection_post_layernorm;
        let hidden_dropout = config.hidden_dropout;

        Ok(Self {
            hidden_size,
            input_layernorm,
            num_heads,
            self_attention,
            post_attention_layernorm,
            mlp,
            apply_residual_connection_post_layernorm,
            hidden_dropout,
        })
    }
}

pub struct BloomModel {
    embed_dim: usize,
    num_heads: usize,
    word_embeddings: Embedding,
    word_embeddings_layernorm: LayerNorm,
    h: Vec<BloomBlock>,
    ln_f: LayerNorm,
    gradient_checkpointing: bool,

    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl BloomModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.n_head;

        let word_embeddings_weights = vb.pp("word_embeddings").get(config.vocab_size, "weight")?;
        let word_embeddings = Embedding::new(word_embeddings_weights, config.hidden_size);

        let word_embeddings_layernorm = LayerNorm::load(
            vb.pp("word_embeddings_layernorm"),
            config.hidden_size,
            config.layer_norm_epsilon,
        )?;

        let h = (0..config.num_hidden_layers)
            .map(|index| BloomBlock::load(vb.pp(format!("h.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let ln_f = LayerNorm::load(vb.pp("ln_f"), config.hidden_size, config.layer_norm_epsilon)?;

        let gradient_checkpointing = config.gradient_checkpointing;

        Ok(Self {
            embed_dim,
            num_heads,
            word_embeddings,
            word_embeddings_layernorm,
            h,
            ln_f,
            gradient_checkpointing,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }
}
