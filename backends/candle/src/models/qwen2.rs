use crate::layers::{HiddenAct, RopeParameters};
use serde::Deserialize;

fn default_is_causal() -> bool {
    tracing::warn!("is_causal not set in Qwen2Config, defaulting to true. e.g. Alibaba-NLP/gte-Qwen2-1.5B-instruct/ was trained with causal=False attention, but jinaai/jina-code-embeddings-0.5b with causal=True. Please set this field explicitly in the huggingface repo to avoid this warning.");
    true
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: Option<f32>,
    pub rope_parameters: Option<RopeParameters>,
    pub sliding_window: Option<usize>,
    pub use_sliding_window: bool,
    #[serde(default = "default_is_causal")]
    pub is_causal: bool,
}
