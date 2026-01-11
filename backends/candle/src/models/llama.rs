use crate::layers::HiddenAct;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "rope_type", rename_all = "lowercase")]
pub enum RopeScaling {
    Llama3 {
        factor: f32,
        high_freq_factor: f32,
        low_freq_factor: f32,
        original_max_position_embeddings: usize,
    },
    // Add other rope types as needed
}

fn default_use_bidirectional_attention() -> bool {
    false
}

fn default_mlp_bias() -> bool {
    false
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LLamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub rms_norm_eps: f32,
    pub model_type: Option<String>,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_use_bidirectional_attention")]
    pub use_bidirectional_attention: bool,
    pub head_dim: Option<usize>,
    pub attention_bias: Option<bool>,
    pub attention_dropout: Option<f32>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub pad_token_id: Option<usize>,
    #[serde(default = "default_mlp_bias")]
    pub mlp_bias: bool,
    pub pretraining_tp: Option<usize>,
    pub tie_word_embeddings: Option<bool>,
}
