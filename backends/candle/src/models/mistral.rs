use crate::layers::{HiddenAct, RopeParameters, RopeScaling};

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MistralConfig {
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
    pub rope_theta: Option<f32>,
    pub rope_parameters: Option<RopeParameters>,
    pub sliding_window: Option<usize>,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub use_bidirectional_attention: Option<bool>,
}
