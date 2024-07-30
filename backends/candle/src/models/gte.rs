use crate::layers::HiddenAct;
use crate::models::PositionEmbeddingType;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NTKScaling {
    pub factor: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum RopeScaling {
    Ntk(NTKScaling),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GTEConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_type: String,
    pub layer_norm_eps: f32,
    pub position_embedding_type: PositionEmbeddingType,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub logn_attention_scale: bool,
    #[serde(default)]
    pub logn_attention_clip1: bool,
}
