use crate::layers::HiddenAct;
use crate::layers::Linear;
use crate::models::PositionEmbeddingType;
use candle::{Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::collections::HashMap;

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
    pub id2label: Option<HashMap<String, String>>,
}

pub trait ClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor>;
}

pub struct GTEClassificationHead {
    pooler: Option<Linear>,
    classifier: Linear,
    span: tracing::Span,
}

impl GTEClassificationHead {
    #[allow(dead_code)]
    pub(crate) fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let n_classes = match &config.id2label {
            None => candle::bail!("`id2label` must be set for classifier models"),
            Some(id2label) => id2label.len(),
        };

        let pooler = if let Ok(pooler_weight) = vb
            .pp("pooler.dense")
            .get((config.hidden_size, config.hidden_size), "weight")
        {
            let pooler_bias = vb.pp("pooler.dense").get(config.hidden_size, "bias")?;
            Some(Linear::new(pooler_weight, Some(pooler_bias), None))
        } else {
            None
        };

        let classifier_weight = vb
            .pp("classifier")
            .get((n_classes, config.hidden_size), "weight")?;
        let classifier_bias = vb.pp("classifier").get(n_classes, "bias")?;
        let classifier = Linear::new(classifier_weight, Some(classifier_bias), None);

        Ok(Self {
            classifier,
            pooler,
            span: tracing::span!(tracing::Level::TRACE, "classifier"),
        })
    }
}

impl ClassificationHead for GTEClassificationHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let mut hidden_states = hidden_states.unsqueeze(1)?;
        if let Some(pooler) = self.pooler.as_ref() {
            hidden_states = pooler.forward(&hidden_states)?;
            hidden_states = hidden_states.tanh()?;
        }

        let hidden_states = self.classifier.forward(&hidden_states)?;
        let hidden_states = hidden_states.squeeze(1)?;
        Ok(hidden_states)
    }
}
