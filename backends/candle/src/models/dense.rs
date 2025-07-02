use crate::layers::Linear;
use candle::{Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub enum DenseActivation {
    Tanh,
    Identity,
}

impl DenseActivation {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Tanh => x.tanh(),
            Self::Identity => Ok(x.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DenseConfig {
    in_features: usize,
    out_features: usize,
    bias: bool,
    activation_function: Option<String>,
}

pub trait DenseLayer {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor>;
}

#[derive(Debug)]
pub struct Dense {
    linear: Linear,
    activation: DenseActivation,
    span: tracing::Span,
}

impl Dense {
    pub fn load(vb: VarBuilder, config: &DenseConfig) -> Result<Self> {
        let weight = vb.get((config.out_features, config.in_features), "linear.weight")?;
        let bias = if config.bias {
            Some(vb.get(config.out_features, "linear.bias")?)
        } else {
            None
        };

        // Here we cannot leverage HiddenAct, since the activation functions for the
        // 2_Dense/config.json are defined as PyTorch imports instead, so the deserialization would
        // be different, as well as the range of commonly used activation functions (mainly tanh
        // and identity)
        let activation = match config.activation_function {
            // e.g. https://huggingface.co/sentence-transformers/LaBSE/blob/main/2_Dense/config.json
            Some(ref act) if act == "torch.nn.modules.activation.Tanh" => DenseActivation::Tanh,
            // e.g. https://huggingface.co/NovaSearch/stella_en_400M_v5/blob/main/2_Dense/config.json
            Some(ref act) if act == "torch.nn.modules.linear.Identity" => DenseActivation::Identity,
            _ => DenseActivation::Identity,
        };

        let linear = Linear::new(weight, bias, None);

        Ok(Self {
            linear,
            activation,
            span: tracing::span!(tracing::Level::TRACE, "dense"),
        })
    }
}

impl DenseLayer for Dense {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states = self.linear.forward(hidden_states)?;
        self.activation.forward(&hidden_states)
    }
}
