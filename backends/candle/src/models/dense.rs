use crate::layers::Linear;
use candle::{Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DenseConfig {
    in_features: usize,
    out_features: usize,
    bias: bool,
    #[allow(unused)]
    activation_function: Option<String>,
}

pub trait DenseLayer {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor>;
}

#[derive(Debug)]
pub struct Dense {
    linear: Linear,
    span: tracing::Span,
}

impl Dense {
    pub fn load(vb: VarBuilder, config: &DenseConfig) -> Result<Self> {
        let dense_weight = vb.get((config.out_features, config.in_features), "linear.weight")?;
        let dense_bias = if config.bias {
            Some(vb.get(config.out_features, "linear.bias")?)
        } else {
            None
        };

        let linear = Linear::new(dense_weight, dense_bias, None);

        Ok(Self {
            linear,
            span: tracing::span!(tracing::Level::TRACE, "dense"),
        })
    }
}

impl DenseLayer for Dense {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.linear.forward(hidden_states)?.tanh()
    }
}
