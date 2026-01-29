use crate::models::{Model, Qwen3Config, Qwen3Model};
use candle::{Result, Tensor};
use candle_nn::VarBuilder;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

// Re-export Qwen3Config as PPLX1Config for API compatibility
pub type PPLX1Config = Qwen3Config;

pub struct PPLX1Model {
    inner: Qwen3Model,
}

impl PPLX1Model {
    pub fn load(vb: VarBuilder, config: &PPLX1Config, model_type: ModelType) -> Result<Self> {
        // Validate: PPLX1 only supports mean pooling
        match &model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for PPLX1")
            }
            ModelType::Embedding(pool) => {
                if *pool != Pool::Mean {
                    candle::bail!("PPLX1 only supports mean pooling, got {:?}", pool);
                }
            }
        };

        // Load the underlying Qwen3 model (with use_causal_mask from config)
        let inner = Qwen3Model::load(vb, config, model_type)?;

        Ok(Self { inner })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // Call the underlying Qwen3 model
        let (pooled, raw) = self.inner.forward(batch)?;

        // Apply PPLX1-specific quantization to pooled embeddings
        let pooled = pooled
            .map(|embeddings| {
                embeddings
                    .tanh()                              // Apply tanh: [-1, 1]
                    .and_then(|t| t.affine(127.0, 0.0)) // Scale: [-127, 127]
                    .and_then(|t| t.round())             // Round to integers
            })
            .transpose()?;

        Ok((pooled, raw))
    }
}

impl Model for PPLX1Model {
    fn is_padded(&self) -> bool {
        self.inner.is_padded()
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
