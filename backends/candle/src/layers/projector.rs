//! Jina v3 Reranker MLP Projector
//!
//! Architecture: Linear(hidden_size → hidden_size/2, bias=False) → ReLU → Linear(hidden_size/2 → 512, bias=False)

use candle::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

#[derive(Debug)]
pub struct Projector {
    fc1: Linear,
    fc2: Linear,
}

impl Projector {
    /// Load projector weights from VarBuilder
    ///
    /// # Arguments
    ///
    /// * `vb` - VarBuilder (should be set to model dtype via vb.set_dtype())
    /// * `hidden_size` - Model hidden size (e.g., 1024 for Qwen3)
    pub fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        let latent_size = hidden_size / 2; // modeling.py: hidden_size → hidden_size/2 → 512

        // VarBuilder paths map to safetensors keys:
        // VarBuilder is already scoped to "projector" from lib.rs:240
        // So vb.pp("0") → "projector.0.weight" (not vb.pp("projector").pp("0"))
        let w1 = vb.pp("0").get((latent_size, hidden_size), "weight")?;
        let w2 = vb.pp("2").get((512, latent_size), "weight")?;

        // Verify projector has no bias (modeling.py: bias=False)
        // Check existence by attempting to load - minimal overhead
        if vb.pp("0").get((latent_size,), "bias").is_ok() || vb.pp("2").get((512,), "bias").is_ok()
        {
            candle::bail!(
                "Projector must be bias-free (bias=False per Jina v3 spec). \
                 This model may not be compatible. Verify weights or use --reranker-mode pairwise"
            );
        }

        let fc1 = Linear::new(w1, None);
        let fc2 = Linear::new(w2, None);
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(hidden)?.relu()?;
        self.fc2.forward(&x)
    }
}
