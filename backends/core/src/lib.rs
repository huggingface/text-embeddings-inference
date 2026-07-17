#[cfg(feature = "clap")]
use clap::ValueEnum;
use nohash_hasher::IntMap;
use serde::Deserialize;
use std::fmt;
use thiserror::Error;

#[derive(Debug)]
pub struct Batch {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub cumulative_seq_lengths: Vec<u32>,
    pub max_length: u32,
    pub pooled_indices: Vec<u32>,
    pub raw_indices: Vec<u32>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.cumulative_seq_lengths.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub enum Embedding {
    Pooled(Vec<f32>),
    All(Vec<Vec<f32>>),
}

pub type Embeddings = IntMap<usize, Embedding>;
pub type Predictions = IntMap<usize, Vec<f32>>;

pub trait Backend {
    fn health(&self) -> Result<(), BackendError>;
    fn max_batch_size(&self) -> Option<usize> {
        None
    }

    fn is_padded(&self) -> bool;

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError>;

    fn predict(&self, batch: Batch) -> Result<Predictions, BackendError>;
}

#[derive(Debug, PartialEq, Clone)]
pub enum ModelType {
    Classifier,
    Embedding(Pool),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "snake_case")]
pub enum Pool {
    /// Select the CLS token as embedding
    Cls,
    /// Apply Mean pooling to the model embeddings
    Mean,
    /// Apply SPLADE (Sparse Lexical and Expansion) to the model embeddings.
    /// This option is only available if the loaded model is a `ForMaskedLM` Transformer
    /// model.
    Splade,
    /// Select the last token as embedding
    LastToken,
    /// Apply the bge-m3 sparse (lexical weights) head to the model embeddings.
    /// This option is only available if the loaded model ships a `sparse_linear.pt`
    /// head, i.e. `BAAI/bge-m3` and its fine-tunes.
    ///
    /// Unlike `Splade`, the weights are indexed by the *input* token ids rather than by
    /// vocabulary argmax, so the pooled vector is sparse by construction.
    // `m3_sparse` is the name the model card and `Display` use; the kebab-case alias keeps the
    // CLI consistent with the other variants (e.g. `last-token`).
    #[cfg_attr(feature = "clap", value(name = "m3_sparse", alias = "m3-sparse"))]
    M3Sparse,
}

impl fmt::Display for Pool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Pool::Cls => write!(f, "cls"),
            Pool::Mean => write!(f, "mean"),
            Pool::Splade => write!(f, "splade"),
            Pool::LastToken => write!(f, "last_token"),
            Pool::M3Sparse => write!(f, "m3_sparse"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_display_matches_the_serde_name() {
        for (pool, name) in [
            (Pool::Cls, "cls"),
            (Pool::Mean, "mean"),
            (Pool::Splade, "splade"),
            (Pool::LastToken, "last_token"),
            (Pool::M3Sparse, "m3_sparse"),
        ] {
            assert_eq!(pool.to_string(), name);
            assert_eq!(
                serde_json::from_str::<Pool>(&format!("\"{name}\"")).unwrap(),
                pool
            );
        }
    }

    #[cfg(feature = "clap")]
    #[test]
    fn pooling_arg_accepts_m3_sparse_and_rejects_unknown_values() {
        use clap::ValueEnum;

        assert_eq!(Pool::from_str("m3_sparse", false).unwrap(), Pool::M3Sparse);
        // kebab-case alias, matching how the other variants are spelled on the CLI
        assert_eq!(Pool::from_str("m3-sparse", false).unwrap(), Pool::M3Sparse);
        // pre-existing variants keep parsing
        assert_eq!(Pool::from_str("splade", false).unwrap(), Pool::Splade);

        for invalid in ["m3sparse", "m3", "sparse", "bogus", ""] {
            assert!(
                Pool::from_str(invalid, false).is_err(),
                "`{invalid}` must not parse as a pooling mode"
            );
        }
    }
}

#[derive(Debug, Error, Clone)]
pub enum BackendError {
    #[error("No backend found")]
    NoBackend,
    #[error("Could not start backend: {0}")]
    Start(String),
    #[error("{0}")]
    Inference(String),
    #[error("Backend is unhealthy")]
    Unhealthy,
    #[error("Weights not found: {0}")]
    WeightsNotFound(String),
}
