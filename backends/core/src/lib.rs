#[cfg(feature = "clap")]
use clap::ValueEnum;
use nohash_hasher::IntMap;
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

#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
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
}

impl fmt::Display for Pool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Pool::Cls => write!(f, "cls"),
            Pool::Mean => write!(f, "mean"),
            Pool::Splade => write!(f, "splade"),
            Pool::LastToken => write!(f, "last_token"),
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
}
