#[cfg(feature = "clap")]
use clap::ValueEnum;
use thiserror::Error;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "clap", derive(Clone, ValueEnum))]
pub enum Pool {
    Cls,
    Mean,
}

#[derive(Debug)]
pub struct Batch {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub cumulative_seq_lengths: Vec<u32>,
    pub max_length: u32,
}

pub type Embedding = Vec<f32>;

pub trait EmbeddingBackend {
    fn health(&self) -> Result<(), BackendError>;

    fn embed(&self, batch: Batch) -> Result<Vec<Embedding>, BackendError>;

    fn max_batch_size(&self) -> Option<usize> {
        None
    }
}

#[derive(Debug, Error, Clone)]
pub enum BackendError {
    #[error("No backend found")]
    NoBackend,
    #[error("Could not start backend: {0}")]
    Start(String),
    #[error("Inference error: {0}")]
    Inference(String),
    #[error("Backend is unhealthy")]
    Unhealthy,
}
