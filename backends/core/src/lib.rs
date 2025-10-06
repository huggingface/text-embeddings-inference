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

/// Input for a single listwise reranking block
#[derive(Debug, Clone)]
pub struct ListwiseBlockInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub embed_token_id: u32,
    pub rerank_token_id: u32,
    pub doc_count: usize,
}

/// Output from a single listwise reranking block
#[derive(Debug, Clone)]
pub struct ListwiseBlockOutput {
    pub query_embedding: Vec<f32>, // 512-d (Jina v3 projector output dimension)
    pub doc_embeddings: Vec<Vec<f32>>, // per-doc 512-d (same dimension)
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

    /// Opt-in listwise reranking support
    fn embed_listwise_block(
        &self,
        _input: ListwiseBlockInput,
    ) -> Result<ListwiseBlockOutput, BackendError> {
        Err(BackendError::Unsupported(
            "listwise reranking not supported".into(),
        ))
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ModelType {
    Classifier,
    Embedding(Pool),
}

#[derive(Debug, PartialEq, Clone, Deserialize)]
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
    #[error("Weights not found: {0}")]
    WeightsNotFound(String),
    #[error("Operation not supported: {0}")]
    Unsupported(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBackend;

    impl Backend for MockBackend {
        fn health(&self) -> Result<(), BackendError> {
            Ok(())
        }

        fn is_padded(&self) -> bool {
            false
        }

        fn embed(&self, _batch: Batch) -> Result<Embeddings, BackendError> {
            Ok(Embeddings::default())
        }

        fn predict(&self, _batch: Batch) -> Result<Predictions, BackendError> {
            Ok(Predictions::default())
        }
    }

    #[test]
    fn test_listwise_block_input_creation() {
        let input = ListwiseBlockInput {
            input_ids: vec![1, 2, 3, 4],
            attention_mask: vec![1, 1, 1, 1],
            embed_token_id: 100,
            rerank_token_id: 101,
            doc_count: 2,
        };

        assert_eq!(input.input_ids.len(), 4);
        assert_eq!(input.doc_count, 2);
    }

    #[test]
    fn test_listwise_block_output_creation() {
        let output = ListwiseBlockOutput {
            query_embedding: vec![0.1; 512],
            doc_embeddings: vec![vec![0.2; 512], vec![0.3; 512]],
        };

        assert_eq!(output.query_embedding.len(), 512);
        assert_eq!(output.doc_embeddings.len(), 2);
        assert_eq!(output.doc_embeddings[0].len(), 512);
    }

    #[test]
    fn test_backend_listwise_default_error() {
        let backend = MockBackend;
        let input = ListwiseBlockInput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            embed_token_id: 100,
            rerank_token_id: 101,
            doc_count: 1,
        };

        let result = backend.embed_listwise_block(input);
        assert!(result.is_err());

        if let Err(BackendError::Unsupported(msg)) = result {
            assert!(msg.contains("listwise reranking not supported"));
        } else {
            panic!("Expected Unsupported error");
        }
    }
}
