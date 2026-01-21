pub mod download;
pub mod infer;
pub mod queue;
pub mod templates;
pub mod tokenization;

use text_embeddings_backend::BackendError;
use thiserror::Error;
use tokio::sync::TryAcquireError;

#[derive(Error, Debug)]
pub enum TextEmbeddingsError {
    #[error("tokenizer error {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("Input validation error: {0}")]
    Validation(String),
    #[error("Input validation error: {0}")]
    Empty(String),
    #[error("Model is overloaded")]
    Overloaded(#[from] TryAcquireError),
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),
}
