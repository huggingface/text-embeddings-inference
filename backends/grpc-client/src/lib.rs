//! Text Embedding backend gRPC client library

mod client;
#[allow(clippy::derive_partial_eq_without_eq)]
mod pb;

pub use client::Client;
pub use pb::embedding::v1::Embedding;
pub use pb::embedding::v1::HealthResponse;
use thiserror::Error;
use tonic::transport;
use tonic::Status;

#[derive(Error, Debug, Clone)]
pub enum ClientError {
    #[error("Could not connect to Text Embedding server: {0}")]
    Connection(String),
    #[error("Server error: {0}")]
    Inference(String),
}

impl From<Status> for ClientError {
    fn from(err: Status) -> Self {
        let err = Self::Inference(err.message().to_string());
        tracing::error!("{err}");
        err
    }
}

impl From<transport::Error> for ClientError {
    fn from(err: transport::Error) -> Self {
        let err = Self::Connection(err.to_string());
        tracing::error!("{err}");
        err
    }
}

pub type Result<T> = std::result::Result<T, ClientError>;
