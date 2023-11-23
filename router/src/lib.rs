use anyhow::Result;
/// Text Embedding Inference Webserver
use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use text_embeddings_core::infer::Infer;
use text_embeddings_core::TextEmbeddingsError;

mod prometheus;

#[cfg(feature = "http")]
mod http;

#[cfg(feature = "grpc")]
mod grpc;
mod shutdown;

pub async fn run(infer: Infer, info: Info, addr: SocketAddr) -> Result<()> {
    let prom_builder = prometheus::prometheus_builer(info.max_input_length)?;

    if cfg!(feature = "http") {
        #[cfg(feature = "http")]
        {
            return http::server::run(infer, info, addr, prom_builder).await;
        }
    }

    if cfg!(feature = "grpc") {
        #[cfg(feature = "grpc")]
        {
            return grpc::server::run(infer, info, addr, prom_builder).await;
        }
    }

    anyhow::bail!("You must use one of `http` or `grpc`");
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct EmbeddingModel {
    #[cfg_attr(feature = "http", schema(example = "cls"))]
    pub pooling: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct ClassifierModel {
    #[cfg_attr(feature = "http", schema(example = json!({"0": "LABEL"})))]
    pub id2label: HashMap<String, String>,
    #[cfg_attr(feature = "http", schema(example = json!({"LABEL": 0})))]
    pub label2id: HashMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Classifier(ClassifierModel),
    Embedding(EmbeddingModel),
    Reranker(ClassifierModel),
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct Info {
    /// Model info
    #[cfg_attr(feature = "http", schema(example = "thenlper/gte-base"))]
    pub model_id: String,
    #[cfg_attr(
        feature = "http",
        schema(nullable = true, example = "fca14538aa9956a46526bd1d0d11d69e19b5a101")
    )]
    pub model_sha: Option<String>,
    #[cfg_attr(feature = "http", schema(example = "float16"))]
    pub model_dtype: String,
    pub model_type: ModelType,
    /// Router Parameters
    #[cfg_attr(feature = "http", schema(example = "128"))]
    pub max_concurrent_requests: usize,
    #[cfg_attr(feature = "http", schema(example = "512"))]
    pub max_input_length: usize,
    #[cfg_attr(feature = "http", schema(example = "2048"))]
    pub max_batch_tokens: usize,
    #[cfg_attr(
        feature = "http",
        schema(nullable = true, example = "null", default = "null")
    )]
    pub max_batch_requests: Option<usize>,
    #[cfg_attr(feature = "http", schema(example = "32"))]
    pub max_client_batch_size: usize,
    #[cfg_attr(feature = "http", schema(example = "4"))]
    pub tokenization_workers: usize,
    /// Router Info
    #[cfg_attr(feature = "http", schema(example = "0.5.0"))]
    pub version: &'static str,
    #[cfg_attr(feature = "http", schema(nullable = true, example = "null"))]
    pub sha: Option<&'static str>,
    #[cfg_attr(feature = "http", schema(nullable = true, example = "null"))]
    pub docker_label: Option<&'static str>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub enum ErrorType {
    Unhealthy,
    Backend,
    Overloaded,
    Validation,
    Tokenizer,
}

#[derive(Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct ErrorResponse {
    pub error: String,
    pub error_type: ErrorType,
}

impl From<TextEmbeddingsError> for ErrorResponse {
    fn from(err: TextEmbeddingsError) -> Self {
        let error_type = match err {
            TextEmbeddingsError::Tokenizer(_) => ErrorType::Tokenizer,
            TextEmbeddingsError::Validation(_) => ErrorType::Validation,
            TextEmbeddingsError::Overloaded(_) => ErrorType::Overloaded,
            TextEmbeddingsError::Backend(_) => ErrorType::Backend,
        };
        Self {
            error: err.to_string(),
            error_type,
        }
    }
}
