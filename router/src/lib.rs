/// Text Embedding Inference Webserver

use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use text_embeddings_core::infer::Infer;

#[cfg(feature = "http")]
mod http;

pub async fn run(
    infer: Infer,
    info: Info,
    addr: SocketAddr,
) -> Result<(), BoxError> {
    if cfg!(feature = "http") {
        #[cfg(feature = "http")]
        {
            return http::server::run(infer, info, addr).await;
        }
    }
    panic!();
}

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct EmbeddingModel {
    #[schema(example = "cls")]
    pub pooling: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct ClassifierModel {
    #[schema(example = json!({"0": "LABEL"}))]
    pub id2label: HashMap<String, String>,
    #[schema(example = json!({"LABEL": 0}))]
    pub label2id: HashMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Classifier(ClassifierModel),
    Embedding(EmbeddingModel),
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct Info {
    /// Model info
    #[schema(example = "thenlper/gte-base")]
    pub model_id: String,
    #[schema(nullable = true, example = "fca14538aa9956a46526bd1d0d11d69e19b5a101")]
    pub model_sha: Option<String>,
    #[schema(example = "float16")]
    pub model_dtype: String,
    pub model_type: ModelType,
    /// Router Parameters
    #[schema(example = "128")]
    pub max_concurrent_requests: usize,
    #[schema(example = "512")]
    pub max_input_length: usize,
    #[schema(example = "2048")]
    pub max_batch_tokens: usize,
    #[schema(nullable = true, example = "null", default = "null")]
    pub max_batch_requests: Option<usize>,
    #[schema(example = "32")]
    pub max_client_batch_size: usize,
    #[schema(example = "4")]
    pub tokenization_workers: usize,
    /// Router Info
    #[schema(example = "0.5.0")]
    pub version: &'static str,
    #[schema(nullable = true, example = "null")]
    pub sha: Option<&'static str>,
    #[schema(nullable = true, example = "null")]
    pub docker_label: Option<&'static str>,
}
