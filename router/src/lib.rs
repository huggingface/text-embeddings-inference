/// Text Embedding Inference Webserver
pub mod server;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct Info {
    /// Model info
    #[schema(example = "thenlper/gte-base")]
    pub model_id: String,
    #[schema(nullable = true, example = "fca14538aa9956a46526bd1d0d11d69e19b5a101")]
    pub model_sha: Option<String>,
    #[schema(example = "float16")]
    pub model_dtype: String,
    #[schema(example = "cls")]
    pub model_pooling: String,
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

#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum Input {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct OpenAICompatRequest {
    pub input: Input,
    #[allow(dead_code)]
    #[schema(nullable = true, example = "null")]
    model: Option<String>,
    #[allow(dead_code)]
    #[schema(nullable = true, example = "null")]
    user: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatEmbedding {
    #[schema(example = "embedding")]
    object: &'static str,
    #[schema(example = json ! (["0.0", "1.0", "2.0"]))]
    embedding: Vec<f32>,
    #[schema(example = "0")]
    index: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatUsage {
    #[schema(example = "512")]
    prompt_tokens: usize,
    #[schema(example = "512")]
    total_tokens: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatResponse {
    #[schema(example = "list")]
    object: &'static str,
    data: Vec<OpenAICompatEmbedding>,
    #[schema(example = "thenlper/gte-base")]
    model: String,
    usage: OpenAICompatUsage,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedRequest {
    pub inputs: Input,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub truncate: bool,
}

#[derive(Serialize, ToSchema)]
#[schema(example = json ! ([["0.0", "1.0", "2.0"]]))]
pub(crate) struct EmbedResponse(Vec<Vec<f32>>);

#[derive(Serialize, ToSchema)]
pub(crate) enum ErrorType {
    Unhealthy,
    Backend,
    Overloaded,
    Validation,
    Tokenizer,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ErrorResponse {
    pub error: String,
    pub error_type: ErrorType,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatErrorResponse {
    pub message: String,
    pub code: u16,
    #[serde(rename(serialize = "type"))]
    pub error_type: ErrorType,
}
