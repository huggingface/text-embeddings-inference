use ::http::HeaderMap;
/// Text Embedding Inference Webserver
use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use text_embeddings_core::infer::Infer;
use text_embeddings_core::TextEmbeddingsError;
use tracing::Span;

mod prometheus;

#[cfg(feature = "http")]
mod http;

#[cfg(feature = "grpc")]
mod grpc;
mod shutdown;

/// Crate entrypoint
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

struct ResponseMetadata {
    compute_chars: usize,
    compute_tokens: usize,
    start_time: Instant,
    tokenization_time: Duration,
    queue_time: Duration,
    inference_time: Duration,
}

impl ResponseMetadata {
    fn new(
        compute_chars: usize,
        compute_tokens: usize,
        start_time: Instant,
        tokenization_time: Duration,
        queue_time: Duration,
        inference_time: Duration,
    ) -> Self {
        Self {
            compute_chars,
            compute_tokens,
            start_time,
            tokenization_time,
            queue_time,
            inference_time,
        }
    }

    fn record_span(&self, span: &Span) {
        // Tracing metadata
        span.record("compute_chars", self.compute_chars);
        span.record("compute_tokens", self.compute_tokens);
        span.record("total_time", format!("{:?}", self.start_time.elapsed()));
        span.record("tokenization_time", format!("{:?}", self.tokenization_time));
        span.record("queue_time", format!("{:?}", self.queue_time));
        span.record("inference_time", format!("{:?}", self.inference_time));
    }

    fn record_metrics(&self) {
        // Metrics
        metrics::histogram!(
            "te_request_duration",
            self.start_time.elapsed().as_secs_f64()
        );
        metrics::histogram!(
            "te_request_tokenization_duration",
            self.tokenization_time.as_secs_f64()
        );
        metrics::histogram!("te_request_queue_duration", self.queue_time.as_secs_f64());
        metrics::histogram!(
            "te_request_inference_duration",
            self.inference_time.as_secs_f64()
        );
    }
}

impl From<ResponseMetadata> for HeaderMap {
    fn from(value: ResponseMetadata) -> Self {
        // Headers
        let mut headers = HeaderMap::new();
        headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
        headers.insert(
            "x-compute-time",
            value
                .start_time
                .elapsed()
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers.insert(
            "x-compute-characters",
            value.compute_chars.to_string().parse().unwrap(),
        );
        headers.insert(
            "x-compute-tokens",
            value.compute_tokens.to_string().parse().unwrap(),
        );
        headers.insert(
            "x-total-time",
            value
                .start_time
                .elapsed()
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers.insert(
            "x-tokenization-time",
            value
                .tokenization_time
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers.insert(
            "x-queue-time",
            value.queue_time.as_millis().to_string().parse().unwrap(),
        );
        headers.insert(
            "x-inference-time",
            value
                .inference_time
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers
    }
}
