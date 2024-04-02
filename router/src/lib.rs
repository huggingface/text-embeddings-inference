/// Text Embedding Inference Webserver
mod logging;
mod prometheus;

#[cfg(feature = "http")]
mod http;

#[cfg(feature = "http")]
use ::http::HeaderMap;

#[cfg(feature = "grpc")]
mod grpc;

#[cfg(feature = "grpc")]
use tonic::codegen::http::HeaderMap;

mod shutdown;

use anyhow::{anyhow, Context, Result};
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::time::{Duration, Instant};
use text_embeddings_backend::DType;
use text_embeddings_core::download::{
    download_artifacts, download_pool_config, download_st_config, ST_CONFIG_NAMES,
};
use text_embeddings_core::infer::Infer;
use text_embeddings_core::queue::Queue;
use text_embeddings_core::tokenization::Tokenization;
use text_embeddings_core::TextEmbeddingsError;
use tokenizers::Tokenizer;
use tracing::Span;

pub use logging::init_logging;

/// Create entrypoint
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_id: String,
    revision: Option<String>,
    tokenization_workers: Option<usize>,
    dtype: Option<DType>,
    pooling: Option<text_embeddings_backend::Pool>,
    max_concurrent_requests: usize,
    max_batch_tokens: usize,
    max_batch_requests: Option<usize>,
    max_client_batch_size: usize,
    auto_truncate: bool,
    hf_api_token: Option<String>,
    hostname: Option<String>,
    port: u16,
    uds_path: Option<String>,
    huggingface_hub_cache: Option<String>,
    payload_limit: usize,
    api_key: Option<String>,
    otlp_endpoint: Option<String>,
    cors_allow_origin: Option<Vec<String>>,
) -> Result<()> {
    let model_id_path = Path::new(&model_id);
    let model_root = if model_id_path.exists() && model_id_path.is_dir() {
        // Using a local model
        model_id_path.to_path_buf()
    } else {
        let mut builder = ApiBuilder::new()
            .with_progress(false)
            .with_token(hf_api_token);

        if let Some(cache_dir) = huggingface_hub_cache {
            builder = builder.with_cache_dir(cache_dir.into());
        }

        let api = builder.build().unwrap();
        let api_repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision.clone().unwrap_or("main".to_string()),
        ));

        // Optionally download the pooling config.
        if pooling.is_none() {
            // If a pooling config exist, download it
            let _ = download_pool_config(&api_repo).await;
        }

        // Download sentence transformers config
        let _ = download_st_config(&api_repo).await;

        // Download model from the Hub
        download_artifacts(&api_repo)
            .await
            .context("Could not download model artifacts")?
    };

    // Load config
    let config_path = model_root.join("config.json");
    let config = fs::read_to_string(config_path).context("`config.json` not found")?;
    let config: ModelConfig =
        serde_json::from_str(&config).context("Failed to parse `config.json`")?;

    // Set model type from config
    let backend_model_type = get_backend_model_type(&config, &model_root, pooling)?;

    // Info model type
    let model_type = match &backend_model_type {
        text_embeddings_backend::ModelType::Classifier => {
            let id2label = config
                .id2label
                .context("`config.json` does not contain `id2label`")?;
            let n_classes = id2label.len();
            let classifier_model = ClassifierModel {
                id2label,
                label2id: config
                    .label2id
                    .context("`config.json` does not contain `label2id`")?,
            };
            if n_classes > 1 {
                ModelType::Classifier(classifier_model)
            } else {
                ModelType::Reranker(classifier_model)
            }
        }
        text_embeddings_backend::ModelType::Embedding(pool) => {
            ModelType::Embedding(EmbeddingModel {
                pooling: pool.to_string(),
            })
        }
    };

    // Load tokenizer
    let tokenizer_path = model_root.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).expect(
        "tokenizer.json not found. text-embeddings-inference only supports fast tokenizers",
    );
    tokenizer.with_padding(None);

    // Position IDs offset. Used for Roberta and camembert.
    let position_offset = if &config.model_type == "xlm-roberta"
        || &config.model_type == "camembert"
        || &config.model_type == "roberta"
    {
        config.pad_token_id + 1
    } else {
        0
    };

    // Try to load ST Config
    let mut st_config: Option<STConfig> = None;
    for name in ST_CONFIG_NAMES {
        let config_path = model_root.join(name);
        if let Ok(config) = fs::read_to_string(config_path) {
            st_config =
                Some(serde_json::from_str(&config).context(format!("Failed to parse `{}`", name))?);
            break;
        }
    }
    let max_input_length = match st_config {
        Some(config) => config.max_seq_length,
        None => {
            tracing::warn!("Could not find a Sentence Transformers config");
            config.max_position_embeddings - position_offset
        }
    };
    tracing::info!("Maximum number of tokens per request: {max_input_length}");

    let tokenization_workers = tokenization_workers.unwrap_or_else(num_cpus::get_physical);

    // Tokenization logic
    let tokenization = Tokenization::new(
        tokenization_workers,
        tokenizer,
        max_input_length,
        position_offset,
    );

    // Get dtype
    let dtype = dtype.unwrap_or({
        #[cfg(any(feature = "accelerate", feature = "mkl", feature = "mkl-dynamic"))]
        {
            DType::Float32
        }
        #[cfg(not(any(feature = "accelerate", feature = "mkl", feature = "mkl-dynamic")))]
        {
            DType::Float16
        }
    });

    // Create backend
    tracing::info!("Starting model backend");
    let backend = text_embeddings_backend::Backend::new(
        model_root,
        dtype.clone(),
        backend_model_type,
        uds_path.unwrap_or("/tmp/text-embeddings-inference-server".to_string()),
        otlp_endpoint.clone(),
    )
    .context("Could not create backend")?;
    backend
        .health()
        .await
        .context("Model backend is not healthy")?;

    let max_batch_requests = backend
        .max_batch_size
        .map(|s| {
            tracing::warn!("Backend does not support a batch size > {s}");
            tracing::warn!("forcing `max_batch_requests={s}`");
            s
        })
        .or(max_batch_requests);

    // Queue logic
    let queue = Queue::new(
        backend.padded_model,
        max_batch_tokens,
        max_batch_requests,
        max_concurrent_requests,
    );

    // Create infer task
    let infer = Infer::new(tokenization, queue, max_concurrent_requests, backend);

    // Endpoint info
    let info = Info {
        model_id,
        model_sha: revision,
        model_dtype: dtype.to_string(),
        model_type,
        max_concurrent_requests,
        max_input_length,
        max_batch_tokens,
        tokenization_workers,
        max_batch_requests,
        max_client_batch_size,
        auto_truncate,
        version: env!("CARGO_PKG_VERSION"),
        sha: option_env!("VERGEN_GIT_SHA"),
        docker_label: option_env!("DOCKER_LABEL"),
    };

    // use AIP_HTTP_PORT if google feature is enabled
    let port = if cfg!(feature = "google") {
        std::env::var("AIP_HTTP_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .context("`AIP_HTTP_PORT` env var must be set for Google Vertex deployments")?
    } else {
        port
    };

    let addr = match hostname.unwrap_or("0.0.0.0".to_string()).parse() {
        Ok(ip) => SocketAddr::new(ip, port),
        Err(_) => {
            tracing::warn!("Invalid hostname, defaulting to 0.0.0.0");
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port)
        }
    };

    let prom_builder = prometheus::prometheus_builer(info.max_input_length)?;

    #[cfg(all(feature = "grpc", feature = "http"))]
    compile_error!("Features `http` and `grpc` cannot be enabled at the same time.");

    #[cfg(all(feature = "grpc", feature = "google"))]
    compile_error!("Features `http` and `google` cannot be enabled at the same time.");

    #[cfg(not(any(feature = "http", feature = "grpc")))]
    compile_error!("Either feature `http` or `grpc` must be enabled.");

    #[cfg(feature = "http")]
    {
        let server = tokio::spawn(async move {
            http::server::run(
                infer,
                info,
                addr,
                prom_builder,
                payload_limit,
                api_key,
                cors_allow_origin,
            )
            .await
        });
        tracing::info!("Ready");
        server.await??;
    }

    #[cfg(feature = "grpc")]
    {
        // cors_allow_origin and payload_limit are not used for gRPC servers
        let _ = cors_allow_origin;
        let _ = payload_limit;
        let server = tokio::spawn(async move {
            grpc::server::run(infer, info, addr, prom_builder, api_key).await
        });
        tracing::info!("Ready");
        server.await??;
    }

    Ok(())
}

fn get_backend_model_type(
    config: &ModelConfig,
    model_root: &Path,
    pooling: Option<text_embeddings_backend::Pool>,
) -> Result<text_embeddings_backend::ModelType> {
    for arch in &config.architectures {
        if Some(text_embeddings_backend::Pool::Splade) == pooling && arch.ends_with("MaskedLM") {
            return Ok(text_embeddings_backend::ModelType::Embedding(
                text_embeddings_backend::Pool::Splade,
            ));
        } else if arch.ends_with("Classification") {
            if pooling.is_some() {
                tracing::warn!(
                    "`--pooling` arg is set but model is a classifier. Ignoring `--pooling` arg."
                );
            }
            return Ok(text_embeddings_backend::ModelType::Classifier);
        }
    }

    if Some(text_embeddings_backend::Pool::Splade) == pooling {
        return Err(anyhow!(
            "Splade pooling is not supported: model is not a ForMaskedLM model"
        ));
    }

    // Set pooling
    let pool = match pooling {
        Some(pool) => pool,
        None => {
            // Load pooling config
            let config_path = model_root.join("1_Pooling/config.json");
            let config = fs::read_to_string(config_path).context("The `--pooling` arg is not set and we could not find a pooling configuration (`1_Pooling/config.json`) for this model.")?;
            let config: PoolConfig =
                serde_json::from_str(&config).context("Failed to parse `1_Pooling/config.json`")?;
            if config.pooling_mode_cls_token {
                text_embeddings_backend::Pool::Cls
            } else if config.pooling_mode_mean_tokens {
                text_embeddings_backend::Pool::Mean
            } else {
                return Err(anyhow!("Pooling config {config:?} is not supported"));
            }
        }
    };
    Ok(text_embeddings_backend::ModelType::Embedding(pool))
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub pad_token_id: usize,
    pub id2label: Option<HashMap<String, String>>,
    pub label2id: Option<HashMap<String, usize>>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PoolConfig {
    pooling_mode_cls_token: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
}

#[derive(Debug, Deserialize)]
pub struct STConfig {
    pub max_seq_length: usize,
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
    pub auto_truncate: bool,
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
