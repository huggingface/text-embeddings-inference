/// Text Embedding Inference Webserver
mod logging;
mod prometheus;
pub mod strategy;

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
use text_embeddings_backend::{DType, Pool};
use text_embeddings_core::download::{download_artifacts, ST_CONFIG_NAMES};
use text_embeddings_core::infer::Infer;
use text_embeddings_core::queue::Queue;
use text_embeddings_core::tokenization::Tokenization;
use text_embeddings_core::TextEmbeddingsError;
use tokenizers::processors::sequence::Sequence;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{PostProcessorWrapper, Tokenizer};
use tracing::Span;

pub use logging::init_logging;

// Re-export ModelKind from core for public API
pub use text_embeddings_core::detection::ModelKind;

use crate::strategy::{RerankMode, RerankOrdering};
use std::sync::Arc;

/// Listwise reranking configuration
#[derive(Debug, Clone)]
pub struct ListwiseConfig {
    pub max_docs_per_pass: usize,
    pub ordering: RerankOrdering,
    pub instruction: Option<String>,
    pub payload_limit_bytes: usize,
    pub block_timeout_ms: u64,
    pub random_seed: Option<u64>,
    pub max_documents_per_request: usize,
    pub max_document_length_bytes: usize,
}

impl Default for ListwiseConfig {
    fn default() -> Self {
        Self {
            max_docs_per_pass: 125,
            ordering: RerankOrdering::Input,
            instruction: None,
            payload_limit_bytes: 2_000_000,
            block_timeout_ms: 30_000,
            random_seed: None,
            max_documents_per_request: 1_000,
            max_document_length_bytes: 102_400,
        }
    }
}

/// Extended application state
#[derive(Clone)]
pub struct AppState {
    pub infer: Arc<Infer>,
    pub info: Arc<Info>,
    pub model_kind: ModelKind,
    pub reranker_mode: RerankMode,
    pub listwise_config: Arc<ListwiseConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modelkind_equality() {
        let embedding = ModelKind::Embedding;
        let classifier = ModelKind::SequenceClassifier;
        let listwise = ModelKind::ListwiseReranker;

        assert_eq!(embedding, ModelKind::Embedding);
        assert_eq!(classifier, ModelKind::SequenceClassifier);
        assert_eq!(listwise, ModelKind::ListwiseReranker);

        assert_ne!(embedding, classifier);
        assert_ne!(classifier, listwise);
        assert_ne!(listwise, embedding);
    }

    #[test]
    fn test_modelkind_debug_format() {
        let embedding = ModelKind::Embedding;
        let classifier = ModelKind::SequenceClassifier;
        let listwise = ModelKind::ListwiseReranker;

        assert_eq!(format!("{:?}", embedding), "Embedding");
        assert_eq!(format!("{:?}", classifier), "SequenceClassifier");
        assert_eq!(format!("{:?}", listwise), "ListwiseReranker");
    }

    #[test]
    fn test_modelkind_clone() {
        let original = ModelKind::ListwiseReranker;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_listwise_config_default() {
        let config = ListwiseConfig::default();

        assert_eq!(config.max_docs_per_pass, 125);
        assert_eq!(config.ordering, RerankOrdering::Input);
        assert_eq!(config.instruction, None);
        assert_eq!(config.payload_limit_bytes, 2_000_000);
        assert_eq!(config.block_timeout_ms, 30_000);
        assert_eq!(config.random_seed, None);
        assert_eq!(config.max_documents_per_request, 1_000);
        assert_eq!(config.max_document_length_bytes, 102_400);
    }

    #[test]
    fn test_listwise_config_custom() {
        let config = ListwiseConfig {
            max_docs_per_pass: 100,
            ordering: RerankOrdering::Random,
            instruction: Some("Focus on relevance".to_string()),
            payload_limit_bytes: 5_000_000,
            block_timeout_ms: 60_000,
            random_seed: Some(42),
            max_documents_per_request: 2_000,
            max_document_length_bytes: 204_800,
        };

        assert_eq!(config.max_docs_per_pass, 100);
        assert_eq!(config.ordering, RerankOrdering::Random);
        assert_eq!(config.instruction, Some("Focus on relevance".to_string()));
        assert_eq!(config.payload_limit_bytes, 5_000_000);
        assert_eq!(config.block_timeout_ms, 60_000);
        assert_eq!(config.random_seed, Some(42));
        assert_eq!(config.max_documents_per_request, 2_000);
        assert_eq!(config.max_document_length_bytes, 204_800);
    }

    #[test]
    fn test_listwise_config_clone() {
        let original = ListwiseConfig {
            max_docs_per_pass: 200,
            ordering: RerankOrdering::Random,
            instruction: Some("Test instruction".to_string()),
            payload_limit_bytes: 3_000_000,
            block_timeout_ms: 45_000,
            random_seed: Some(123),
            max_documents_per_request: 500,
            max_document_length_bytes: 50_000,
        };

        let cloned = original.clone();
        assert_eq!(original.max_docs_per_pass, cloned.max_docs_per_pass);
        assert_eq!(original.ordering, cloned.ordering);
        assert_eq!(original.instruction, cloned.instruction);
        assert_eq!(original.payload_limit_bytes, cloned.payload_limit_bytes);
        assert_eq!(original.block_timeout_ms, cloned.block_timeout_ms);
        assert_eq!(original.random_seed, cloned.random_seed);
        assert_eq!(
            original.max_documents_per_request,
            cloned.max_documents_per_request
        );
        assert_eq!(
            original.max_document_length_bytes,
            cloned.max_document_length_bytes
        );
    }

    #[test]
    fn test_listwise_config_debug_format() {
        let config = ListwiseConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("ListwiseConfig"));
        assert!(debug_str.contains("max_docs_per_pass: 125"));
        assert!(debug_str.contains("ordering: Input"));
    }
}

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
    default_prompt: Option<String>,
    default_prompt_name: Option<String>,
    dense_path: Option<String>,
    hf_token: Option<String>,
    hostname: Option<String>,
    port: u16,
    uds_path: Option<String>,
    huggingface_hub_cache: Option<String>,
    payload_limit: usize,
    api_key: Option<String>,
    otlp_endpoint: Option<String>,
    otlp_service_name: String,
    prometheus_port: u16,
    cors_allow_origin: Option<Vec<String>>,
    // Listwise reranking parameters
    reranker_mode: strategy::RerankMode,
    max_listwise_docs_per_pass: usize,
    rerank_ordering: strategy::RerankOrdering,
    rerank_instruction: Option<String>,
    listwise_payload_limit_bytes: usize,
    listwise_block_timeout_ms: u64,
    max_documents_per_request: usize,
    max_document_length_bytes: usize,
    rerank_rand_seed: Option<u64>,
) -> Result<()> {
    let model_id_path = Path::new(&model_id);
    let (model_root, api_repo) = if model_id_path.exists() && model_id_path.is_dir() {
        // Using a local model
        (model_id_path.to_path_buf(), None)
    } else {
        let mut builder = ApiBuilder::from_env()
            .with_progress(false)
            .with_token(hf_token);

        if let Some(cache_dir) = huggingface_hub_cache {
            builder = builder.with_cache_dir(cache_dir.into());
        }

        if let Ok(origin) = std::env::var("HF_HUB_USER_AGENT_ORIGIN") {
            builder = builder.with_user_agent("origin", origin.as_str());
        }

        let api = builder.build().unwrap();
        let api_repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision.clone().unwrap_or("main".to_string()),
        ));

        // Download model from the Hub
        (
            download_artifacts(&api_repo, pooling.is_none(), dense_path.clone())
                .await
                .context("Could not download model artifacts")?,
            Some(api_repo),
        )
    };

    // Build path to Dense module, if applicable, otherwise None
    let dense_root = dense_path.map(|path| model_root.join(path));

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
    // Qwen2 updates the post processor manually instead of into the tokenizer.json...
    // https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct/blob/main/tokenization_qwen.py#L246
    if config.model_type == "qwen2" {
        let template = TemplateProcessing::builder()
            .try_single("$A:0 <|endoftext|>:0")
            .unwrap()
            .try_pair("$A:0 <|endoftext|>:0 $B:1 <|endoftext|>:1")
            .unwrap()
            .special_tokens(vec![("<|endoftext|>", 151643)])
            .build()
            .unwrap();
        match tokenizer.get_post_processor() {
            None => tokenizer.with_post_processor(Some(template)),
            Some(post_processor) => {
                let post_processor = Sequence::new(vec![
                    post_processor.clone(),
                    PostProcessorWrapper::Template(template),
                ]);
                tokenizer.with_post_processor(Some(post_processor))
            }
        };
    }

    // Detect model kind for listwise reranking support
    let model_kind = text_embeddings_core::detection::detect_model_kind(&tokenizer, &model_root)
        .context("Failed to detect model type")?;
    tracing::info!("✅ Detected model kind: {:?}", model_kind);

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

    let tokenization_workers = tokenization_workers.unwrap_or_else(num_cpus::get);

    // Try to load new ST Config
    let mut new_st_config: Option<NewSTConfig> = None;
    let config_path = model_root.join("config_sentence_transformers.json");
    if let Ok(config) = fs::read_to_string(config_path) {
        new_st_config = Some(
            serde_json::from_str(&config)
                .context("Failed to parse `config_sentence_transformers.json`")?,
        );
    }
    let prompts = new_st_config.and_then(|c| c.prompts);
    let default_prompt = if let Some(default_prompt_name) = default_prompt_name.as_ref() {
        match &prompts {
            None => {
                anyhow::bail!(format!("`default-prompt-name` is set to `{default_prompt_name}` but no prompts were found in the Sentence Transformers configuration"));
            }
            Some(prompts) if !prompts.contains_key(default_prompt_name) => {
                anyhow::bail!(format!("`default-prompt-name` is set to `{default_prompt_name}` but it was not found in the Sentence Transformers prompts. Available prompts: {:?}", prompts.keys()));
            }
            Some(prompts) => prompts.get(default_prompt_name).cloned(),
        }
    } else {
        default_prompt
    };

    // Tokenization logic
    let tokenization = Tokenization::new(
        tokenization_workers,
        tokenizer,
        max_input_length,
        position_offset,
        default_prompt,
        prompts,
    );

    // Get dtype
    let dtype = dtype.unwrap_or_default();

    // Create backend
    tracing::info!("Starting model backend");
    let backend = text_embeddings_backend::Backend::new(
        model_root,
        api_repo,
        dtype.clone(),
        backend_model_type,
        dense_root,
        uds_path.unwrap_or("/tmp/text-embeddings-inference-server".to_string()),
        otlp_endpoint.clone(),
        otlp_service_name.clone(),
    )
    .await
    .context("Could not create backend")?;
    backend
        .health()
        .await
        .context("Model backend is not healthy")?;

    tracing::info!("Warming up model");
    backend
        .warmup(max_input_length, max_batch_tokens, max_batch_requests)
        .await
        .context("Model backend is not healthy")?;

    let max_batch_requests = backend
        .max_batch_size
        .inspect(|&s| {
            tracing::warn!("Backend does not support a batch size > {s}");
            tracing::warn!("forcing `max_batch_requests={s}`");
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

    // Create ListwiseConfig from CLI parameters
    let listwise_config = ListwiseConfig {
        max_docs_per_pass: max_listwise_docs_per_pass,
        ordering: rerank_ordering,
        instruction: rerank_instruction,
        payload_limit_bytes: listwise_payload_limit_bytes,
        block_timeout_ms: listwise_block_timeout_ms,
        random_seed: rerank_rand_seed,
        max_documents_per_request,
        max_document_length_bytes,
    };
    tracing::info!("✅ Listwise config: {:?}", listwise_config);

    // use AIP_HTTP_PORT if google feature is enabled
    let port = if cfg!(feature = "google") {
        std::env::var("AIP_HTTP_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .inspect(|&p| {
                tracing::info!("`AIP_HTTP_PORT` is set: overriding port {port} by port {p}");
            })
            .unwrap_or(port)
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

    let prom_builder = prometheus::prometheus_builer(addr, prometheus_port, info.max_input_length)?;

    #[cfg(all(feature = "grpc", feature = "http"))]
    compile_error!("Features `http` and `grpc` cannot be enabled at the same time.");

    #[cfg(all(feature = "grpc", feature = "google"))]
    compile_error!("Features `http` and `google` cannot be enabled at the same time.");

    #[cfg(not(any(feature = "http", feature = "grpc")))]
    compile_error!("Either feature `http` or `grpc` must be enabled.");

    // Note: model_kind, reranker_mode, and listwise_config are detected and logged above
    // These will be wired into HTTP/gRPC handlers in future milestones (Milestone 2+)
    // For now, we verify detection works correctly
    let _ = model_kind;
    let _ = reranker_mode;
    let _ = listwise_config;

    #[cfg(feature = "http")]
    {
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
    }

    #[cfg(feature = "grpc")]
    {
        // cors_allow_origin and payload_limit are not used for gRPC servers
        let _ = cors_allow_origin;
        let _ = payload_limit;
        grpc::server::run(infer, info, addr, prom_builder, api_key).await
    }
}

fn get_backend_model_type(
    config: &ModelConfig,
    model_root: &Path,
    pooling: Option<text_embeddings_backend::Pool>,
) -> Result<text_embeddings_backend::ModelType> {
    for arch in &config.architectures {
        // Edge case affecting `Alibaba-NLP/gte-multilingual-base` and possibly other fine-tunes of
        // the same base model. More context at https://huggingface.co/Alibaba-NLP/gte-multilingual-base/discussions/7
        if arch == "NewForTokenClassification"
            && (config.id2label.is_none() | config.label2id.is_none())
        {
            tracing::warn!("Provided `--model-id` is likely an AlibabaNLP GTE model, but the `config.json` contains the architecture `NewForTokenClassification` but it doesn't contain the `id2label` and `label2id` mapping, so `NewForTokenClassification` architecture will be ignored.");
            continue;
        }

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

            match fs::read_to_string(config_path) {
                Ok(config) => {
                    let config: PoolConfig = serde_json::from_str(&config)
                        .context("Failed to parse `1_Pooling/config.json`")?;
                    Pool::try_from(config)?
                }
                Err(err) => {
                    if !config.model_type.to_lowercase().contains("bert") {
                        return Err(err).context("The `--pooling` arg is not set and we could not find a pooling configuration (`1_Pooling/config.json`) for this model.");
                    }
                    tracing::warn!("The `--pooling` arg is not set and we could not find a pooling configuration (`1_Pooling/config.json`) for this model but the model is a BERT variant. Defaulting to `CLS` pooling.");
                    text_embeddings_backend::Pool::Cls
                }
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
    #[serde(default)]
    pooling_mode_lasttoken: bool,
}

impl TryFrom<PoolConfig> for Pool {
    type Error = anyhow::Error;

    fn try_from(config: PoolConfig) -> std::result::Result<Self, Self::Error> {
        if config.pooling_mode_cls_token {
            return Ok(Pool::Cls);
        }
        if config.pooling_mode_mean_tokens {
            return Ok(Pool::Mean);
        }
        if config.pooling_mode_lasttoken {
            return Ok(Pool::LastToken);
        }
        Err(anyhow!("Pooling config {config:?} is not supported"))
    }
}

#[derive(Debug, Deserialize)]
pub struct STConfig {
    pub max_seq_length: usize,
}

#[derive(Debug, Deserialize)]
pub struct NewSTConfig {
    pub prompts: Option<HashMap<String, String>>,
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
    Empty,
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
        let histogram = metrics::histogram!("te_request_duration");
        histogram.record(self.start_time.elapsed().as_secs_f64());
        let histogram = metrics::histogram!("te_request_tokenization_duration");
        histogram.record(self.tokenization_time.as_secs_f64());
        let histogram = metrics::histogram!("te_request_queue_duration");
        histogram.record(self.queue_time.as_secs_f64());
        let histogram = metrics::histogram!("te_request_inference_duration");
        histogram.record(self.inference_time.as_secs_f64());
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
