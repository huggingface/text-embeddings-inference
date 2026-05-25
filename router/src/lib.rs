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
use hf_hub::api::tokio::{ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::time::{Duration, Instant};
use text_embeddings_backend::{BackendOutput, DType, ModelType as BackendModelType, Pool};
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

/// Create entrypoint
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_id: String,
    revision: Option<String>,
    tokenization_workers: Option<usize>,
    dtype: Option<DType>,
    served_model_name: String,
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
) -> Result<()> {
    let model_id_path = Path::new(&model_id);
    let (model_root, api_repo) = if model_id_path.exists() && model_id_path.is_dir() {
        // Using a local model
        (model_id_path.to_path_buf(), None)
    } else {
        let mut builder = ApiBuilder::from_env().with_progress(false);

        if let Some(cache_dir) = huggingface_hub_cache {
            builder = builder.with_cache_dir(cache_dir.into());
        }

        // NOTE: Only set the `token` if it's not None, otherwise leave it as default so that the
        // token from the cache location is pulled instead, if exists
        if hf_token.is_some() {
            builder = builder.with_token(hf_token);
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
            download_artifacts(&api_repo, pooling.is_none())
                .await
                .context("Could not download model artifacts")?,
            Some(api_repo),
        )
    };

    if let Some(api_repo) = api_repo.as_ref() {
        download_modular_reranker_detection_files(api_repo).await?;
    }

    // Load config
    let config_path = model_root.join("config.json");
    let config = fs::read_to_string(config_path).context("`config.json` not found")?;
    let config: ModelConfig =
        serde_json::from_str(&config).context("Failed to parse `config.json`")?;

    // Set model type from config
    let (backend_model_type, backend_output) =
        get_backend_model_type(&config, &model_root, pooling)?;

    // Info model type
    let model_type = match (&backend_model_type, backend_output) {
        (text_embeddings_backend::ModelType::Classifier, _) => classifier_model_type(&config)?,
        (
            text_embeddings_backend::ModelType::Embedding(_),
            text_embeddings_backend::BackendOutput::Predict,
        ) => reranker_model_type(&config),
        (
            text_embeddings_backend::ModelType::Embedding(pool),
            text_embeddings_backend::BackendOutput::Embed,
        ) => ModelType::Embedding(EmbeddingModel {
            pooling: pool.to_string(),
        }),
    };

    // Load tokenizer
    let tokenizer_path = model_root.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).expect(
        "tokenizer.json not found. text-embeddings-inference only supports fast tokenizers",
    );
    tokenizer.with_padding(None);
    // Old Qwen2  repos updates the post processor manually instead of into the tokenizer.json.
    // Newer ones (https://huggingface.co/jinaai/jina-code-embeddings-0.5b/tree/main) have it in the tokenizer.json. This is to support both cases.
    // https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct/blob/main/tokenization_qwen.py#L246
    if config.model_type == "qwen2"
        && config
            .auto_map
            .as_ref()
            .is_some_and(|m| m.get("AutoModel") == Some(&"modeling_qwen.Qwen2Model".to_string()))
    {
        tracing::warn!("Model is detected as a Qwen2 model with remote code. Adding a post processor manually as the tokenizer.json does not contain a post processor.");
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

    // Position IDs offset. Used for Roberta and camembert.
    let position_offset = if &config.model_type == "xlm-roberta"
        || &config.model_type == "camembert"
        || &config.model_type == "roberta"
    {
        config.pad_token_id.unwrap_or(0) + 1
    } else {
        0
    };

    // Try to load ST Config
    let mut st_config: Option<STConfig> = None;
    for name in ST_CONFIG_NAMES {
        let config_path = model_root.join(name);
        if let Ok(config) = fs::read_to_string(config_path) {
            let config: STConfig =
                serde_json::from_str(&config).context(format!("Failed to parse `{}`", name))?;
            if config.max_seq_length.is_some() {
                st_config = Some(config);
                break;
            } else {
                tracing::warn!(
                    "`{}` does not define `max_seq_length`; falling back to `config.json` maximum position embeddings.",
                    name
                );
            }
        }
    }

    let base_input_length = match st_config {
        Some(config) => config.max_seq_length.expect("checked above"),
        None => {
            tracing::warn!("Could not find a Sentence Transformers config");
            config.max_position_embeddings - position_offset
        }
    };

    let max_input_length = if base_input_length > max_batch_tokens {
        if !auto_truncate {
            anyhow::bail!(
                "The maximum input length is `{base_input_length}` which exceeds `--max-batch-tokens={max_batch_tokens}`. Either increase `--max-batch-tokens` to at least `{base_input_length}`, or set `--auto-truncate true` so that regardless the maximum input length, those are truncated to `{max_batch_tokens}` tokens."
            );
        }
        tracing::warn!(
            "The maximum input length is `{base_input_length}` which exceeds `--max-batch-tokens={max_batch_tokens}`. Input sequences will be truncated to `{max_batch_tokens}` tokens, as `--auto-truncate` is either not provided (defaults to true) or provided as true. To avoid truncation, increase `--max-batch-tokens` to at least `{base_input_length}` and set `--auto-truncate false`."
        );
        max_batch_tokens
    } else {
        base_input_length
    };
    tracing::info!("Maximum number of tokens per request: {max_input_length}");

    // fall-back to num_cpus - 1 to leave some CPU for the backend, and at most 64 workers.
    let tokenization_workers =
        tokenization_workers.unwrap_or_else(|| (num_cpus::get() - 1).clamp(1, 64));

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

    // NOTE: `gemma3_text` won't support Float16 but only Float32, given that with `candle-cuda`
    // feature, the default `Dtype::Float16` this overrides that to prevent issues when running a
    // `gemma3_text` model without specifying a `--dtype`
    let dtype = if dtype.is_none() && config.model_type == "gemma3_text" {
        DType::Float32
    } else {
        dtype.unwrap_or_default()
    };

    // Create backend
    tracing::info!("Starting model backend");
    let backend = text_embeddings_backend::Backend::new_with_output(
        model_root,
        api_repo,
        dtype.clone(),
        backend_model_type,
        backend_output,
        dense_path,
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
        .warmup(
            max_input_length,
            max_batch_tokens,
            max_batch_requests,
            backend.padded_model,
        )
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
        served_model_name,
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
) -> Result<(BackendModelType, BackendOutput)> {
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
            return Ok((
                BackendModelType::Embedding(text_embeddings_backend::Pool::Splade),
                BackendOutput::Embed,
            ));
        } else if arch.ends_with("Classification") {
            if pooling.is_some() {
                tracing::warn!(
                    "`--pooling` arg is set but model is a classifier. Ignoring `--pooling` arg."
                );
            }
            return Ok((BackendModelType::Classifier, BackendOutput::Predict));
        }
    }

    if Some(text_embeddings_backend::Pool::Splade) == pooling {
        return Err(anyhow!(
            "Splade pooling is not supported: model is not a ForMaskedLM model"
        ));
    }

    if let Some(pool) = detect_modular_reranker(model_root)? {
        if pooling.is_some() {
            tracing::warn!(
                "`--pooling` arg is set but model is a modular reranker. Using the pooling module config instead."
            );
        }
        return Ok((BackendModelType::Embedding(pool), BackendOutput::Predict));
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

    Ok((BackendModelType::Embedding(pool), BackendOutput::Embed))
}

fn detect_modular_reranker(model_root: &Path) -> Result<Option<Pool>> {
    let modules_path = model_root.join("modules.json");
    let modules = match fs::read_to_string(&modules_path) {
        Ok(content) => content,
        Err(_) => return Ok(None),
    };
    let modules: Vec<ModuleConfig> = match serde_json::from_str(&modules) {
        Ok(modules) => modules,
        Err(err) => {
            tracing::warn!("Failed to parse `modules.json` for modular reranker detection: {err}");
            return Ok(None);
        }
    };

    if modules
        .first()
        .is_none_or(|module| !module.is_transformer())
    {
        return Ok(None);
    }

    let Some(pooling_index) = modules.iter().position(|module| module.is_pooling()) else {
        return Ok(None);
    };
    let pooling_module = &modules[pooling_index];

    let Some(final_module) = modules.last().filter(|module| module.is_dense()) else {
        return Ok(None);
    };

    // The final dense config is what distinguishes a reranker (out_features == 1,
    // output "scores") from an embedding model that merely ends in a Dense module,
    // so until it is read the model is not yet confirmed to be a reranker and a
    // missing/unreadable config falls back to the embedding path rather than erroring.
    let dense_config_path = model_root.join(&final_module.path).join("config.json");
    let dense_config = match fs::read_to_string(&dense_config_path) {
        Ok(content) => content,
        Err(_) => return Ok(None),
    };
    let dense_config: DenseDetectionConfig = match serde_json::from_str(&dense_config) {
        Ok(config) => config,
        Err(err) => {
            tracing::warn!(
                "Failed to parse `{}` for modular reranker detection: {err}",
                dense_config_path.display()
            );
            return Ok(None);
        }
    };

    if dense_config.out_features != 1
        || dense_config.module_output_name.as_deref() != Some("scores")
    {
        return Ok(None);
    }

    for module in &modules[pooling_index + 1..] {
        if !module.is_supported_prediction_layer() {
            return Err(anyhow!(
                "Unsupported module in modular reranker head: `{}` at `{}`",
                module.module_type,
                module.path
            ));
        }
    }

    // At this point the model is confirmed to be a modular reranker, so a missing
    // pooling config is a broken model rather than a signal to treat it as an
    // embedding model: fail loudly instead of falling back.
    let pooling_config_path = model_root.join(&pooling_module.path).join("config.json");
    let pooling_config = fs::read_to_string(&pooling_config_path).with_context(|| {
        format!(
            "Failed to read modular reranker pooling config `{}`",
            pooling_config_path.display()
        )
    })?;
    let pooling_config: PoolConfig = serde_json::from_str(&pooling_config)
        .context("Failed to parse modular reranker pooling config")?;

    Ok(Some(Pool::try_from(pooling_config)?))
}

/// Pre-downloads the config files [`detect_modular_reranker`] inspects locally.
///
/// A transformer -> pooling -> ... -> dense pipeline is either a modular reranker
/// or an embedding model whose last module is a Dense projection; the pooling and
/// final dense configs are what tell the two apart. Downloading them is therefore
/// required, not best-effort: skipping a reranker's configs would silently route
/// it as an embedding model and disable `/rerank`, and the final dense config is
/// needed to load the embedding Dense module anyway.
async fn download_modular_reranker_detection_files(api: &ApiRepo) -> Result<()> {
    let Ok(modules_path) = api.get("modules.json").await else {
        return Ok(());
    };
    let Ok(modules) = fs::read_to_string(modules_path) else {
        return Ok(());
    };
    let Ok(modules) = serde_json::from_str::<Vec<ModuleConfig>>(&modules) else {
        return Ok(());
    };

    // Only a transformer -> pooling -> ... -> dense pipeline needs this extra
    // classification; anything else is left to the regular embedding path.
    if modules
        .first()
        .is_none_or(|module| !module.is_transformer())
    {
        return Ok(());
    }
    let Some(pooling_module) = modules.iter().find(|module| module.is_pooling()) else {
        return Ok(());
    };
    let Some(final_module) = modules.last().filter(|module| module.is_dense()) else {
        return Ok(());
    };

    let pooling_config = format!("{}/config.json", pooling_module.path);
    api.get(&pooling_config).await.map_err(|err| {
        anyhow!("Could not download `{pooling_config}` needed to classify the Sentence Transformers pipeline: {err}")
    })?;
    let dense_config = format!("{}/config.json", final_module.path);
    api.get(&dense_config).await.map_err(|err| {
        anyhow!("Could not download `{dense_config}` needed to classify the Sentence Transformers pipeline: {err}")
    })?;

    Ok(())
}

#[derive(Debug, Deserialize)]
struct ModuleConfig {
    path: String,
    #[serde(rename = "type")]
    module_type: String,
}

impl ModuleConfig {
    fn is_transformer(&self) -> bool {
        matches!(
            self.module_type.as_str(),
            "sentence_transformers.models.Transformer"
                | "sentence_transformers.base.modules.transformer.Transformer"
        )
    }

    fn is_pooling(&self) -> bool {
        matches!(
            self.module_type.as_str(),
            "sentence_transformers.models.Pooling"
                | "sentence_transformers.sentence_transformer.modules.pooling.Pooling"
        )
    }

    fn is_dense(&self) -> bool {
        matches!(
            self.module_type.as_str(),
            "sentence_transformers.models.Dense" | "sentence_transformers.base.modules.dense.Dense"
        )
    }

    fn is_layer_norm(&self) -> bool {
        matches!(
            self.module_type.as_str(),
            "sentence_transformers.models.LayerNorm"
                | "sentence_transformers.sentence_transformer.modules.layer_norm.LayerNorm"
        )
    }

    fn is_supported_prediction_layer(&self) -> bool {
        self.is_dense() || self.is_layer_norm()
    }
}

#[derive(Debug, Deserialize)]
struct DenseDetectionConfig {
    out_features: usize,
    module_output_name: Option<String>,
}

fn classifier_model_type(config: &ModelConfig) -> Result<ModelType> {
    let id2label = config
        .id2label
        .clone()
        .context("`config.json` does not contain `id2label`")?;
    let n_classes = id2label.len();
    let classifier_model = ClassifierModel {
        id2label,
        label2id: config
            .label2id
            .clone()
            .context("`config.json` does not contain `label2id`")?,
    };
    if n_classes > 1 {
        Ok(ModelType::Classifier(classifier_model))
    } else {
        Ok(ModelType::Reranker(classifier_model))
    }
}

fn reranker_model_type(config: &ModelConfig) -> ModelType {
    ModelType::Reranker(ClassifierModel {
        id2label: config.id2label.clone().unwrap_or_else(default_id2label),
        label2id: config.label2id.clone().unwrap_or_else(default_label2id),
    })
}

fn default_id2label() -> HashMap<String, String> {
    HashMap::from([("0".to_string(), "LABEL_0".to_string())])
}

fn default_label2id() -> HashMap<String, usize> {
    HashMap::from([("LABEL_0".to_string(), 0)])
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    pub id2label: Option<HashMap<String, String>>,
    pub label2id: Option<HashMap<String, usize>>,
    pub auto_map: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PoolConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
    #[serde(default)]
    pooling_mode: Option<String>,
}

impl TryFrom<PoolConfig> for Pool {
    type Error = anyhow::Error;

    fn try_from(config: PoolConfig) -> std::result::Result<Self, Self::Error> {
        match config.pooling_mode.as_deref() {
            Some("cls") => return Ok(Pool::Cls),
            Some("mean") => return Ok(Pool::Mean),
            Some("lasttoken" | "last_token") => return Ok(Pool::LastToken),
            Some(_) | None => {}
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn test_model_config() -> ModelConfig {
        test_model_config_with_architecture("ModernBertModel", "modernbert")
    }

    fn test_model_config_without_labels() -> ModelConfig {
        let mut config = test_model_config();
        config.id2label = None;
        config.label2id = None;
        config
    }

    fn test_model_config_with_architecture(architecture: &str, model_type: &str) -> ModelConfig {
        ModelConfig {
            architectures: vec![architecture.to_string()],
            model_type: model_type.to_string(),
            max_position_embeddings: 8192,
            pad_token_id: Some(50283),
            id2label: Some(HashMap::from([("0".to_string(), "LABEL_0".to_string())])),
            label2id: Some(HashMap::from([("LABEL_0".to_string(), 0)])),
            auto_map: None,
        }
    }

    fn temp_model_dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "tei-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_file(path: PathBuf, contents: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
    }

    fn write_pooling_config(model_root: &Path) {
        write_file(
            model_root.join("1_Pooling/config.json"),
            r#"{
                "pooling_mode_cls_token": true,
                "pooling_mode_mean_tokens": false,
                "pooling_mode_max_tokens": false
            }"#,
        );
    }

    fn write_pooling_mode_config(model_root: &Path) {
        write_file(
            model_root.join("1_Pooling/config.json"),
            r#"{
                "pooling_mode": "cls"
            }"#,
        );
    }

    #[test]
    fn detects_modular_sentence_transformers_reranker() {
        let model_root = temp_model_dir("modular-reranker");
        write_pooling_mode_config(&model_root);
        write_file(
            model_root.join("modules.json"),
            r#"[
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"},
                {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.base.modules.dense.Dense"},
                {"idx": 3, "name": "3", "path": "3_LayerNorm", "type": "sentence_transformers.sentence_transformer.modules.layer_norm.LayerNorm"},
                {"idx": 4, "name": "4", "path": "4_Dense", "type": "sentence_transformers.base.modules.dense.Dense"}
            ]"#,
        );
        write_file(
            model_root.join("4_Dense/config.json"),
            r#"{
                "in_features": 384,
                "out_features": 1,
                "bias": true,
                "activation_function": "torch.nn.modules.linear.Identity",
                "module_input_name": "sentence_embedding",
                "module_output_name": "scores"
            }"#,
        );

        let model_type = get_backend_model_type(&test_model_config(), &model_root, None).unwrap();

        assert_eq!(
            model_type,
            (
                text_embeddings_backend::ModelType::Embedding(Pool::Cls),
                BackendOutput::Predict
            )
        );
    }

    #[test]
    fn detects_legacy_sentence_transformers_reranker_for_other_backbone() {
        let model_root = temp_model_dir("legacy-reranker");
        write_pooling_config(&model_root);
        write_file(
            model_root.join("modules.json"),
            r#"[
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
                {"idx": 3, "name": "3", "path": "3_LayerNorm", "type": "sentence_transformers.models.LayerNorm"},
                {"idx": 4, "name": "4", "path": "4_Dense", "type": "sentence_transformers.models.Dense"}
            ]"#,
        );
        write_file(
            model_root.join("4_Dense/config.json"),
            r#"{
                "in_features": 768,
                "out_features": 1,
                "bias": true,
                "activation_function": "torch.nn.modules.linear.Identity",
                "module_input_name": "sentence_embedding",
                "module_output_name": "scores"
            }"#,
        );

        let config = test_model_config_with_architecture("BertModel", "bert");
        let model_type = get_backend_model_type(&config, &model_root, None).unwrap();

        assert_eq!(
            model_type,
            (
                text_embeddings_backend::ModelType::Embedding(Pool::Cls),
                BackendOutput::Predict
            )
        );
    }

    #[test]
    fn keeps_sentence_transformers_embedding_dense_as_embedding() {
        let model_root = temp_model_dir("sentence-transformers-embedding");
        write_pooling_config(&model_root);
        write_file(
            model_root.join("modules.json"),
            r#"[
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"}
            ]"#,
        );
        write_file(
            model_root.join("2_Dense/config.json"),
            r#"{
                "in_features": 1024,
                "out_features": 1024,
                "bias": true,
                "activation_function": "torch.nn.modules.linear.Identity",
                "module_output_name": "sentence_embedding"
            }"#,
        );

        let model_type = get_backend_model_type(&test_model_config(), &model_root, None).unwrap();

        assert_eq!(
            model_type,
            (
                text_embeddings_backend::ModelType::Embedding(Pool::Cls),
                BackendOutput::Embed
            )
        );
    }

    #[test]
    fn rejects_modular_reranker_with_unsupported_head_module() {
        let model_root = temp_model_dir("unsupported-reranker");
        write_pooling_config(&model_root);
        write_file(
            model_root.join("modules.json"),
            r#"[
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
                {"idx": 3, "name": "3", "path": "3_Normalize", "type": "sentence_transformers.models.Normalize"},
                {"idx": 4, "name": "4", "path": "4_Dense", "type": "sentence_transformers.models.Dense"}
            ]"#,
        );
        write_file(
            model_root.join("4_Dense/config.json"),
            r#"{
                "in_features": 768,
                "out_features": 1,
                "bias": true,
                "activation_function": "torch.nn.modules.linear.Identity",
                "module_input_name": "sentence_embedding",
                "module_output_name": "scores"
            }"#,
        );

        let err = get_backend_model_type(&test_model_config(), &model_root, None).unwrap_err();

        assert!(err
            .to_string()
            .contains("Unsupported module in modular reranker head"));
    }

    #[test]
    fn modular_reranker_uses_default_label_map_when_config_omits_labels() {
        let model_type = reranker_model_type(&test_model_config_without_labels());

        let ModelType::Reranker(classifier) = model_type else {
            panic!("expected reranker model type");
        };
        assert_eq!(classifier.id2label.get("0").unwrap(), "LABEL_0");
        assert_eq!(classifier.label2id.get("LABEL_0").copied(), Some(0));
    }
}

#[derive(Debug, Deserialize)]
pub struct STConfig {
    pub max_seq_length: Option<usize>,
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
    #[cfg_attr(feature = "http", schema(example = "thenlper/gte-base"))]
    pub served_model_name: String,
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
            TextEmbeddingsError::Empty(_) => ErrorType::Empty,
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
