mod alibi;
#[cfg(feature = "cuda")]
mod compute_cap;
#[cfg(feature = "cuda")]
mod flash_attn;
mod layers;
mod models;

use anyhow::Context;
use candle::{DType, Device};
use candle_nn::VarBuilder;
use nohash_hasher::BuildNoHashHasher;
use serde::{de::Deserializer, Deserialize};
use std::collections::HashMap;
use std::path::Path;
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Predictions,
};

#[cfg(feature = "cuda")]
use crate::compute_cap::{
    compatible_compute_cap, get_compile_compute_cap, get_runtime_compute_cap,
};
use crate::models::{
    BertConfig, BertModel, DebertaV2Config, DebertaV2Model, Dense, DenseConfig, DenseLayer,
    DistilBertConfig, DistilBertModel, GTEConfig, GTEModel, Gemma3Config, Gemma3Model,
    JinaBertModel, JinaCodeBertModel, MPNetConfig, MPNetModel, MistralConfig, Model,
    ModernBertConfig, ModernBertModel, NomicBertModel, NomicConfig, Qwen2Config, Qwen3Config,
    Qwen3Model, StaticEmbeddingConfig, StaticEmbeddingModel,
};
#[cfg(feature = "cuda")]
use crate::models::{
    FlashBertModel, FlashDistilBertModel, FlashGTEModel, FlashJinaBertModel,
    FlashJinaCodeBertModel, FlashMistralModel, FlashModernBertModel, FlashNomicBertModel,
    FlashQwen2Model, FlashQwen3Model,
};

/// This enum is needed to be able to differentiate between jina models that also use
/// the `bert` model type and valid Bert models.
#[derive(Debug, Clone, PartialEq)]
pub enum BertConfigWrapper {
    JinaBert(BertConfig),
    JinaCodeBert(BertConfig),
    Bert(BertConfig),
}

/// Custom deserializer is required as we need to capture both whether the `_name_or_path` value
/// is any of the JinaBERT alternatives, or alternatively to also support fine-tunes and re-uploads
/// with Sentence Transformers, we also need to check the value for the `auto_map.AutoConfig`
/// configuration file, and see if that points to the relevant remote code repositories on the Hub
impl<'de> Deserialize<'de> for BertConfigWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        #[allow(unused_mut)]
        let mut value = serde_json::Value::deserialize(deserializer)?;

        let name_or_path = value
            .get("_name_or_path")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .unwrap_or_default();

        let auto_config = value
            .get("auto_map")
            .and_then(|v| v.get("AutoConfig"))
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .unwrap_or_default();

        let config = BertConfig::deserialize(value).map_err(Error::custom)?;

        if name_or_path == "jinaai/jina-bert-implementation"
            || auto_config.contains("jinaai/jina-bert-implementation")
        {
            // https://huggingface.co/jinaai/jina-bert-implementation
            Ok(Self::JinaBert(config))
        } else if name_or_path == "jinaai/jina-bert-v2-qk-post-norm"
            || auto_config.contains("jinaai/jina-bert-v2-qk-post-norm")
        {
            // https://huggingface.co/jinaai/jina-bert-v2-qk-post-norm
            Ok(Self::JinaCodeBert(config))
        } else {
            Ok(Self::Bert(config))
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "model_type", rename_all = "kebab-case")]
enum Config {
    Bert(BertConfigWrapper),
    Camembert(BertConfig),
    #[serde(rename(deserialize = "deberta-v2"))]
    DebertaV2(DebertaV2Config),
    #[serde(rename(deserialize = "distilbert"))]
    DistilBert(DistilBertConfig),
    #[serde(rename(deserialize = "gemma3_text"))]
    Gemma3(Gemma3Config),
    #[serde(alias = "new")]
    Gte(GTEConfig),
    #[serde(rename = "mpnet")]
    MPNet(MPNetConfig),
    #[allow(dead_code)]
    Mistral(MistralConfig),
    #[serde(rename(deserialize = "modernbert"))]
    ModernBert(ModernBertConfig),
    #[serde(rename(deserialize = "nomic_bert"))]
    NomicBert(NomicConfig),
    #[allow(dead_code)]
    Qwen2(Qwen2Config),
    #[allow(dead_code)]
    Qwen3(Qwen3Config),
    Roberta(BertConfig),
    #[serde(rename(deserialize = "static-embedding"))]
    StaticEmbedding(StaticEmbeddingConfig),
    XlmRoberta(BertConfig),
}

pub struct CandleBackend {
    device: Device,
    model: Box<dyn Model + Send>,
    dense_layers: Vec<Box<dyn DenseLayer + Send>>,
}

impl CandleBackend {
    pub fn new(
        model_path: &Path,
        dtype: String,
        model_type: ModelType,
        dense_paths: Option<Vec<String>>,
    ) -> Result<Self, BackendError> {
        // Default files
        let default_safetensors = model_path.join("model.safetensors");
        let default_pytorch = model_path.join("pytorch_model.bin");
        let static_embedding_safetensors = model_path.join("0_StaticEmbedding/model.safetensors");

        // Single Files
        let model_files = if default_safetensors.exists() {
            vec![default_safetensors]
        } else if default_pytorch.exists() {
            vec![default_pytorch]
        } else if static_embedding_safetensors.exists() {
            vec![static_embedding_safetensors]
        }
        // Sharded weights
        else {
            // Get index file
            let index_file = model_path.join("model.safetensors.index.json");

            // Parse file
            let index_file_string: String = std::fs::read_to_string(&index_file)
                .map_err(|err| BackendError::Start(err.to_string()))?;
            let json: serde_json::Value = serde_json::from_str(&index_file_string)
                .map_err(|err| BackendError::Start(err.to_string()))?;

            let weight_map = match json.get("weight_map") {
                None => {
                    return Err(BackendError::Start(format!(
                        "no weight map in {index_file:?}"
                    )));
                }
                Some(serde_json::Value::Object(map)) => map,
                Some(_) => {
                    return Err(BackendError::Start(format!(
                        "weight map in {index_file:?} is not a map"
                    )));
                }
            };
            let mut safetensors_files = std::collections::HashSet::new();
            for value in weight_map.values() {
                if let Some(file) = value.as_str() {
                    safetensors_files.insert(file.to_string());
                }
            }

            // Collect paths
            safetensors_files
                .iter()
                .map(|n| model_path.join(n))
                .collect()
        };

        // Load config
        let config: String = std::fs::read_to_string(model_path.join("config.json"))
            .context("Unable to read config file")
            .map_err(|err| BackendError::Start(format!("{err:?}")))?;
        let config: Config = serde_json::from_str(&config)
            .context("Model is not supported")
            .map_err(|err| BackendError::Start(format!("{err:?}")))?;

        // Get candle device
        let device = if candle::utils::cuda_is_available() {
            #[cfg(feature = "cuda")]
            match compatible_compute_cap() {
                Ok(true) => Device::new_cuda(0),
                Ok(false) => {
                    return Err(BackendError::Start(format!(
                        "Runtime compute cap {} is not compatible with compile time compute cap {}",
                        get_runtime_compute_cap().unwrap(),
                        get_compile_compute_cap().unwrap()
                    )));
                }
                Err(err) => {
                    tracing::warn!("Could not find a compatible CUDA device on host: {err:?}");
                    tracing::warn!("Using CPU instead");
                    Ok(Device::Cpu)
                }
            }
            #[cfg(not(feature = "cuda"))]
            Ok(Device::Cpu)
        } else if candle::utils::metal_is_available() {
            Device::new_metal(0)
        } else {
            Ok(Device::Cpu)
        }
        .map_err(|err| BackendError::Start(err.to_string()))?;

        // Get candle dtype
        let dtype = if &dtype == "float32" {
            Ok(DType::F32)
        } else if &dtype == "float16" {
            Ok(DType::F16)
        } else {
            Err(BackendError::Start(format!(
                "DType {dtype} is not supported"
            )))
        }?;

        let vb = if model_files.len() == 1 && model_files[0].extension().unwrap() == "bin" {
            VarBuilder::from_pth(&model_files[0], dtype, &device)
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device) }
        }
        .s()?;

        let model: Result<Box<dyn Model + Send>, BackendError> = match (config, &device) {
            #[cfg(not(feature = "cuda"))]
            (_, Device::Cuda(_)) => Err(BackendError::Start(
                "`cuda` feature is not enabled".to_string(),
            )),
            (Config::Bert(config), Device::Cpu | Device::Metal(_)) => match config {
                BertConfigWrapper::JinaBert(config) => {
                    tracing::info!("Starting JinaBert model on {:?}", device);
                    Ok(Box::new(JinaBertModel::load(vb, &config, model_type).s()?))
                }
                BertConfigWrapper::JinaCodeBert(config) => {
                    tracing::info!("Starting JinaCodeBert model on {:?}", device);
                    Ok(Box::new(
                        JinaCodeBertModel::load(vb, &config, model_type).s()?,
                    ))
                }
                BertConfigWrapper::Bert(config) => {
                    tracing::info!("Starting Bert model on {:?}", device);
                    Ok(Box::new(BertModel::load(vb, &config, model_type).s()?))
                }
            },
            (
                Config::Camembert(config) | Config::Roberta(config) | Config::XlmRoberta(config),
                Device::Cpu | Device::Metal(_),
            ) => {
                tracing::info!("Starting Bert model on {:?}", device);
                Ok(Box::new(
                    BertModel::load_roberta(vb, &config, model_type).s()?,
                ))
            }
            (Config::DebertaV2(config), Device::Cpu | Device::Metal(_)) => {
                tracing::info!("Starting DebertaV2 model on {:?}", device);
                Ok(Box::new(DebertaV2Model::load(vb, &config, model_type).s()?))
            }
            (Config::DistilBert(config), Device::Cpu | Device::Metal(_)) => {
                tracing::info!("Starting DistilBert model on {:?}", device);
                Ok(Box::new(
                    DistilBertModel::load(vb, &config, model_type).s()?,
                ))
            }
            (Config::Gemma3(config), Device::Cpu | Device::Metal(_)) => {
                if dtype != DType::F32 {
                    Err(BackendError::Start(
                        "Gemma3 is only supported in fp32 precision".to_string(),
                    ))
                } else {
                    tracing::info!("Starting Gemma3 model on {:?}", device);
                    Ok(Box::new(Gemma3Model::load(vb, &config, model_type).s()?))
                }
            }
            (Config::Gte(config), Device::Cpu | Device::Metal(_)) => {
                tracing::info!("Starting GTE model on {:?}", device);
                Ok(Box::new(GTEModel::load(vb, &config, model_type).s()?))
            }
            (Config::MPNet(config), _) => {
                tracing::info!("Starting MPNet model on {:?}", device);
                Ok(Box::new(MPNetModel::load(vb, &config, model_type).s()?))
            }
            (Config::Mistral(_), Device::Cpu | Device::Metal(_)) => Err(BackendError::Start(
                "Mistral is only supported on Cuda devices in fp16 with flash attention enabled"
                    .to_string(),
            )),
            (Config::ModernBert(config), Device::Cpu | Device::Metal(_)) => {
                tracing::info!("Starting ModernBert model on {:?}", device);
                Ok(Box::new(
                    ModernBertModel::load(vb, &config, model_type).s()?,
                ))
            }
            (Config::NomicBert(config), Device::Cpu | Device::Metal(_)) => {
                tracing::info!("Starting NomicBert model on {:?}", device);
                Ok(Box::new(NomicBertModel::load(vb, &config, model_type).s()?))
            }
            (Config::Qwen2(_), Device::Cpu | Device::Metal(_)) => Err(BackendError::Start(
                "Qwen2 is only supported on Cuda devices in fp16 with flash attention enabled"
                    .to_string(),
            )),
            (Config::Qwen3(config), Device::Cpu | Device::Metal(_)) => {
                tracing::info!("Starting Qwen3 model on {:?}", device);
                Ok(Box::new(Qwen3Model::load(vb, &config, model_type).s()?))
            }
            (Config::StaticEmbedding(config), Device::Cpu | Device::Metal(_)) => {
                tracing::info!("Starting StaticEmbedding model on {:?}", device);
                Ok(Box::new(
                    StaticEmbeddingModel::load(vb, &config, model_type).s()?,
                ))
            }
            #[cfg(feature = "cuda")]
            (Config::Bert(config), Device::Cuda(_)) => {
                if cfg!(any(feature = "flash-attn", feature = "flash-attn-v1"))
                    && dtype == DType::F16
                    // Allow disabling because of flash attention v1 precision problems
                    // See: https://github.com/huggingface/text-embeddings-inference/issues/37
                    && &std::env::var("USE_FLASH_ATTENTION").unwrap_or("True".to_string()).to_lowercase() == "true"
                {
                    match config {
                        BertConfigWrapper::JinaBert(config) => {
                            tracing::info!("Starting FlashJinaBert model on {:?}", device);
                            Ok(Box::new(
                                FlashJinaBertModel::load(vb, &config, model_type).s()?,
                            ))
                        }
                        BertConfigWrapper::JinaCodeBert(config) => {
                            tracing::info!("Starting FlashJinaCodeBert model on {:?}", device);
                            Ok(Box::new(
                                FlashJinaCodeBertModel::load(vb, &config, model_type).s()?,
                            ))
                        }
                        BertConfigWrapper::Bert(config) => {
                            tracing::info!("Starting FlashBert model on {:?}", device);
                            Ok(Box::new(FlashBertModel::load(vb, &config, model_type).s()?))
                        }
                    }
                } else {
                    match config {
                        BertConfigWrapper::JinaBert(config) => {
                            tracing::info!("Starting JinaBert model on {:?}", device);
                            Ok(Box::new(JinaBertModel::load(vb, &config, model_type).s()?))
                        }
                        BertConfigWrapper::JinaCodeBert(config) => {
                            tracing::info!("Starting JinaCodeBert model on {:?}", device);
                            Ok(Box::new(
                                JinaCodeBertModel::load(vb, &config, model_type).s()?,
                            ))
                        }
                        BertConfigWrapper::Bert(config) => {
                            tracing::info!("Starting Bert model on {:?}", device);
                            Ok(Box::new(BertModel::load(vb, &config, model_type).s()?))
                        }
                    }
                }
            }
            #[cfg(feature = "cuda")]
            (
                Config::Camembert(config) | Config::Roberta(config) | Config::XlmRoberta(config),
                Device::Cuda(_),
            ) => {
                if cfg!(any(feature = "flash-attn", feature = "flash-attn-v1"))
                    && dtype == DType::F16
                    // Allow disabling because of flash attention v1 precision problems
                    // See: https://github.com/huggingface/text-embeddings-inference/issues/37
                    && &std::env::var("USE_FLASH_ATTENTION").unwrap_or("True".to_string()).to_lowercase() == "true"
                {
                    tracing::info!("Starting FlashBert model on {:?}", device);
                    Ok(Box::new(
                        FlashBertModel::load_roberta(vb, &config, model_type).s()?,
                    ))
                } else {
                    tracing::info!("Starting Bert model on {:?}", device);
                    Ok(Box::new(
                        BertModel::load_roberta(vb, &config, model_type).s()?,
                    ))
                }
            }
            #[cfg(feature = "cuda")]
            (Config::DebertaV2(config), Device::Cuda(_)) => {
                tracing::info!("Starting DebertaV2 model on {:?}", device);
                Ok(Box::new(DebertaV2Model::load(vb, &config, model_type).s()?))
            }
            #[cfg(feature = "cuda")]
            (Config::DistilBert(config), Device::Cuda(_)) => {
                if cfg!(feature = "flash-attn")
                    && dtype == DType::F16
                    && &std::env::var("USE_FLASH_ATTENTION")
                        .unwrap_or("True".to_string())
                        .to_lowercase()
                        == "true"
                {
                    tracing::info!("Starting FlashDistilBert model on {:?}", device);
                    Ok(Box::new(
                        FlashDistilBertModel::load(vb, &config, model_type).s()?,
                    ))
                } else {
                    tracing::info!("Starting DistilBertModel model on {:?}", device);
                    Ok(Box::new(
                        DistilBertModel::load(vb, &config, model_type).s()?,
                    ))
                }
            }
            #[cfg(feature = "cuda")]
            (Config::Gemma3(config), Device::Cuda(_)) => {
                if dtype != DType::F32 {
                    Err(BackendError::Start(
                        "Gemma3 is only supported in fp32 precision".to_string(),
                    ))
                } else {
                    tracing::info!("Starting Gemma3 model on {:?}", device);
                    Ok(Box::new(Gemma3Model::load(vb, &config, model_type).s()?))
                }
            }
            #[cfg(feature = "cuda")]
            (Config::Gte(config), Device::Cuda(_)) => {
                if dtype != DType::F16
                    || !cfg!(any(feature = "flash-attn", feature = "flash-attn-v1"))
                    || &std::env::var("USE_FLASH_ATTENTION")
                        .unwrap_or("True".to_string())
                        .to_lowercase()
                        != "true"
                {
                    tracing::info!("Starting GTE model on {:?}", device);
                    Ok(Box::new(GTEModel::load(vb, &config, model_type).s()?))
                } else {
                    tracing::info!("Starting FlashGTE model on {:?}", device);
                    Ok(Box::new(FlashGTEModel::load(vb, &config, model_type).s()?))
                }
            }
            #[cfg(feature = "cuda")]
            (Config::Mistral(config), Device::Cuda(_)) => {
                if dtype != DType::F16
                    || !cfg!(feature = "flash-attn")
                    || get_runtime_compute_cap().unwrap() < 80
                    || &std::env::var("USE_FLASH_ATTENTION")
                        .unwrap_or("True".to_string())
                        .to_lowercase()
                        != "true"
                {
                    return Err(BackendError::Start("Mistral is only supported on Cuda devices in fp16 with flash attention v2 enabled".to_string()));
                }
                tracing::info!("Starting FlashMistral model on {:?}", device);
                Ok(Box::new(
                    FlashMistralModel::load(vb, &config, model_type).s()?,
                ))
            }
            #[cfg(feature = "cuda")]
            (Config::ModernBert(config), Device::Cuda(_)) => {
                if cfg!(feature = "flash-attn")
                    && dtype == DType::F16
                    // Allow disabling because of flash attention v1 precision problems
                    // See: https://github.com/huggingface/text-embeddings-inference/issues/37
                    && &std::env::var("USE_FLASH_ATTENTION").unwrap_or("True".to_string()).to_lowercase() == "true"
                {
                    tracing::info!("Starting FlashModernBert model on {:?}", device);
                    Ok(Box::new(
                        FlashModernBertModel::load(vb, &config, model_type).s()?,
                    ))
                } else {
                    #[cfg(feature = "flash-attn-v1")]
                    tracing::warn!("Flash attention V1 cannot be used with ModernBert because it lacks windowing support.");
                    tracing::info!("Starting ModernBert model on {:?}", device);
                    Ok(Box::new(
                        ModernBertModel::load(vb, &config, model_type).s()?,
                    ))
                }
            }
            #[cfg(feature = "cuda")]
            (Config::NomicBert(config), Device::Cuda(_)) => {
                if cfg!(feature = "flash-attn")
                    && dtype == DType::F16
                    && &std::env::var("USE_FLASH_ATTENTION")
                        .unwrap_or("True".to_string())
                        .to_lowercase()
                        == "true"
                {
                    tracing::info!("Starting FlashNomicBert model on {:?}", device);
                    Ok(Box::new(
                        FlashNomicBertModel::load(vb, &config, model_type).s()?,
                    ))
                } else {
                    tracing::info!("Starting NomicBert model on {:?}", device);
                    Ok(Box::new(NomicBertModel::load(vb, &config, model_type).s()?))
                }
            }
            #[cfg(feature = "cuda")]
            (Config::Qwen2(config), Device::Cuda(_)) => {
                if dtype != DType::F16
                    || !cfg!(any(feature = "flash-attn", feature = "flash-attn-v1"))
                    || &std::env::var("USE_FLASH_ATTENTION")
                        .unwrap_or("True".to_string())
                        .to_lowercase()
                        != "true"
                {
                    return Err(BackendError::Start("Qwen2 is only supported on Cuda devices in fp16 with flash attention v2 enabled".to_string()));
                }
                tracing::info!("Starting FlashQwen2 model on {:?}", device);
                Ok(Box::new(
                    FlashQwen2Model::load(vb, &config, model_type).s()?,
                ))
            }
            #[cfg(feature = "cuda")]
            (Config::Qwen3(config), Device::Cuda(_)) => {
                if dtype != DType::F16
                    || !cfg!(any(feature = "flash-attn", feature = "flash-attn-v1"))
                    || &std::env::var("USE_FLASH_ATTENTION")
                        .unwrap_or("True".to_string())
                        .to_lowercase()
                        != "true"
                {
                    tracing::info!("Starting Qwen3 model on {:?}", device);
                    Ok(Box::new(Qwen3Model::load(vb, &config, model_type).s()?))
                } else {
                    tracing::info!("Starting FlashQwen3 model on {:?}", device);
                    Ok(Box::new(
                        FlashQwen3Model::load(vb, &config, model_type).s()?,
                    ))
                }
            }
            #[cfg(feature = "cuda")]
            (Config::StaticEmbedding(config), Device::Cuda(_)) => {
                tracing::info!("Starting StaticEmbedding model on {:?}", device);
                Ok(Box::new(
                    StaticEmbeddingModel::load(vb, &config, model_type).s()?,
                ))
            }
        };

        let mut dense_layers = Vec::new();
        if let Some(dense_paths) = dense_paths {
            if !dense_paths.is_empty() {
                tracing::info!("Loading Dense module/s from path/s: {dense_paths:?}");

                for dense_path in dense_paths.iter() {
                    let dense_safetensors =
                        model_path.join(format!("{dense_path}/model.safetensors"));
                    let dense_pytorch = model_path.join(format!("{dense_path}/pytorch_model.bin"));

                    if dense_safetensors.exists() || dense_pytorch.exists() {
                        let dense_config_path =
                            model_path.join(format!("{dense_path}/config.json"));

                        let dense_config_str = std::fs::read_to_string(&dense_config_path)
                            .map_err(|err| {
                                BackendError::Start(format!(
                                    "Unable to read `{dense_path}/config.json` file: {err:?}",
                                ))
                            })?;
                        let dense_config: DenseConfig = serde_json::from_str(&dense_config_str)
                            .map_err(|err| {
                                BackendError::Start(format!(
                                    "Unable to parse `{dense_path}/config.json`: {err:?}",
                                ))
                            })?;

                        let dense_vb = if dense_safetensors.exists() {
                            unsafe {
                                VarBuilder::from_mmaped_safetensors(
                                    &[dense_safetensors],
                                    dtype,
                                    &device,
                                )
                            }
                            .s()?
                        } else {
                            VarBuilder::from_pth(&dense_pytorch, dtype, &device).s()?
                        };

                        let dense_layer = Box::new(Dense::load(dense_vb, &dense_config).s()?)
                            as Box<dyn DenseLayer + Send>;
                        dense_layers.push(dense_layer);

                        tracing::info!("Loaded Dense module from path: {dense_path}");
                    } else {
                        tracing::warn!("Dense module files not found for path: {dense_path}",);
                    }
                }
            }
        }

        Ok(Self {
            device,
            model: model?,
            dense_layers,
        })
    }
}

impl Backend for CandleBackend {
    fn max_batch_size(&self) -> Option<usize> {
        // Limit max batch size to 4 on CPU
        if matches!(self.device, Device::Cpu) {
            return Some(4);
        }
        None
    }

    fn health(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn is_padded(&self) -> bool {
        self.model.is_padded()
    }

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        let batch_size = batch.len();
        let pooled_indices = batch.pooled_indices.clone();
        let raw_indices = batch.raw_indices.clone();

        // Used for indexing in the raw_embeddings tensor
        let input_lengths: Vec<usize> = (0..batch.len())
            .map(|i| {
                (batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i]) as usize
            })
            .collect();

        // Run forward
        let (pooled_embeddings, raw_embeddings) = self.model.embed(batch).e()?;

        // Apply Dense layers sequentially if available
        let pooled_embeddings = match pooled_embeddings {
            None => None,
            Some(mut pooled_embeddings) => {
                for dense in &self.dense_layers {
                    pooled_embeddings = dense.forward(&pooled_embeddings).e()?;
                }
                Some(pooled_embeddings)
            }
        };

        // Device => Host data transfer
        let pooled_embeddings = match pooled_embeddings {
            None => vec![],
            Some(pooled_embeddings) => pooled_embeddings.to_dtype(DType::F32).e()?.to_vec2().e()?,
        };

        // This transfer is expensive...
        let raw_embeddings = match raw_embeddings {
            None => vec![],
            Some(raw_embeddings) => raw_embeddings.to_dtype(DType::F32).e()?.to_vec2().e()?,
        };

        let mut embeddings =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());
        for (i, e) in pooled_indices.into_iter().zip(pooled_embeddings) {
            embeddings.insert(i as usize, Embedding::Pooled(e));
        }

        let mut cumulative_length = 0;
        for i in raw_indices.into_iter() {
            let length = input_lengths[i as usize];
            let e = raw_embeddings[cumulative_length..cumulative_length + length].to_vec();
            embeddings.insert(i as usize, Embedding::All(e));
            cumulative_length += length;
        }

        Ok(embeddings)
    }

    fn predict(&self, batch: Batch) -> Result<Predictions, BackendError> {
        let batch_size = batch.len();

        let results = self.model.predict(batch).e()?;

        let results = results.to_dtype(DType::F32).e()?.to_vec2().e()?;

        let mut predictions =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());
        for (i, r) in results.into_iter().enumerate() {
            predictions.insert(i, r);
        }

        Ok(predictions)
    }
}

pub trait WrapErr<O> {
    fn s(self) -> Result<O, BackendError>;
    fn e(self) -> Result<O, BackendError>;
}

impl<O> WrapErr<O> for Result<O, candle::Error> {
    fn s(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Start(e.to_string()))
    }
    fn e(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Inference(e.to_string()))
    }
}
