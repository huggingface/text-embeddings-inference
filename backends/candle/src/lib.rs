mod alibi;
#[cfg(feature = "cuda")]
mod compute_cap;
#[cfg(feature = "cuda")]
mod flash_attn;
mod layers;
mod models;

#[cfg(feature = "cuda")]
use crate::compute_cap::{get_runtime_compute_cap, incompatible_compute_cap, get_compile_compute_cap};
#[cfg(feature = "cuda")]
use crate::models::FlashBertModel;
use crate::models::{BertModel, JinaBertModel, Model, PositionEmbeddingType};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use models::Config;
use std::path::PathBuf;
use text_embeddings_backend_core::{Backend, BackendError, Batch, Embedding, ModelType};

pub struct CandleBackend {
    model: Box<dyn Model + Send>,
}

impl CandleBackend {
    pub fn new(
        model_path: PathBuf,
        dtype: String,
        model_type: ModelType,
    ) -> Result<Self, BackendError> {
        // Load config
        let config: String = std::fs::read_to_string(model_path.join("config.json"))
            .map_err(|err| BackendError::Start(err.to_string()))?;
        let config: Config =
            serde_json::from_str(&config).map_err(|err| BackendError::Start(err.to_string()))?;

        // Get candle device
        let device = match Device::cuda_if_available(0) {
            Ok(device) => device,
            Err(err) => return Err(BackendError::Start(err.to_string())),
        };

        // Check model type
        if config.model_type != Some("bert".to_string())
            && config.model_type != Some("xlm-roberta".to_string())
            && config.model_type != Some("camembert".to_string())
            && config.model_type != Some("roberta".to_string())
        {
            return Err(BackendError::Start(format!(
                "Model {:?} is not supported",
                config.model_type
            )));
        }

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

        let safetensors_path = model_path.join("model.safetensors");
        let vb = if safetensors_path.exists() {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[model_path.join("model.safetensors")],
                    dtype,
                    &device,
                )
            }
        } else {
            VarBuilder::from_pth(model_path.join("pytorch_model.bin"), dtype, &device)
        }
        .s()?;

        let model: Box<dyn Model + Send> = match device {
            Device::Cpu => {
                if config.position_embedding_type == PositionEmbeddingType::Alibi {
                    tracing::info!("Starting JinaBert model on CPU");
                    Box::new(JinaBertModel::load(vb, &config, model_type).s()?)
                } else {
                    tracing::info!("Starting Bert model on CPU");
                    Box::new(BertModel::load(vb, &config, model_type).s()?)
                }
            }
            Device::Cuda(_) => {
                #[cfg(not(feature = "cuda"))]
                return Err(BackendError::Start(
                    "`cuda` feature is not enabled".to_string(),
                ));
                #[cfg(feature = "cuda")]
                {
                    if incompatible_compute_cap() {
                        return Err(BackendError::Start(format!("Runtime compute cap {} is not compatible with compile time compute cap {}", get_runtime_compute_cap(), get_compile_compute_cap())));
                    }

                    if cfg!(any(feature = "flash-attn", feature = "flash-attn-v1"))
                        && dtype == DType::F16
                        && config.position_embedding_type == PositionEmbeddingType::Absolute
                        // Allow disabling because of flash attention v1 precision problems
                        // See: https://github.com/huggingface/text-embeddings-inference/issues/37
                        && &std::env::var("USE_FLASH_ATTENTION").unwrap_or("True".to_string()).to_lowercase() == "true"
                    {
                        tracing::info!("Starting FlashBert model on Cuda");
                        Box::new(FlashBertModel::load(vb, &config, model_type).s()?)
                    } else if config.position_embedding_type == PositionEmbeddingType::Alibi {
                        tracing::info!("Starting JinaBert model on Cuda");
                        Box::new(JinaBertModel::load(vb, &config, model_type).s()?)
                    } else {
                        tracing::info!("Starting Bert model on Cuda");
                        Box::new(BertModel::load(vb, &config, model_type).s()?)
                    }
                }
            }
        };

        Ok(Self { model })
    }
}

impl Backend for CandleBackend {
    fn health(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn is_padded(&self) -> bool {
        self.model.is_padded()
    }

    fn embed(&self, batch: Batch) -> Result<Vec<Embedding>, BackendError> {
        let results = self.model.embed(batch).e()?;
        let results = results.to_dtype(DType::F32).e()?.to_vec2().e()?;
        Ok(results)
    }

    fn predict(&self, batch: Batch) -> Result<Vec<Vec<f32>>, BackendError> {
        let results = self.model.predict(batch).e()?;
        let results = results.to_dtype(DType::F32).e()?.to_vec2().e()?;
        Ok(results)
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

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};
    use candle_nn::seq;
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};
    use text_embeddings_backend_core::Pool;
    use tokenizers::pre_tokenizers::metaspace::PrependScheme;
    use tokenizers::{PreTokenizerWrapper, Tokenizer};

    fn download_artifacts(model_id: &'static str) -> Result<PathBuf> {
        let mut builder = ApiBuilder::new().with_progress(false);

        let api = builder.build().unwrap();
        let api_repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        api_repo.get("config.json")?;
        api_repo.get("tokenizer.json")?;

        let model_root = match api_repo.get("model.safetensors") {
            Ok(p) => p,
            Err(_) => {
                let p = api_repo.get("pytorch_model.bin")?;
                tracing::warn!("`model.safetensors` not found. Using `pytorch_model.bin` instead. Model loading will be significantly slower.");
                p
            }
        }
            .parent().unwrap()
            .to_path_buf();
        Ok(model_root)
    }

    fn load_tokenizer(model_root: &PathBuf) -> Result<Tokenizer> {
        // Load tokenizer
        let tokenizer_path = model_root.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).expect("tokenizer.json not found");
        // See https://github.com/huggingface/tokenizers/pull/1357
        if let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer() {
            if let PreTokenizerWrapper::Metaspace(m) = pre_tokenizer {
                // We are forced to clone since `Tokenizer` does not have a `get_mut` for `pre_tokenizer`
                let mut m = m.clone();
                m.set_prepend_scheme(PrependScheme::First);
                tokenizer.with_pre_tokenizer(PreTokenizerWrapper::Metaspace(m));
            } else if let PreTokenizerWrapper::Sequence(s) = pre_tokenizer {
                // We are forced to clone since `Tokenizer` does not have a `get_mut` for `pre_tokenizer`
                let mut s = s.clone();
                for pre_tokenizer in s.get_pre_tokenizers_mut() {
                    if let PreTokenizerWrapper::Metaspace(m) = pre_tokenizer {
                        m.set_prepend_scheme(PrependScheme::First);
                    }
                }
                tokenizer.with_pre_tokenizer(PreTokenizerWrapper::Sequence(s));
            }
        }

        tokenizer.with_padding(None);
        Ok(tokenizer)
    }

    fn round(v: &Vec<f32>, digits: i32) -> Vec<f32> {
        let b = 10f32.powi(digits);
        v.iter().map(|v| f32::round(v * b) / b).collect()
    }

    #[test]
    fn test_mini() -> Result<()> {
        let model_root = download_artifacts("sentence-transformers/all-MiniLM-L6-v2")?;

        let tokenizer = load_tokenizer(&model_root)?;
        let backend = CandleBackend::new(
            model_root,
            "float32".to_string(),
            ModelType::Embedding(Pool::Mean),
        )?;

        let encoding = tokenizer.encode("What is Deep Learning?", true).unwrap();
        let seq_len = encoding.len();

        let batch = Batch {
            input_ids: encoding.get_ids().to_vec(),
            token_type_ids: encoding.get_type_ids().to_vec(),
            position_ids: (0..seq_len as u32).collect(),
            cumulative_seq_lengths: vec![0, seq_len as u32],
            max_length: seq_len as u32,
        };

        let embeddings = backend.embed(batch)?;
        insta::assert_yaml_snapshot!(round(&embeddings[0], 3));
        Ok(())
    }
}
