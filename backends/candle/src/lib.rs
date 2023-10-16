#[cfg(feature = "cuda")]
mod compute_cap;
#[cfg(feature = "cuda")]
mod flash_attn;
mod models;

#[cfg(feature = "cuda")]
use crate::compute_cap::{incompatible_compute_cap, COMPILE_COMPUTE_CAP, RUNTIME_COMPUTE_CAP};
use crate::models::{BertModel, EmbeddingModel, QuantBertModel};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use models::Config;
use std::path::PathBuf;
use text_embeddings_backend_core::{BackendError, Batch, Embedding, EmbeddingBackend, Pool};

pub struct CandleBackend {
    model: Box<dyn EmbeddingModel + Send>,
    device: Device,
}

impl CandleBackend {
    pub fn new(model_path: PathBuf, dtype: String, pool: Pool) -> Result<Self, BackendError> {
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
        {
            return Err(BackendError::Start(format!(
                "Model {:?} is not supported",
                config.model_type
            )));
        }

        let model: Box<dyn EmbeddingModel + Send> = match device {
            Device::Cpu => {
                tracing::info!("Starting Bert model on CPU");

                if &dtype == "float32" || &dtype == "float16" {
                    let dtype = if &dtype == "float32" {
                        DType::F32
                    } else {
                        DType::F16
                    };

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

                    Box::new(BertModel::load(vb, &config, pool).s()?)
                } else if &dtype == "q6k" {
                    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                        model_path.join("ggml-model-q6k.bin"),
                    )
                    .map_err(|err| BackendError::Start(err.to_string()))?;
                    tracing::info!("vb");

                    Box::new(QuantBertModel::load(vb, &config, pool).s()?)
                } else {
                    return Err(BackendError::Start(format!(
                        "dtype {dtype} is not supported"
                    )));
                }
            }
            Device::Cuda(_) => {
                #[cfg(not(feature = "cuda"))]
                return Err(BackendError::Start(
                    "`cuda` feature is not enabled".to_string(),
                ));
                #[cfg(feature = "cuda")]
                {
                    use crate::models::FlashBertModel;

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

                    if incompatible_compute_cap() {
                        return Err(BackendError::Start(format!("Runtime compute cap {} is not compatible with compile time compute cap {}", *RUNTIME_COMPUTE_CAP, *COMPILE_COMPUTE_CAP)));
                    }

                    tracing::info!("Starting FlashBert model on Cuda");
                    Box::new(FlashBertModel::load(vb, &config, pool).s()?)
                }
            }
        };

        Ok(Self { model, device })
    }
}

impl EmbeddingBackend for CandleBackend {
    fn health(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn embed(&self, batch: Batch) -> Result<Vec<Embedding>, BackendError> {
        let results = self.model.embed(batch).e()?;
        let results = results.to_dtype(DType::F32).e()?.to_vec2().e()?;
        Ok(results)
    }

    fn max_batch_size(&self) -> Option<usize> {
        match self.device {
            Device::Cpu => Some(1),
            Device::Cuda(_) => None,
        }
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
