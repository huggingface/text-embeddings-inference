mod dtype;

use hf_hub::api::tokio::{ApiError, ApiRepo};
use std::cmp::{max, min};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use text_embeddings_backend_core::{Backend as CoreBackend, Predictions};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{instrument, Span};

pub use crate::dtype::DType;
pub use text_embeddings_backend_core::{
    BackendError, Batch, Embedding, Embeddings, ModelType, Pool,
};

#[cfg(feature = "candle")]
use text_embeddings_backend_candle::CandleBackend;

#[cfg(feature = "ort")]
use text_embeddings_backend_ort::OrtBackend;

#[cfg(feature = "python")]
use text_embeddings_backend_python::PythonBackend;

#[derive(Debug, Clone)]
pub struct Backend {
    /// Channel to communicate with the background thread
    backend_sender: mpsc::Sender<BackendCommand>,
    /// Health status
    health_receiver: watch::Receiver<bool>,
    _backend_thread: Arc<BackendThread>,
    pub padded_model: bool,
    pub max_batch_size: Option<usize>,
    pub model_type: ModelType,
}

impl Backend {
    pub fn new(
        model_path: PathBuf,
        dtype: DType,
        model_type: ModelType,
        uds_path: String,
        otlp_endpoint: Option<String>,
        otlp_service_name: String,
    ) -> Result<Self, BackendError> {
        let (backend_sender, backend_receiver) = mpsc::channel(8);

        let backend = init_backend(
            model_path,
            dtype,
            model_type.clone(),
            uds_path,
            otlp_endpoint,
            otlp_service_name,
        )?;
        let padded_model = backend.is_padded();
        let max_batch_size = backend.max_batch_size();

        let (health_sender, health_receiver) = watch::channel(false);
        let _backend_thread =
            Arc::new(BackendThread::new(backend, backend_receiver, health_sender));

        Ok(Self {
            backend_sender,
            health_receiver,
            _backend_thread,
            padded_model,
            max_batch_size,
            model_type,
        })
    }

    #[instrument(skip(self))]
    pub async fn warmup(
        &self,
        max_input_length: usize,
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
    ) -> Result<(), BackendError> {
        let mut input_ids = Vec::with_capacity(max_batch_tokens);
        let mut token_type_ids = Vec::with_capacity(max_batch_tokens);
        let mut position_ids = Vec::with_capacity(max_batch_tokens);

        let mut cumulative_seq_lengths = vec![0];
        let mut pooled_indices = Vec::new();

        let mut i = 0_u32;
        let mut remaining = max_batch_tokens;
        let mut cumulative_length = 0;
        let mut max_length = 0;

        while remaining > 0 {
            let request_length = min(remaining, max_input_length);
            cumulative_length += request_length;
            max_length = max(max_length, request_length as u32);

            input_ids.extend(vec![0; request_length]);
            token_type_ids.extend(vec![0; request_length]);
            position_ids.extend((0..request_length as u32).collect::<Vec<u32>>());

            cumulative_seq_lengths.push(cumulative_length as u32);
            pooled_indices.push(i);

            i += 1;
            remaining = remaining.saturating_sub(max_input_length);
            if let Some(max_batch_requests) = &max_batch_requests {
                if i as usize == *max_batch_requests {
                    break;
                }
            }
        }

        let batch = Batch {
            input_ids,
            token_type_ids,
            position_ids,
            cumulative_seq_lengths,
            max_length,
            pooled_indices,
            raw_indices: vec![],
        };

        match &self.model_type {
            ModelType::Classifier => self.predict(batch).await.map(|_| ()),
            ModelType::Embedding(_) => self.embed(batch).await.map(|_| ()),
        }
    }

    #[instrument(skip(self))]
    pub async fn health(&self) -> Result<(), BackendError> {
        if *self.health_receiver.borrow() {
            // The backend is healthy. Only do a basic health check by calling the
            // the underlying health method.

            let (sender, receiver) = oneshot::channel();
            self.backend_sender
                .send(BackendCommand::Health(Span::current(), sender))
                .await
                .expect("No backend receiver. This is a bug.");
            receiver.await.expect(
                "Backend blocking task dropped the sender without sending a response. This is a bug.",
            )
        } else {
            // The backend is un-healthy or only just started. Do a more advanced health check
            // by calling the model forward on a test batch

            let batch = Batch {
                input_ids: vec![0],
                token_type_ids: vec![0],
                position_ids: vec![0],
                cumulative_seq_lengths: vec![0, 1],
                max_length: 1,
                pooled_indices: vec![0],
                raw_indices: vec![],
            };
            match &self.model_type {
                ModelType::Classifier => self.predict(batch).await.map(|_| ()),
                ModelType::Embedding(_) => self.embed(batch).await.map(|_| ()),
            }
        }
    }

    #[instrument(skip(self))]
    pub fn health_watcher(&self) -> watch::Receiver<bool> {
        self.health_receiver.clone()
    }

    #[instrument(skip_all)]
    pub async fn embed(&self, batch: Batch) -> Result<(Embeddings, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .try_send(BackendCommand::Embed(batch, Span::current(), sender))
            .expect("No backend receiver. This is a bug.");
        receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        )
    }

    #[instrument(skip_all)]
    pub async fn predict(&self, batch: Batch) -> Result<(Predictions, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .try_send(BackendCommand::Predict(batch, Span::current(), sender))
            .expect("No backend receiver. This is a bug.");
        receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        )
    }
}

#[allow(unused)]
fn init_backend(
    model_path: PathBuf,
    dtype: DType,
    model_type: ModelType,
    uds_path: String,
    otlp_endpoint: Option<String>,
    otlp_service_name: String,
) -> Result<Box<dyn CoreBackend + Send>, BackendError> {
    if cfg!(feature = "candle") {
        #[cfg(feature = "candle")]
        return Ok(Box::new(CandleBackend::new(
            model_path,
            dtype.to_string(),
            model_type,
        )?));
    } else if cfg!(feature = "python") {
        #[cfg(feature = "python")]
        {
            return Ok(Box::new(
                std::thread::spawn(move || {
                    PythonBackend::new(
                        model_path.to_str().unwrap().to_string(),
                        dtype.to_string(),
                        model_type,
                        uds_path,
                        otlp_endpoint,
                        otlp_service_name,
                    )
                })
                .join()
                .expect("Python Backend management thread failed")?,
            ));
        }
    } else if cfg!(feature = "ort") {
        #[cfg(feature = "ort")]
        return Ok(Box::new(OrtBackend::new(
            model_path,
            dtype.to_string(),
            model_type,
        )?));
    }
    Err(BackendError::NoBackend)
}

#[derive(Debug)]
struct BackendThread(Option<JoinHandle<()>>);

impl BackendThread {
    fn new(
        backend: Box<dyn CoreBackend + Send>,
        mut backend_receiver: mpsc::Receiver<BackendCommand>,
        health_sender: watch::Sender<bool>,
    ) -> Self {
        let handle = std::thread::spawn(move || {
            while let Some(cmd) = backend_receiver.blocking_recv() {
                let start = Instant::now();
                let mut healthy = false;
                match cmd {
                    BackendCommand::Health(span, sender) => {
                        let _span = span.entered();
                        let _ = sender.send(backend.health().map(|_| healthy = true));
                    }
                    BackendCommand::Embed(batch, span, sender) => {
                        let _span = span.entered();
                        let _ = sender.send(backend.embed(batch).map(|e| {
                            healthy = true;
                            (e, start.elapsed())
                        }));
                    }
                    BackendCommand::Predict(batch, span, sender) => {
                        let _span = span.entered();
                        let _ = sender.send(backend.predict(batch).map(|e| {
                            healthy = true;
                            (e, start.elapsed())
                        }));
                    }
                };
                let _ = health_sender.send(healthy);
            }
        });
        Self(Some(handle))
    }
}

impl Drop for BackendThread {
    fn drop(&mut self) {
        self.0.take().unwrap().join().unwrap();
    }
}

enum BackendCommand {
    Health(Span, oneshot::Sender<Result<(), BackendError>>),
    Embed(
        Batch,
        Span,
        oneshot::Sender<Result<(Embeddings, Duration), BackendError>>,
    ),
    Predict(
        Batch,
        Span,
        #[allow(clippy::type_complexity)]
        oneshot::Sender<Result<(Predictions, Duration), BackendError>>,
    ),
}

pub async fn download_weights(api: &ApiRepo) -> Result<Vec<PathBuf>, ApiError> {
    let model_files = if cfg!(feature = "python") || cfg!(feature = "candle") {
        match download_safetensors(api).await {
            Ok(p) => p,
            Err(_) => {
                tracing::warn!("safetensors weights not found. Using `pytorch_model.bin` instead. Model loading will be significantly slower.");
                tracing::info!("Downloading `pytorch_model.bin`");
                let p = api.get("pytorch_model.bin").await?;
                vec![p]
            }
        }
    } else if cfg!(feature = "ort") {
        match download_onnx(api).await {
            Ok(p) => p,
            Err(err) => {
                panic!("failed to download `model.onnx` or `model.onnx_data`. Check the onnx file exists in the repository. {err}");
            }
        }
    } else {
        unreachable!()
    };

    Ok(model_files)
}

async fn download_safetensors(api: &ApiRepo) -> Result<Vec<PathBuf>, ApiError> {
    // Single file
    tracing::info!("Downloading `model.safetensors`");
    match api.get("model.safetensors").await {
        Ok(p) => return Ok(vec![p]),
        Err(err) => tracing::warn!("Could not download `model.safetensors`: {}", err),
    };

    // Sharded weights
    // Download and parse index file
    tracing::info!("Downloading `model.safetensors.index.json`");
    let index_file = api.get("model.safetensors.index.json").await?;
    let index_file_string: String =
        std::fs::read_to_string(index_file).expect("model.safetensors.index.json is corrupted");
    let json: serde_json::Value = serde_json::from_str(&index_file_string)
        .expect("model.safetensors.index.json is corrupted");

    let weight_map = match json.get("weight_map") {
        Some(serde_json::Value::Object(map)) => map,
        _ => panic!("model.safetensors.index.json is corrupted"),
    };

    let mut safetensors_filenames = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_filenames.insert(file.to_string());
        }
    }

    // Download weight files
    let mut safetensors_files = Vec::new();
    for n in safetensors_filenames {
        tracing::info!("Downloading `{}`", n);
        safetensors_files.push(api.get(&n).await?);
    }

    Ok(safetensors_files)
}

async fn download_onnx(api: &ApiRepo) -> Result<Vec<PathBuf>, ApiError> {
    let mut model_files: Vec<PathBuf> = Vec::new();

    tracing::info!("Downloading `model.onnx`");
    match api.get("model.onnx").await {
        Ok(p) => model_files.push(p),
        Err(err) => {
            tracing::warn!("Could not download `model.onnx`: {err}");
            tracing::info!("Downloading `onnx/model.onnx`");
            let p = api.get("onnx/model.onnx").await?;
            model_files.push(p.parent().unwrap().to_path_buf())
        }
    };

    tracing::info!("Downloading `model.onnx_data`");
    match api.get("model.onnx_data").await {
        Ok(p) => model_files.push(p),
        Err(err) => {
            tracing::warn!("Could not download `model.onnx_data`: {err}");
            tracing::info!("Downloading `onnx/model.onnx_data`");

            match api.get("onnx/model.onnx_data").await {
                Ok(p) => model_files.push(p.parent().unwrap().to_path_buf()),
                Err(err) => tracing::warn!("Could not download `onnx/model.onnx_data`: {err}"),
            }
        }
    }

    Ok(model_files)
}
