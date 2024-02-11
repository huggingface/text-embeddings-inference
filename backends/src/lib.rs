mod dtype;

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

#[cfg(feature = "python")]
use text_embeddings_backend_python::PythonBackend;

#[derive(Debug, Clone)]
pub struct Backend {
    /// Channel to communicate with the background thread
    backend_sender: mpsc::UnboundedSender<BackendCommand>,
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
    ) -> Result<Self, BackendError> {
        let (backend_sender, backend_receiver) = mpsc::unbounded_channel();

        let backend = init_backend(
            model_path,
            dtype,
            model_type.clone(),
            uds_path,
            otlp_endpoint,
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
    pub async fn health(&self) -> Result<(), BackendError> {
        if *self.health_receiver.borrow() {
            // The backend is healthy. Only do a basic health check by calling the
            // the underlying health method.

            let (sender, receiver) = oneshot::channel();
            self.backend_sender
                .send(BackendCommand::Health(Span::current(), sender))
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
            .send(BackendCommand::Embed(batch, Span::current(), sender))
            .expect("No backend receiver. This is a bug.");
        receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        )
    }

    #[instrument(skip_all)]
    pub async fn predict(&self, batch: Batch) -> Result<(Predictions, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .send(BackendCommand::Predict(batch, Span::current(), sender))
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
                    )
                })
                .join()
                .expect("Python Backend management thread failed")?,
            ));
        }
    }
    Err(BackendError::NoBackend)
}

#[derive(Debug)]
struct BackendThread(Option<JoinHandle<()>>);

impl BackendThread {
    fn new(
        backend: Box<dyn CoreBackend + Send>,
        mut backend_receiver: mpsc::UnboundedReceiver<BackendCommand>,
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
