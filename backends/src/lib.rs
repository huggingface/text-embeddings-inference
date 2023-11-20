mod dtype;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use text_embeddings_backend_core::Backend as CoreBackend;
use tokio::sync::{mpsc, oneshot};
use tracing::{instrument, Span};

pub use crate::dtype::DType;
pub use text_embeddings_backend_core::{BackendError, Batch, Embedding, ModelType, Pool};

#[cfg(feature = "candle")]
use text_embeddings_backend_candle::CandleBackend;

#[cfg(feature = "python")]
use text_embeddings_backend_python::PythonBackend;

#[derive(Debug, Clone)]
pub struct Backend {
    /// Channel to communicate with the background thread
    backend_sender: mpsc::UnboundedSender<BackendCommand>,
    /// Health status
    health: Arc<AtomicBool>,
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
        let max_batch_size = backend.max_batch_size();

        tokio::task::spawn_blocking(move || backend_blocking_task(backend, backend_receiver));

        Ok(Self {
            backend_sender,
            health: Arc::new(AtomicBool::new(false)),
            max_batch_size,
            model_type,
        })
    }

    #[instrument(skip(self))]
    pub async fn health(&self) -> Result<(), BackendError> {
        let result = if self.health.load(Ordering::SeqCst) {
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
            };
            match &self.model_type {
                ModelType::Classifier => self.predict(batch).await.map(|_| ()),
                ModelType::Embedding(_) => self.embed(batch).await.map(|_| ()),
            }
        };

        // Update health
        self.health.store(result.is_ok(), Ordering::SeqCst);
        result
    }

    #[instrument(skip_all)]
    pub async fn embed(&self, batch: Batch) -> Result<(Vec<Embedding>, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .send(BackendCommand::Embed(batch, Span::current(), sender))
            .expect("No backend receiver. This is a bug.");
        let result = receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        );

        // Update health
        self.health.store(result.is_ok(), Ordering::SeqCst);
        result
    }

    #[instrument(skip_all)]
    pub async fn predict(&self, batch: Batch) -> Result<(Vec<Vec<f32>>, Duration), BackendError> {
        let (sender, receiver) = oneshot::channel();

        self.backend_sender
            .send(BackendCommand::Predict(batch, Span::current(), sender))
            .expect("No backend receiver. This is a bug.");
        let result = receiver.await.expect(
            "Backend blocking task dropped the sender without send a response. This is a bug.",
        );

        // Update health
        self.health.store(result.is_ok(), Ordering::SeqCst);
        result
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
            use std::thread;

            return Ok(Box::new(
                thread::spawn(move || {
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

fn backend_blocking_task(
    backend: Box<dyn CoreBackend + Send>,
    mut command_receiver: mpsc::UnboundedReceiver<BackendCommand>,
) {
    while let Some(cmd) = command_receiver.blocking_recv() {
        let start = Instant::now();
        match cmd {
            BackendCommand::Health(span, sender) => {
                let _span = span.entered();
                let _ = sender.send(backend.health());
            }
            BackendCommand::Embed(batch, span, sender) => {
                let _span = span.entered();
                let _ = sender.send(backend.embed(batch).map(|e| (e, start.elapsed())));
            }
            BackendCommand::Predict(batch, span, sender) => {
                let _span = span.entered();
                let _ = sender.send(backend.predict(batch).map(|e| (e, start.elapsed())));
            }
        }
    }
}

enum BackendCommand {
    Health(Span, oneshot::Sender<Result<(), BackendError>>),
    Embed(
        Batch,
        Span,
        oneshot::Sender<Result<(Vec<Embedding>, Duration), BackendError>>,
    ),
    Predict(
        Batch,
        Span,
        #[allow(clippy::type_complexity)]
        oneshot::Sender<Result<(Vec<Vec<f32>>, Duration), BackendError>>,
    ),
}
