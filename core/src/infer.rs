use crate::queue::{Entry, Metadata, Queue};
use crate::tokenization::Tokenization;
use crate::TextEmbeddingsError;
use std::sync::Arc;
use std::time::{Duration, Instant};
use text_embeddings_backend::{Backend, Embedding};
use tokio::sync::{oneshot, Notify, Semaphore};
use tracing::{instrument, Span};

/// Inference struct
#[derive(Clone)]
pub struct Infer {
    tokenization: Tokenization,
    queue: Queue,
    /// Shared notify
    notify_batching_task: Arc<Notify>,
    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,
    backend: Backend,
}

impl Infer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenization: Tokenization,
        queue: Queue,
        max_concurrent_requests: usize,
        backend: Backend,
    ) -> Self {
        let notify_batching_task = Arc::new(Notify::new());

        // Create two batching tasks to prefetch batches
        tokio::spawn(batching_task(
            backend.clone(),
            queue.clone(),
            notify_batching_task.clone(),
        ));
        tokio::spawn(batching_task(
            backend.clone(),
            queue.clone(),
            notify_batching_task.clone(),
        ));

        // Inference limit with a semaphore
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));

        Self {
            tokenization,
            queue,
            notify_batching_task,
            limit_concurrent_requests: semaphore,
            backend,
        }
    }

    #[instrument(skip(self))]
    pub async fn embed(
        &self,
        inputs: String,
        truncate: bool,
    ) -> Result<InferResponse, TextEmbeddingsError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        let _permit = self
            .clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                metrics::increment_counter!("te_request_failure", "err" => "overloaded");
                tracing::error!("{err}");
                err
            })?;

        let start_time = Instant::now();
        metrics::increment_counter!("te_embed_count");

        // Tokenization
        let encoding = self
            .tokenization
            .encode(inputs, truncate)
            .await
            .map_err(|err| {
                metrics::increment_counter!("te_request_failure", "err" => "tokenization");
                tracing::error!("{err}");
                err
            })?;

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = oneshot::channel();

        // Append the request to the queue
        self.queue.append(Entry {
            metadata: Metadata {
                response_tx,
                span: Span::current(),
                tokenization: start_time.elapsed(),
                queue_time: Instant::now(),
                batch_time: None,
                prompt_tokens: encoding.input_ids.len(),
            },
            encoding,
        });

        self.notify_batching_task.notify_one();

        let response = response_rx
            .await
            .expect(
                "Infer batching task dropped the sender without sending a response. This is a bug.",
            )
            .map_err(|err| {
                metrics::increment_counter!("te_request_failure", "err" => "inference");
                tracing::error!("{err}");
                err
            })?;

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        metrics::increment_counter!("te_embed_success");
        metrics::histogram!("te_embed_duration", total_time.as_secs_f64());
        metrics::histogram!(
            "te_embed_tokenization_duration",
            response.tokenization.as_secs_f64()
        );
        metrics::histogram!("te_embed_queue_duration", response.queue.as_secs_f64());
        metrics::histogram!(
            "te_embed_inference_duration",
            response.inference.as_secs_f64()
        );

        Ok(response)
    }

    #[instrument(skip(self))]
    pub async fn health(&self) -> bool {
        self.backend.health().await.is_ok()
    }
}

#[instrument(skip_all)]
async fn batching_task(backend: Backend, queue: Queue, notify: Arc<Notify>) {
    loop {
        notify.notified().await;

        while let Some(batch) = queue.next_batch().await {
            let results = backend.embed(batch.1).await;

            // Handle sending responses in another thread to not starve the model
            tokio::task::spawn_blocking(move || match results {
                Ok(embeddings) => {
                    batch.0.into_iter().zip(embeddings).for_each(|(m, e)| {
                        let _ = m.response_tx.send(Ok(InferResponse {
                            embeddings: e,
                            prompt_tokens: m.prompt_tokens,
                            tokenization: m.tokenization,
                            queue: m
                                .batch_time
                                .expect("Start time was not set. This is a bug.")
                                - m.queue_time,
                            inference: Instant::now()
                                - m.batch_time
                                    .expect("Start time was not set. This is a bug."),
                        }));
                    });
                }
                Err(err) => {
                    batch.0.into_iter().for_each(|m| {
                        let _ = m.response_tx.send(Err(err.clone()));
                    });
                }
            });
        }
    }
}

#[derive(Debug)]
pub struct InferResponse {
    pub embeddings: Embedding,
    pub prompt_tokens: usize,
    pub tokenization: Duration,
    pub queue: Duration,
    pub inference: Duration,
}
