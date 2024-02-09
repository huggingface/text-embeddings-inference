use crate::queue::{Entry, Metadata, NextBatch, Queue};
use crate::tokenization::{EncodingInput, RawEncoding, Tokenization};
use crate::TextEmbeddingsError;
use std::sync::Arc;
use std::time::{Duration, Instant};
use text_embeddings_backend::{Backend, BackendError, ModelType};
use tokio::sync::{mpsc, oneshot, watch, Notify, OwnedSemaphorePermit, Semaphore};
use tracing::instrument;

/// Inference struct
#[derive(Debug, Clone)]
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

        let (embed_sender, embed_receiver) = mpsc::unbounded_channel();

        // Create two batching tasks to prefetch batches
        tokio::spawn(batching_task(
            queue.clone(),
            notify_batching_task.clone(),
            embed_sender.clone(),
        ));
        tokio::spawn(batching_task(
            queue.clone(),
            notify_batching_task.clone(),
            embed_sender,
        ));

        // Create embed task to communicate with backend
        tokio::spawn(backend_task(backend.clone(), embed_receiver));

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
    pub async fn tokenize<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        add_special_tokens: bool,
    ) -> Result<RawEncoding, TextEmbeddingsError> {
        self.tokenization
            .tokenize(inputs.into(), add_special_tokens)
            .await
            .map_err(|err| {
                metrics::increment_counter!("te_request_failure", "err" => "tokenization");
                tracing::error!("{err}");
                err
            })
    }

    #[instrument(skip(self))]
    pub fn try_acquire_permit(&self) -> Result<OwnedSemaphorePermit, TextEmbeddingsError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        self.clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                metrics::increment_counter!("te_request_failure", "err" => "overloaded");
                tracing::error!("{err}");
                TextEmbeddingsError::from(err)
            })
    }
    #[instrument(skip(self))]
    pub async fn acquire_permit(&self) -> OwnedSemaphorePermit {
        // Limit concurrent requests by acquiring a permit from the semaphore
        self.clone()
            .limit_concurrent_requests
            .acquire_owned()
            .await
            .expect("Semaphore has been closed. This is a bug.")
    }

    #[instrument(skip(self, permit))]
    pub async fn embed_raw<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        permit: OwnedSemaphorePermit,
    ) -> Result<RawEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        let results = self
            .embed(inputs, truncate, false, &start_time, permit)
            .await?;

        let response = match results {
            InferResult::RawEmbedding(embeddings) => embeddings,
            _ => panic!("unexpected enum variant. this is a bug"),
        };

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        metrics::increment_counter!("te_embed_success");
        metrics::histogram!("te_embed_duration", total_time.as_secs_f64());
        metrics::histogram!(
            "te_embed_tokenization_duration",
            response.metadata.tokenization.as_secs_f64()
        );
        metrics::histogram!(
            "te_embed_queue_duration",
            response.metadata.queue.as_secs_f64()
        );
        metrics::histogram!(
            "te_embed_inference_duration",
            response.metadata.inference.as_secs_f64()
        );

        Ok(response)
    }

    #[instrument(skip(self, permit))]
    pub async fn embed_pooled<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        normalize: bool,
        permit: OwnedSemaphorePermit,
    ) -> Result<PooledEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        let results = self
            .embed(inputs, truncate, true, &start_time, permit)
            .await?;

        let response = match results {
            InferResult::PooledEmbedding(mut embeddings) => {
                if normalize {
                    // Normalize embedding
                    let scale = (1.0
                        / embeddings
                            .results
                            .iter()
                            .map(|v| {
                                let v = *v as f64;
                                v * v
                            })
                            .sum::<f64>()
                            .sqrt()) as f32;
                    for v in embeddings.results.iter_mut() {
                        *v *= scale;
                    }
                }
                embeddings
            }
            _ => panic!("unexpected enum variant. this is a bug"),
        };

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        metrics::increment_counter!("te_embed_success");
        metrics::histogram!("te_embed_duration", total_time.as_secs_f64());
        metrics::histogram!(
            "te_embed_tokenization_duration",
            response.metadata.tokenization.as_secs_f64()
        );
        metrics::histogram!(
            "te_embed_queue_duration",
            response.metadata.queue.as_secs_f64()
        );
        metrics::histogram!(
            "te_embed_inference_duration",
            response.metadata.inference.as_secs_f64()
        );

        Ok(response)
    }

    async fn embed<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        pooling: bool,
        start_time: &Instant,
        _permit: OwnedSemaphorePermit,
    ) -> Result<InferResult, TextEmbeddingsError> {
        if self.is_classifier() {
            metrics::increment_counter!("te_request_failure", "err" => "model_type");
            let message = "Model is not an embedding model".to_string();
            tracing::error!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        metrics::increment_counter!("te_embed_count");

        // Tokenization
        let encoding = self
            .tokenization
            .encode(inputs.into(), truncate)
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
                tokenization: start_time.elapsed(),
                queue_time: Instant::now(),
                prompt_tokens: encoding.input_ids.len(),
                pooling,
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

        Ok(response)
    }

    #[instrument(skip(self, _permit))]
    pub async fn predict<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        raw_scores: bool,
        _permit: OwnedSemaphorePermit,
    ) -> Result<ClassificationInferResponse, TextEmbeddingsError> {
        if !self.is_classifier() {
            metrics::increment_counter!("te_request_failure", "err" => "model_type");
            let message = "Model is not a classifier model".to_string();
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let start_time = Instant::now();
        metrics::increment_counter!("te_predict_count");

        // Tokenization
        let encoding = self
            .tokenization
            .encode(inputs.into(), truncate)
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
                tokenization: start_time.elapsed(),
                queue_time: Instant::now(),
                prompt_tokens: encoding.input_ids.len(),
                pooling: true,
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

        let mut response = match response {
            InferResult::Classification(response) => response,
            _ => panic!("unexpected enum variant. this is a bug"),
        };

        if !raw_scores {
            // Softmax
            if response.results.len() > 1 {
                let max = *response
                    .results
                    .iter()
                    .max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
                    .unwrap();

                let mut den = 0.0;
                for v in response.results.iter_mut() {
                    *v = (*v - max).exp();
                    den += *v;
                }
                for v in response.results.iter_mut() {
                    *v /= den;
                }
            }
            // Sigmoid
            else {
                response.results[0] = 1.0 / (1.0 + (-response.results[0]).exp());
            }
        }

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        metrics::increment_counter!("te_predict_success");
        metrics::histogram!("te_predict_duration", total_time.as_secs_f64());
        metrics::histogram!(
            "te_predict_tokenization_duration",
            response.metadata.tokenization.as_secs_f64()
        );
        metrics::histogram!(
            "te_predict_queue_duration",
            response.metadata.queue.as_secs_f64()
        );
        metrics::histogram!(
            "te_predict_inference_duration",
            response.metadata.inference.as_secs_f64()
        );

        Ok(response)
    }

    #[instrument(skip(self))]
    pub fn is_classifier(&self) -> bool {
        matches!(self.backend.model_type, ModelType::Classifier)
    }

    #[instrument(skip(self))]
    pub async fn health(&self) -> bool {
        self.backend.health().await.is_ok()
    }

    #[instrument(skip(self))]
    pub fn health_watcher(&self) -> watch::Receiver<bool> {
        self.backend.health_watcher()
    }
}

#[instrument(skip_all)]
async fn batching_task(
    queue: Queue,
    notify: Arc<Notify>,
    embed_sender: mpsc::UnboundedSender<(NextBatch, oneshot::Sender<()>)>,
) {
    loop {
        notify.notified().await;

        while let Some(next_batch) = queue.next_batch().await {
            let (callback_sender, callback_receiver) = oneshot::channel();
            embed_sender
                .send((next_batch, callback_sender))
                .expect("embed receiver was dropped. This is a bug.");
            let _ = callback_receiver.await;
        }
    }
}

#[instrument(skip_all)]
async fn backend_task(
    backend: Backend,
    mut embed_receiver: mpsc::UnboundedReceiver<(NextBatch, oneshot::Sender<()>)>,
) {
    while let Some((batch, _callback)) = embed_receiver.recv().await {
        match &backend.model_type {
            ModelType::Classifier => {
                let results = backend.predict(batch.1).await;

                // Handle sending responses in another thread to avoid starving the backend
                std::thread::spawn(move || match results {
                    Ok((results, inference_duration)) => {
                        batch.0.into_iter().zip(results).for_each(|(m, r)| {
                            let infer_metadata = InferMetadata {
                                prompt_tokens: m.prompt_tokens,
                                tokenization: m.tokenization,
                                queue: m.queue_time.elapsed() - inference_duration,
                                inference: inference_duration,
                            };

                            let _ = m.response_tx.send(Ok(InferResult::Classification(
                                ClassificationInferResponse {
                                    results: r,
                                    metadata: infer_metadata,
                                },
                            )));
                        });
                    }
                    Err(err) => {
                        batch.0.into_iter().for_each(|m| {
                            let _ = m.response_tx.send(Err(err.clone()));
                        });
                    }
                });
            }
            ModelType::Embedding(_) => {
                let input_lengths: Vec<usize> = (0..batch.1.len())
                    .map(|i| {
                        (batch.1.cumulative_seq_lengths[i + 1] - batch.1.cumulative_seq_lengths[i])
                            as usize
                    })
                    .collect();

                let results = backend.embed(batch.1).await;

                // Handle sending responses in another thread to avoid starving the backend
                std::thread::spawn(move || match results {
                    Ok((embeddings, inference_duration)) => {
                        // Required to correctly index in the embeddings
                        let mut pooled_index = 0;
                        let mut raw_cumulative_length = 0;

                        for (i, m) in batch.0.into_iter().enumerate() {
                            let infer_metadata = InferMetadata {
                                prompt_tokens: m.prompt_tokens,
                                tokenization: m.tokenization,
                                queue: m.queue_time.elapsed() - inference_duration,
                                inference: inference_duration,
                            };

                            let results = if m.pooling {
                                let results = embeddings.pooled_embeddings[pooled_index].clone();
                                pooled_index += 1;
                                InferResult::PooledEmbedding(PooledEmbeddingsInferResponse {
                                    results,
                                    metadata: infer_metadata,
                                })
                            } else {
                                let length = input_lengths[i];
                                let results = embeddings.raw_embeddings
                                    [raw_cumulative_length..raw_cumulative_length + length]
                                    .to_vec();
                                raw_cumulative_length += length;
                                InferResult::RawEmbedding(RawEmbeddingsInferResponse {
                                    results,
                                    metadata: infer_metadata,
                                })
                            };

                            let _ = m.response_tx.send(Ok(results));
                        }
                    }
                    Err(err) => {
                        batch.0.into_iter().for_each(|m| {
                            let _ = m.response_tx.send(Err(err.clone()));
                        });
                    }
                });
            }
        };
    }
}

#[derive(Debug)]
pub struct InferMetadata {
    pub prompt_tokens: usize,
    pub tokenization: Duration,
    pub queue: Duration,
    pub inference: Duration,
}

#[derive(Debug)]
pub(crate) enum InferResult {
    Classification(ClassificationInferResponse),
    PooledEmbedding(PooledEmbeddingsInferResponse),
    RawEmbedding(RawEmbeddingsInferResponse),
}

#[derive(Debug)]
pub struct ClassificationInferResponse {
    pub results: Vec<f32>,
    pub metadata: InferMetadata,
}

#[derive(Debug)]
pub struct PooledEmbeddingsInferResponse {
    pub results: Vec<f32>,
    pub metadata: InferMetadata,
}

#[derive(Debug)]
pub struct RawEmbeddingsInferResponse {
    pub results: Vec<Vec<f32>>,
    pub metadata: InferMetadata,
}
