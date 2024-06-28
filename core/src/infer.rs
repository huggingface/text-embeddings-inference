use crate::queue::{Entry, Metadata, NextBatch, Queue};
use crate::tokenization::{EncodingInput, RawEncoding, Tokenization};
use crate::TextEmbeddingsError;
use std::sync::Arc;
use std::time::{Duration, Instant};
use text_embeddings_backend::{Backend, BackendError, Embedding, ModelType};
use tokenizers::TruncationDirection;
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

        // Bound channel to 1 to be able to prefetch one batch
        let (embed_sender, embed_receiver) = mpsc::channel(1);

        // Batching task
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

    #[instrument(skip(self, inputs))]
    pub async fn tokenize<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        add_special_tokens: bool,
        prompt_name: Option<String>,
    ) -> Result<(Option<String>, RawEncoding), TextEmbeddingsError> {
        self.tokenization
            .tokenize(inputs.into(), add_special_tokens, prompt_name)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "tokenization");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })
    }

    #[instrument(skip(self, ids))]
    pub async fn decode(
        &self,
        ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> Result<String, TextEmbeddingsError> {
        self.tokenization
            .decode(ids, skip_special_tokens)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "tokenization");
                counter.increment(1);
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
                let counter = metrics::counter!("te_request_failure", "err" => "overloaded");
                counter.increment(1);
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

    #[instrument(skip(self, inputs, permit))]
    pub async fn embed_all<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        permit: OwnedSemaphorePermit,
    ) -> Result<AllEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        if self.is_splade() {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
            let message = "`embed_all` is not available for SPLADE models".to_string();
            tracing::error!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let results = self
            .embed(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                false,
                &start_time,
                permit,
            )
            .await?;

        let InferResult::AllEmbedding(response) = results else {
            panic!("unexpected enum variant")
        };

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        let counter = metrics::counter!("te_embed_success");
        counter.increment(1);
        let histogram = metrics::histogram!("te_embed_duration");
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_tokenization_duration");
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_queue_duration");
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_inference_duration");
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[instrument(skip(self, inputs, permit))]
    pub async fn embed_sparse<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        permit: OwnedSemaphorePermit,
    ) -> Result<PooledEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        if !self.is_splade() {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
            let message = "Model is not an embedding model with SPLADE pooling".to_string();
            tracing::error!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let results = self
            .embed(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                true,
                &start_time,
                permit,
            )
            .await?;

        let InferResult::PooledEmbedding(response) = results else {
            panic!("unexpected enum variant")
        };

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        let counter = metrics::counter!("te_embed_success");
        counter.increment(1);
        let histogram = metrics::histogram!("te_embed_duration");
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_tokenization_duration");
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_queue_duration");
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_inference_duration");
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[instrument(skip(self, inputs, permit))]
    pub async fn embed_pooled<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        normalize: bool,
        permit: OwnedSemaphorePermit,
    ) -> Result<PooledEmbeddingsInferResponse, TextEmbeddingsError> {
        let start_time = Instant::now();

        if self.is_splade() && normalize {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
            let message = "`normalize` is not available for SPLADE models".to_string();
            tracing::error!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let results = self
            .embed(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                true,
                &start_time,
                permit,
            )
            .await?;

        let InferResult::PooledEmbedding(mut response) = results else {
            panic!("unexpected enum variant")
        };

        if normalize {
            // Normalize embedding
            let scale = (1.0
                / response
                    .results
                    .iter()
                    .map(|v| {
                        let v = *v as f64;
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt()) as f32;
            for v in response.results.iter_mut() {
                *v *= scale;
            }
        }

        // Timings
        let total_time = start_time.elapsed();

        // Metrics
        let counter = metrics::counter!("te_embed_success");
        counter.increment(1);
        let histogram = metrics::histogram!("te_embed_duration");
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_tokenization_duration");
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_queue_duration");
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("te_embed_inference_duration");
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[allow(clippy::too_many_arguments)]
    async fn embed<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
        pooling: bool,
        start_time: &Instant,
        _permit: OwnedSemaphorePermit,
    ) -> Result<InferResult, TextEmbeddingsError> {
        if self.is_classifier() {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
            let message = "Model is not an embedding model".to_string();
            tracing::error!("{message}");
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let counter = metrics::counter!("te_embed_count");
        counter.increment(1);

        // Tokenization
        let encoding = self
            .tokenization
            .encode(inputs.into(), truncate, truncation_direction, prompt_name)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "tokenization");
                counter.increment(1);
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
                let counter = metrics::counter!("te_request_failure", "err" => "inference");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })?;

        Ok(response)
    }

    #[instrument(skip(self, inputs, _permit))]
    pub async fn predict<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: TruncationDirection,
        raw_scores: bool,
        _permit: OwnedSemaphorePermit,
    ) -> Result<ClassificationInferResponse, TextEmbeddingsError> {
        if !self.is_classifier() {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
            let message = "Model is not a classifier model".to_string();
            return Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )));
        }

        let start_time = Instant::now();
        let counter = metrics::counter!("te_predict_count");
        counter.increment(1);

        // Tokenization
        let encoding = self
            .tokenization
            .encode(inputs.into(), truncate, truncation_direction, None)
            .await
            .map_err(|err| {
                let counter = metrics::counter!("te_request_failure", "err" => "tokenization");
                counter.increment(1);
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
                let counter = metrics::counter!("te_request_failure", "err" => "inference");
                counter.increment(1);
                tracing::error!("{err}");
                err
            })?;

        let InferResult::Classification(mut response) = response else {
            panic!("unexpected enum variant")
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
        let counter = metrics::counter!("te_predict_success");
        counter.increment(1);
        let histogram = metrics::histogram!("te_predict_duration");
        histogram.record(total_time.as_secs_f64());
        let histogram = metrics::histogram!("te_predict_tokenization_duration");
        histogram.record(response.metadata.tokenization.as_secs_f64());
        let histogram = metrics::histogram!("te_predict_queue_duration");
        histogram.record(response.metadata.queue.as_secs_f64());
        let histogram = metrics::histogram!("te_predict_inference_duration");
        histogram.record(response.metadata.inference.as_secs_f64());

        Ok(response)
    }

    #[instrument(skip(self))]
    pub fn is_classifier(&self) -> bool {
        matches!(self.backend.model_type, ModelType::Classifier)
    }

    #[instrument(skip(self))]
    pub fn is_splade(&self) -> bool {
        matches!(
            self.backend.model_type,
            ModelType::Embedding(text_embeddings_backend::Pool::Splade)
        )
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
async fn batching_task(queue: Queue, notify: Arc<Notify>, embed_sender: mpsc::Sender<NextBatch>) {
    loop {
        notify.notified().await;

        {
            let mut permit = embed_sender
                .reserve()
                .await
                .expect("embed receiver was dropped. This is a bug.");

            while let Some(next_batch) = queue.next_batch().await {
                permit.send(next_batch);
                permit = embed_sender
                    .reserve()
                    .await
                    .expect("embed receiver was dropped. This is a bug.");
            }
        }
    }
}

#[instrument(skip_all)]
async fn backend_task(backend: Backend, mut embed_receiver: mpsc::Receiver<NextBatch>) {
    while let Some(batch) = embed_receiver.recv().await {
        match &backend.model_type {
            ModelType::Classifier => {
                let results = backend.predict(batch.1).await;

                // Handle sending responses in another thread to avoid starving the backend
                std::thread::spawn(move || match results {
                    Ok((mut predictions, inference_duration)) => {
                        batch.0.into_iter().enumerate().for_each(|(i, m)| {
                            let infer_metadata = InferMetadata {
                                prompt_tokens: m.prompt_tokens,
                                tokenization: m.tokenization,
                                queue: m.queue_time.elapsed() - inference_duration,
                                inference: inference_duration,
                            };

                            let _ = m.response_tx.send(Ok(InferResult::Classification(
                                ClassificationInferResponse {
                                    results: predictions.remove(&i).expect(
                                        "prediction not found in results. This is a backend bug.",
                                    ),
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
                let results = backend.embed(batch.1).await;

                // Handle sending responses in another thread to avoid starving the backend
                std::thread::spawn(move || match results {
                    Ok((mut embeddings, inference_duration)) => {
                        batch.0.into_iter().enumerate().for_each(|(i, m)| {
                            let metadata = InferMetadata {
                                prompt_tokens: m.prompt_tokens,
                                tokenization: m.tokenization,
                                queue: m.queue_time.elapsed() - inference_duration,
                                inference: inference_duration,
                            };

                            let results = match embeddings
                                .remove(&i)
                                .expect("embedding not found in results. This is a backend bug.")
                            {
                                Embedding::Pooled(e) => {
                                    InferResult::PooledEmbedding(PooledEmbeddingsInferResponse {
                                        results: e,
                                        metadata,
                                    })
                                }
                                Embedding::All(e) => {
                                    InferResult::AllEmbedding(AllEmbeddingsInferResponse {
                                        results: e,
                                        metadata,
                                    })
                                }
                            };

                            let _ = m.response_tx.send(Ok(results));
                        })
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
    AllEmbedding(AllEmbeddingsInferResponse),
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
pub struct AllEmbeddingsInferResponse {
    pub results: Vec<Vec<f32>>,
    pub metadata: InferMetadata,
}
