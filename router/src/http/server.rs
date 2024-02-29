/// HTTP Server logic
use crate::http::types::{
    EmbedAllRequest, EmbedAllResponse, EmbedRequest, EmbedResponse, Input, OpenAICompatEmbedding,
    OpenAICompatErrorResponse, OpenAICompatRequest, OpenAICompatResponse, OpenAICompatUsage,
    PredictInput, PredictRequest, PredictResponse, Prediction, Rank, RerankRequest, RerankResponse,
    Sequence, SimpleToken, TokenizeRequest, TokenizeResponse,
};
use crate::{
    shutdown, ClassifierModel, EmbeddingModel, ErrorResponse, ErrorType, Info, ModelType,
    ResponseMetadata,
};
use ::http::HeaderMap;
use anyhow::Context;
use axum::extract::Extension;
use axum::http::HeaderValue;
use axum::http::{Method, StatusCode};
use axum::routing::{get, post};
use axum::{http, Json, Router};
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;
use futures::future::join_all;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use text_embeddings_backend::BackendError;
use text_embeddings_core::infer::{
    AllEmbeddingsInferResponse, Infer, PooledEmbeddingsInferResponse,
};
use text_embeddings_core::TextEmbeddingsError;
use tokio::sync::OwnedSemaphorePermit;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::instrument;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

///Text Embeddings Inference endpoint info
#[utoipa::path(
get,
tag = "Text Embeddings Inference",
path = "/info",
responses((status = 200, description = "Served model info", body = Info))
)]
#[instrument]
async fn get_model_info(info: Extension<Info>) -> Json<Info> {
    Json(info.0)
}

#[utoipa::path(
get,
tag = "Text Embeddings Inference",
path = "/health",
responses(
(status = 200, description = "Everything is working fine"),
(status = 503, description = "Text embeddings Inference is down", body = ErrorResponse,
example = json ! ({"error": "unhealthy", "error_type": "unhealthy"})),
)
)]
#[instrument(skip(infer))]
/// Health check method
async fn health(infer: Extension<Infer>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    match infer.health().await {
        true => Ok(()),
        false => Err(ErrorResponse {
            error: "unhealthy".to_string(),
            error_type: ErrorType::Unhealthy,
        })?,
    }
}

/// Get Predictions. Returns a 424 status code if the model is not a Sequence Classification model
#[utoipa::path(
post,
tag = "Text Embeddings Inference",
path = "/predict",
request_body = PredictRequest,
responses(
(status = 200, description = "Predictions", body = PredictResponse),
(status = 424, description = "Prediction Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn predict(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<PredictRequest>,
) -> Result<(HeaderMap, Json<PredictResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();

    // Closure for predict
    let predict_inner = move |inputs: Sequence,
                              truncate: bool,
                              raw_scores: bool,
                              infer: Infer,
                              info: Info,
                              permit: Option<OwnedSemaphorePermit>| async move {
        let permit = match permit {
            None => infer.acquire_permit().await,
            Some(permit) => permit,
        };

        let response = infer
            .predict(inputs, truncate, raw_scores, permit)
            .await
            .map_err(ErrorResponse::from)?;

        let id2label = match &info.model_type {
            ModelType::Classifier(classifier) => &classifier.id2label,
            ModelType::Reranker(classifier) => &classifier.id2label,
            _ => panic!(),
        };

        let mut predictions = Vec::with_capacity(response.results.len());
        for (i, s) in response.results.into_iter().enumerate() {
            // Check that s is not NaN or the partial_cmp below will panic
            if s.is_nan() {
                return Err(ErrorResponse {
                    error: "score is NaN".to_string(),
                    error_type: ErrorType::Backend,
                });
            }
            // Map score to label
            predictions.push(Prediction {
                score: s,
                label: id2label.get(&i.to_string()).unwrap().clone(),
            });
        }
        // Reverse sort
        predictions.sort_by(|x, y| x.score.partial_cmp(&y.score).unwrap());
        predictions.reverse();

        Ok::<(usize, Duration, Duration, Duration, Vec<Prediction>), ErrorResponse>((
            response.metadata.prompt_tokens,
            response.metadata.tokenization,
            response.metadata.queue,
            response.metadata.inference,
            predictions,
        ))
    };

    let (response, metadata) = match req.inputs {
        PredictInput::Single(inputs) => {
            metrics::increment_counter!("te_request_count", "method" => "single");

            let compute_chars = inputs.count_chars();
            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let (prompt_tokens, tokenization, queue, inference, predictions) = predict_inner(
                inputs,
                req.truncate,
                req.raw_scores,
                infer.0,
                info.0,
                Some(permit),
            )
            .await?;

            metrics::increment_counter!("te_request_success", "method" => "single");

            (
                PredictResponse::Single(predictions),
                ResponseMetadata::new(
                    compute_chars,
                    prompt_tokens,
                    start_time,
                    tokenization,
                    queue,
                    inference,
                ),
            )
        }
        PredictInput::Batch(inputs) => {
            metrics::increment_counter!("te_request_count", "method" => "batch");

            let batch_size = inputs.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                metrics::increment_counter!("te_request_failure", "err" => "batch_size");
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.count_chars();
                let local_infer = infer.clone();
                let local_info = info.clone();
                futures.push(predict_inner(
                    input,
                    req.truncate,
                    req.raw_scores,
                    local_infer.0,
                    local_info.0,
                    None,
                ))
            }
            let results = join_all(futures).await.into_iter().collect::<Result<
                Vec<(usize, Duration, Duration, Duration, Vec<Prediction>)>,
                ErrorResponse,
            >>()?;

            let mut predictions = Vec::with_capacity(batch_size);
            let mut total_tokenization_time = 0;
            let mut total_queue_time = 0;
            let mut total_inference_time = 0;
            let mut total_compute_tokens = 0;

            for r in results {
                total_compute_tokens += r.0;
                total_tokenization_time += r.1.as_nanos() as u64;
                total_queue_time += r.2.as_nanos() as u64;
                total_inference_time += r.3.as_nanos() as u64;
                predictions.push(r.4);
            }
            let batch_size = batch_size as u64;

            metrics::increment_counter!("te_request_success", "method" => "batch");

            (
                PredictResponse::Batch(predictions),
                ResponseMetadata::new(
                    compute_chars,
                    total_compute_tokens,
                    start_time,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                ),
            )
        }
    };

    metadata.record_span(&span);
    metadata.record_metrics();

    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    Ok((headers, Json(response)))
}

/// Get Ranks. Returns a 424 status code if the model is not a Sequence Classification model with
/// a single class.
#[utoipa::path(
post,
tag = "Text Embeddings Inference",
path = "/rerank",
request_body = RerankRequest,
responses(
(status = 200, description = "Ranks", body = RerankResponse),
(status = 424, description = "Rerank Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn rerank(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<RerankRequest>,
) -> Result<(HeaderMap, Json<RerankResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();

    if req.texts.is_empty() {
        let message = "`texts` cannot be empty".to_string();
        tracing::error!("{message}");
        let err = ErrorResponse {
            error: message,
            error_type: ErrorType::Validation,
        };
        metrics::increment_counter!("te_request_failure", "err" => "validation");
        Err(err)?;
    }

    match &info.model_type {
        ModelType::Reranker(_) => Ok(()),
        ModelType::Classifier(_) | ModelType::Embedding(_) => {
            metrics::increment_counter!("te_request_failure", "err" => "model_type");
            let message = "model is not a re-ranker model".to_string();
            Err(TextEmbeddingsError::Backend(BackendError::Inference(
                message,
            )))
        }
    }
    .map_err(|err| {
        tracing::error!("{err}");
        ErrorResponse::from(err)
    })?;

    // Closure for rerank
    let rerank_inner = move |query: String,
                             text: String,
                             truncate: bool,
                             raw_scores: bool,
                             infer: Infer| async move {
        let permit = infer.acquire_permit().await;

        let response = infer
            .predict((query, text), truncate, raw_scores, permit)
            .await
            .map_err(ErrorResponse::from)?;

        let score = response.results[0];

        Ok::<(usize, Duration, Duration, Duration, f32), ErrorResponse>((
            response.metadata.prompt_tokens,
            response.metadata.tokenization,
            response.metadata.queue,
            response.metadata.inference,
            score,
        ))
    };

    let (response, metadata) = {
        metrics::increment_counter!("te_request_count", "method" => "batch");

        let batch_size = req.texts.len();
        if batch_size > info.max_client_batch_size {
            let message = format!(
                "batch size {batch_size} > maximum allowed batch size {}",
                info.max_client_batch_size
            );
            tracing::error!("{message}");
            let err = ErrorResponse {
                error: message,
                error_type: ErrorType::Validation,
            };
            metrics::increment_counter!("te_request_failure", "err" => "batch_size");
            Err(err)?;
        }

        let mut futures = Vec::with_capacity(batch_size);
        let query_chars = req.query.chars().count();
        let mut compute_chars = query_chars * batch_size;

        for text in &req.texts {
            compute_chars += text.chars().count();
            let local_infer = infer.clone();
            futures.push(rerank_inner(
                req.query.clone(),
                text.clone(),
                req.truncate,
                req.raw_scores,
                local_infer.0,
            ))
        }
        let results = join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<(usize, Duration, Duration, Duration, f32)>, ErrorResponse>>()?;

        let mut ranks = Vec::with_capacity(batch_size);
        let mut total_tokenization_time = 0;
        let mut total_queue_time = 0;
        let mut total_inference_time = 0;
        let mut total_compute_tokens = 0;

        for (index, r) in results.into_iter().enumerate() {
            total_compute_tokens += r.0;
            total_tokenization_time += r.1.as_nanos() as u64;
            total_queue_time += r.2.as_nanos() as u64;
            total_inference_time += r.3.as_nanos() as u64;
            let text = if req.return_text {
                Some(req.texts[index].clone())
            } else {
                None
            };

            let score = r.4;
            // Check that s is not NaN or the partial_cmp below will panic
            if score.is_nan() {
                Err(ErrorResponse {
                    error: "score is NaN".to_string(),
                    error_type: ErrorType::Backend,
                })?;
            }

            ranks.push(Rank { index, text, score })
        }

        // Reverse sort
        ranks.sort_by(|x, y| x.score.partial_cmp(&y.score).unwrap());
        ranks.reverse();

        let batch_size = batch_size as u64;

        metrics::increment_counter!("te_request_success", "method" => "batch");

        (
            RerankResponse(ranks),
            ResponseMetadata::new(
                compute_chars,
                total_compute_tokens,
                start_time,
                Duration::from_nanos(total_tokenization_time / batch_size),
                Duration::from_nanos(total_queue_time / batch_size),
                Duration::from_nanos(total_inference_time / batch_size),
            ),
        )
    };

    metadata.record_span(&span);
    metadata.record_metrics();

    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    Ok((headers, Json(response)))
}

/// Get Embeddings. Returns a 424 status code if the model is not an embedding model.
#[utoipa::path(
post,
tag = "Text Embeddings Inference",
path = "/embed",
request_body = EmbedRequest,
responses(
(status = 200, description = "Embeddings", body = EmbedResponse),
(status = 424, description = "Embedding Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn embed(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<EmbedRequest>,
) -> Result<(HeaderMap, Json<EmbedResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();

    let (response, metadata) = match req.inputs {
        Input::Single(input) => {
            metrics::increment_counter!("te_request_count", "method" => "single");

            let compute_chars = input.chars().count();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_pooled(input, req.truncate, req.normalize, permit)
                .await
                .map_err(ErrorResponse::from)?;

            metrics::increment_counter!("te_request_success", "method" => "single");

            (
                EmbedResponse(vec![response.results]),
                ResponseMetadata::new(
                    compute_chars,
                    response.metadata.prompt_tokens,
                    start_time,
                    response.metadata.tokenization,
                    response.metadata.queue,
                    response.metadata.inference,
                ),
            )
        }
        Input::Batch(inputs) => {
            metrics::increment_counter!("te_request_count", "method" => "batch");

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                metrics::increment_counter!("te_request_failure", "err" => "validation");
                Err(err)?;
            }

            let batch_size = inputs.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                metrics::increment_counter!("te_request_failure", "err" => "batch_size");
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.chars().count();

                let local_infer = infer.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    local_infer
                        .embed_pooled(input, req.truncate, req.normalize, permit)
                        .await
                })
            }
            let results = join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<PooledEmbeddingsInferResponse>, TextEmbeddingsError>>()
                .map_err(ErrorResponse::from)?;

            let mut embeddings = Vec::with_capacity(batch_size);
            let mut total_tokenization_time = 0;
            let mut total_queue_time = 0;
            let mut total_inference_time = 0;
            let mut total_compute_tokens = 0;

            for r in results {
                total_tokenization_time += r.metadata.tokenization.as_nanos() as u64;
                total_queue_time += r.metadata.queue.as_nanos() as u64;
                total_inference_time += r.metadata.inference.as_nanos() as u64;
                total_compute_tokens += r.metadata.prompt_tokens;
                embeddings.push(r.results);
            }
            let batch_size = batch_size as u64;

            metrics::increment_counter!("te_request_success", "method" => "batch");

            (
                EmbedResponse(embeddings),
                ResponseMetadata::new(
                    compute_chars,
                    total_compute_tokens,
                    start_time,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                ),
            )
        }
    };

    metadata.record_span(&span);
    metadata.record_metrics();

    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    Ok((headers, Json(response)))
}

/// Get all Embeddings without Pooling.
/// Returns a 424 status code if the model is not an embedding model.
#[utoipa::path(
post,
tag = "Text Embeddings Inference",
path = "/embed_all",
request_body = EmbedAllRequest,
responses(
(status = 200, description = "Embeddings", body = EmbedAllResponse),
(status = 424, description = "Embedding Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn embed_all(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<EmbedAllRequest>,
) -> Result<(HeaderMap, Json<EmbedAllResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();

    let (response, metadata) = match req.inputs {
        Input::Single(input) => {
            metrics::increment_counter!("te_request_count", "method" => "single");

            let compute_chars = input.chars().count();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_all(input, req.truncate, permit)
                .await
                .map_err(ErrorResponse::from)?;

            metrics::increment_counter!("te_request_success", "method" => "single");

            (
                EmbedAllResponse(vec![response.results]),
                ResponseMetadata::new(
                    compute_chars,
                    response.metadata.prompt_tokens,
                    start_time,
                    response.metadata.tokenization,
                    response.metadata.queue,
                    response.metadata.inference,
                ),
            )
        }
        Input::Batch(inputs) => {
            metrics::increment_counter!("te_request_count", "method" => "batch");

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                metrics::increment_counter!("te_request_failure", "err" => "validation");
                Err(err)?;
            }

            let batch_size = inputs.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                metrics::increment_counter!("te_request_failure", "err" => "batch_size");
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.chars().count();

                let local_infer = infer.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    local_infer.embed_all(input, req.truncate, permit).await
                })
            }
            let results = join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<AllEmbeddingsInferResponse>, TextEmbeddingsError>>()
                .map_err(ErrorResponse::from)?;

            let mut embeddings = Vec::with_capacity(batch_size);
            let mut total_tokenization_time = 0;
            let mut total_queue_time = 0;
            let mut total_inference_time = 0;
            let mut total_compute_tokens = 0;

            for r in results {
                total_tokenization_time += r.metadata.tokenization.as_nanos() as u64;
                total_queue_time += r.metadata.queue.as_nanos() as u64;
                total_inference_time += r.metadata.inference.as_nanos() as u64;
                total_compute_tokens += r.metadata.prompt_tokens;
                embeddings.push(r.results);
            }
            let batch_size = batch_size as u64;

            metrics::increment_counter!("te_request_success", "method" => "batch");

            (
                EmbedAllResponse(embeddings),
                ResponseMetadata::new(
                    compute_chars,
                    total_compute_tokens,
                    start_time,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                ),
            )
        }
    };

    metadata.record_span(&span);
    metadata.record_metrics();

    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    Ok((headers, Json(response)))
}

/// OpenAI compatible route. Returns a 424 status code if the model is not an embedding model.
#[utoipa::path(
post,
tag = "Text Embeddings Inference",
path = "/embeddings",
request_body = OpenAICompatRequest,
responses(
(status = 200, description = "Embeddings", body = OpenAICompatResponse),
(status = 424, description = "Embedding Error", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Inference failed", "type": "backend"})),
(status = 429, description = "Model is overloaded", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Model is overloaded", "type": "overloaded"})),
(status = 422, description = "Tokenization error", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Tokenization error", "type": "tokenizer"})),
(status = 413, description = "Batch size error", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Batch size error", "type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn openai_embed(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<OpenAICompatRequest>,
) -> Result<(HeaderMap, Json<OpenAICompatResponse>), (StatusCode, Json<OpenAICompatErrorResponse>)>
{
    let span = tracing::Span::current();
    let start_time = Instant::now();

    let (embeddings, metadata) = match req.input {
        Input::Single(input) => {
            metrics::increment_counter!("te_request_count", "method" => "single");

            let compute_chars = input.chars().count();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_pooled(input, false, true, permit)
                .await
                .map_err(ErrorResponse::from)?;

            metrics::increment_counter!("te_request_success", "method" => "single");

            (
                vec![OpenAICompatEmbedding {
                    object: "embedding",
                    embedding: response.results,
                    index: 0,
                }],
                ResponseMetadata::new(
                    compute_chars,
                    response.metadata.prompt_tokens,
                    start_time,
                    response.metadata.tokenization,
                    response.metadata.queue,
                    response.metadata.inference,
                ),
            )
        }
        Input::Batch(inputs) => {
            metrics::increment_counter!("te_request_count", "method" => "batch");

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                metrics::increment_counter!("te_request_failure", "err" => "validation");
                Err(err)?;
            }

            let batch_size = inputs.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                metrics::increment_counter!("te_request_failure", "err" => "batch_size");
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.chars().count();

                let local_infer = infer.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    local_infer.embed_pooled(input, false, true, permit).await
                })
            }
            let results = join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<PooledEmbeddingsInferResponse>, TextEmbeddingsError>>()
                .map_err(ErrorResponse::from)?;

            let mut embeddings = Vec::with_capacity(batch_size);
            let mut total_tokenization_time = 0;
            let mut total_queue_time = 0;
            let mut total_inference_time = 0;
            let mut total_compute_tokens = 0;

            for (i, r) in results.into_iter().enumerate() {
                total_tokenization_time += r.metadata.tokenization.as_nanos() as u64;
                total_queue_time += r.metadata.queue.as_nanos() as u64;
                total_inference_time += r.metadata.inference.as_nanos() as u64;
                total_compute_tokens += r.metadata.prompt_tokens;
                embeddings.push(OpenAICompatEmbedding {
                    object: "embedding",
                    embedding: r.results,
                    index: i,
                });
            }
            let batch_size = batch_size as u64;

            metrics::increment_counter!("te_request_success", "method" => "batch");

            (
                embeddings,
                ResponseMetadata::new(
                    compute_chars,
                    total_compute_tokens,
                    start_time,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                ),
            )
        }
    };

    metadata.record_span(&span);
    metadata.record_metrics();

    let compute_tokens = metadata.compute_tokens;
    let headers = HeaderMap::from(metadata);

    tracing::info!("Success");

    let response = OpenAICompatResponse {
        object: "list",
        data: embeddings,
        model: info.model_id.clone(),
        usage: OpenAICompatUsage {
            prompt_tokens: compute_tokens,
            total_tokens: compute_tokens,
        },
    };
    Ok((headers, Json(response)))
}

/// Tokenize inputs
#[utoipa::path(
post,
tag = "Text Embeddings Inference",
path = "/tokenize",
request_body = TokenizeRequest,
responses(
(status = 200, description = "Tokenized ids", body = TokenizeResponse),
(status = 422, description = "Tokenization error", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Tokenization error", "type": "tokenizer"})),
)
)]
#[instrument(skip_all)]
async fn tokenize(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let tokenize_inner = move |input: String, add_special_tokens: bool, infer: Infer| async move {
        let encoding = infer
            .tokenize(input.clone(), add_special_tokens)
            .await
            .map_err(ErrorResponse::from)?;
        let tokens: Vec<SimpleToken> = encoding
            .get_ids()
            .iter()
            .zip(encoding.get_offsets())
            .zip(encoding.get_special_tokens_mask())
            .zip(encoding.get_tokens())
            .map(|(((&id, &(start, stop)), special), token)| {
                let special = *special == 1;
                match special {
                    true => SimpleToken {
                        id,
                        text: token.clone(),
                        special,
                        start: None,
                        stop: None,
                    },
                    false => {
                        let text: String = input.chars().skip(start).take(stop - start).collect();
                        SimpleToken {
                            id,
                            text,
                            special,
                            start: Some(start),
                            stop: Some(stop),
                        }
                    }
                }
            })
            .collect();
        Ok::<Vec<SimpleToken>, ErrorResponse>(tokens)
    };

    let tokens = match req.inputs {
        Input::Single(input) => vec![tokenize_inner(input, req.add_special_tokens, infer.0).await?],
        Input::Batch(inputs) => {
            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                metrics::increment_counter!("te_request_failure", "err" => "validation");
                Err(err)?;
            }

            let batch_size = inputs.len();
            if batch_size > info.max_client_batch_size {
                let message = format!(
                    "batch size {batch_size} > maximum allowed batch size {}",
                    info.max_client_batch_size
                );
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Validation,
                };
                metrics::increment_counter!("te_request_failure", "err" => "batch_size");
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            for input in inputs {
                futures.push(tokenize_inner(
                    input,
                    req.add_special_tokens,
                    infer.0.clone(),
                ));
            }

            join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<Vec<SimpleToken>>, ErrorResponse>>()?
        }
    };
    Ok(Json(TokenizeResponse(tokens)))
}

/// Prometheus metrics scrape endpoint
#[utoipa::path(
get,
tag = "Text Embeddings Inference",
path = "/metrics",
responses((status = 200, description = "Prometheus Metrics", body = String))
)]
async fn metrics(prom_handle: Extension<PrometheusHandle>) -> String {
    prom_handle.render()
}

/// Serving method
pub async fn run(
    infer: Infer,
    info: Info,
    addr: SocketAddr,
    prom_builder: PrometheusBuilder,
    cors_allow_origin: Option<Vec<String>>,
) -> Result<(), anyhow::Error> {
    // OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
    paths(
    get_model_info,
    health,
    predict,
    rerank,
    embed,
    embed_all,
    openai_embed,
    tokenize,
    metrics,
    ),
    components(
    schemas(
    PredictInput,
    Input,
    Info,
    ModelType,
    ClassifierModel,
    EmbeddingModel,
    PredictRequest,
    Prediction,
    PredictResponse,
    OpenAICompatRequest,
    OpenAICompatEmbedding,
    OpenAICompatUsage,
    OpenAICompatResponse,
    EmbedAllRequest,
    EmbedAllResponse,
    RerankRequest,
    Rank,
    RerankResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    OpenAICompatErrorResponse,
    TokenizeRequest,
    TokenizeResponse,
    SimpleToken,
    ErrorType,
    )
    ),
    tags(
    (name = "Text Embeddings Inference", description = "Hugging Face Text Embeddings Inference API")
    ),
    info(
    title = "Text Embeddings Inference",
    license(
    name = "HFOIL",
    )
    )
    )]
    struct ApiDoc;

    // CORS allowed origins
    // map to go inside the option and then map to parse from String to HeaderValue
    // Finally, convert to AllowOrigin
    let allow_origin: Option<AllowOrigin> = cors_allow_origin.map(|cors_allow_origin| {
        AllowOrigin::list(
            cors_allow_origin
                .into_iter()
                .map(|origin| origin.parse::<HeaderValue>().unwrap()),
        )
    });

    let prom_handle = prom_builder
        .install_recorder()
        .context("failed to install metrics recorder")?;

    // CORS layer
    let allow_origin = allow_origin.unwrap_or(AllowOrigin::any());
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    // Create router
    let app = Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()))
        // Base routes
        .route("/info", get(get_model_info))
        .route("/embed", post(embed))
        .route("/embed_all", post(embed_all))
        .route("/predict", post(predict))
        .route("/rerank", post(rerank))
        .route("/tokenize", post(tokenize))
        // OpenAI compat route
        .route("/embeddings", post(openai_embed))
        // Base Health route
        .route("/health", get(health))
        // Inference API health route
        .route("/", get(health))
        // AWS Sagemaker health route
        .route("/ping", get(health))
        // Prometheus metrics route
        .route("/metrics", get(metrics));

    // Set default routes
    let app = match &info.model_type {
        ModelType::Classifier(_) => {
            app.route("/", post(predict))
                // AWS Sagemaker route
                .route("/invocations", post(predict))
        }
        ModelType::Reranker(_) => {
            app.route("/", post(rerank))
                // AWS Sagemaker route
                .route("/invocations", post(rerank))
        }
        ModelType::Embedding(_) => {
            app.route("/", post(embed))
                // AWS Sagemaker route
                .route("/invocations", post(embed))
        }
    };

    let app = app
        .layer(Extension(infer))
        .layer(Extension(info))
        .layer(Extension(prom_handle.clone()))
        .layer(OtelAxumLayer::default())
        .layer(cors_layer);

    // Run server
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

    axum::serve(listener, app)
        // Wait until all requests are finished to shut down
        .with_graceful_shutdown(shutdown::shutdown_signal())
        .await?;

    Ok(())
}

impl From<&ErrorType> for StatusCode {
    fn from(value: &ErrorType) -> Self {
        match value {
            ErrorType::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
            ErrorType::Backend => StatusCode::FAILED_DEPENDENCY,
            ErrorType::Overloaded => StatusCode::TOO_MANY_REQUESTS,
            ErrorType::Tokenizer => StatusCode::UNPROCESSABLE_ENTITY,
            ErrorType::Validation => StatusCode::PAYLOAD_TOO_LARGE,
        }
    }
}

impl From<ErrorResponse> for OpenAICompatErrorResponse {
    fn from(value: ErrorResponse) -> Self {
        OpenAICompatErrorResponse {
            message: value.error,
            code: StatusCode::from(&value.error_type).as_u16(),
            error_type: value.error_type,
        }
    }
}

/// Convert to Axum supported formats
impl From<ErrorResponse> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: ErrorResponse) -> Self {
        (StatusCode::from(&err.error_type), Json(err))
    }
}

impl From<ErrorResponse> for (StatusCode, Json<OpenAICompatErrorResponse>) {
    fn from(err: ErrorResponse) -> Self {
        (StatusCode::from(&err.error_type), Json(err.into()))
    }
}
