/// HTTP Server logic
use crate::http::types::{
    DecodeRequest, DecodeResponse, EmbedAllRequest, EmbedAllResponse, EmbedRequest, EmbedResponse,
    EmbedSparseRequest, EmbedSparseResponse, Embedding, EncodingFormat, Input, InputIds, InputType,
    OpenAICompatEmbedding, OpenAICompatErrorResponse, OpenAICompatRequest, OpenAICompatResponse,
    OpenAICompatUsage, PredictInput, PredictRequest, PredictResponse, Prediction, Rank,
    RerankRequest, RerankResponse, Sequence, SimilarityInput, SimilarityParameters,
    SimilarityRequest, SimilarityResponse, SimpleToken, SparseValue, TokenizeInput,
    TokenizeRequest, TokenizeResponse, TruncationDirection, VertexPrediction, VertexRequest,
    VertexResponse,
};
use crate::{
    shutdown, ClassifierModel, EmbeddingModel, ErrorResponse, ErrorType, Info, ModelType,
    ResponseMetadata,
};
use ::http::HeaderMap;
use anyhow::Context;
use axum::extract::{DefaultBodyLimit, Extension};
use axum::http::HeaderValue;
use axum::http::{Method, StatusCode};
use axum::routing::{get, post};
use axum::{http, Json, Router};
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use futures::future::join_all;
use futures::FutureExt;
use http::header::AUTHORIZATION;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use simsimd::SpatialSimilarity;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use text_embeddings_backend::BackendError;
use text_embeddings_core::infer::{
    AllEmbeddingsInferResponse, Infer, InferMetadata, PooledEmbeddingsInferResponse,
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
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
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
                              infer: Infer,
                              info: Info,
                              permit: Option<OwnedSemaphorePermit>| async move {
        let permit = match permit {
            None => infer.acquire_permit().await,
            Some(permit) => permit,
        };

        let response = infer
            .predict(
                inputs,
                truncate,
                req.truncation_direction.into(),
                req.raw_scores,
                permit,
            )
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

    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = match req.inputs {
        PredictInput::Single(inputs) => {
            let counter = metrics::counter!("te_request_count", "method" => "single");
            counter.increment(1);

            let compute_chars = inputs.count_chars();
            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let (prompt_tokens, tokenization, queue, inference, predictions) =
                predict_inner(inputs, truncate, infer.0, info.0, Some(permit)).await?;

            let counter = metrics::counter!("te_request_count", "method" => "single");
            counter.increment(1);

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
            let counter = metrics::counter!("te_request_count", "method" => "batch");
            counter.increment(1);

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
                let counter = metrics::counter!("te_request_failure", "err" => "batch_size");
                counter.increment(1);
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
                    truncate,
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

            let counter = metrics::counter!("te_request_success", "method" => "batch");
            counter.increment(1);

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
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
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
            error_type: ErrorType::Empty,
        };
        let counter = metrics::counter!("te_request_failure", "err" => "validation");
        counter.increment(1);
        Err(err)?;
    }

    match &info.model_type {
        ModelType::Reranker(_) => Ok(()),
        ModelType::Classifier(_) | ModelType::Embedding(_) => {
            let counter = metrics::counter!("te_request_failure", "err" => "model_type");
            counter.increment(1);
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
    let rerank_inner = move |query: String, text: String, truncate: bool, infer: Infer| async move {
        let permit = infer.acquire_permit().await;

        let response = infer
            .predict(
                (query, text),
                truncate,
                req.truncation_direction.into(),
                req.raw_scores,
                permit,
            )
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

    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = {
        let counter = metrics::counter!("te_request_count", "method" => "batch");
        counter.increment(1);

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
            let counter = metrics::counter!("te_request_failure", "err" => "batch_size");
            counter.increment(1);
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
                truncate,
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

        let counter = metrics::counter!("te_request_success", "method" => "batch");
        counter.increment(1);

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

/// Get Sentence Similarity. Returns a 424 status code if the model is not an embedding model.
#[utoipa::path(
post,
tag = "Text Embeddings Inference",
path = "/similarity",
request_body = SimilarityRequest,
responses(
(status = 200, description = "Sentence Similarity", body = SimilarityResponse),
(status = 424, description = "Embedding Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn similarity(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<SimilarityRequest>,
) -> Result<(HeaderMap, Json<SimilarityResponse>), (StatusCode, Json<ErrorResponse>)> {
    if req.inputs.sentences.is_empty() {
        let message = "`inputs.sentences` cannot be empty".to_string();
        tracing::error!("{message}");
        let err = ErrorResponse {
            error: message,
            error_type: ErrorType::Empty,
        };
        let counter = metrics::counter!("te_request_failure", "err" => "validation");
        counter.increment(1);
        Err(err)?;
    }
    // +1 because of the source sentence
    let batch_size = req.inputs.sentences.len() + 1;
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
        let counter = metrics::counter!("te_request_failure", "err" => "batch_size");
        counter.increment(1);
        Err(err)?;
    }

    // Convert request to embed request
    let mut inputs = Vec::with_capacity(req.inputs.sentences.len() + 1);
    inputs.push(InputType::String(req.inputs.source_sentence));
    for s in req.inputs.sentences {
        inputs.push(InputType::String(s));
    }
    let parameters = req.parameters.unwrap_or_default();
    let embed_req = EmbedRequest {
        inputs: Input::Batch(inputs),
        truncate: parameters.truncate,
        truncation_direction: parameters.truncation_direction,
        prompt_name: parameters.prompt_name,
        normalize: false,
    };

    // Get embeddings
    let (header_map, embed_response) = embed(infer, info, Json(embed_req)).await?;
    let embeddings = embed_response.0 .0;

    // Compute cosine
    let distances = (1..batch_size)
        .map(|i| 1.0 - f32::cosine(&embeddings[0], &embeddings[i]).unwrap() as f32)
        .collect();

    Ok((header_map, Json(SimilarityResponse(distances))))
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
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
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

    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = match req.inputs {
        Input::Single(input) => {
            let counter = metrics::counter!("te_request_count", "method" => "single");
            counter.increment(1);

            let compute_chars = input.count_chars();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_pooled(
                    input,
                    truncate,
                    req.truncation_direction.into(),
                    req.prompt_name,
                    req.normalize,
                    permit,
                )
                .await
                .map_err(ErrorResponse::from)?;

            let counter = metrics::counter!("te_request_count", "method" => "single");
            counter.increment(1);

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
            let counter = metrics::counter!("te_request_count", "method" => "batch");
            counter.increment(1);

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("te_request_failure", "err" => "validation");
                counter.increment(1);
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
                let counter = metrics::counter!("te_request_failure", "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.count_chars();

                let local_infer = infer.clone();
                let prompt_name = req.prompt_name.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    local_infer
                        .embed_pooled(
                            input,
                            truncate,
                            req.truncation_direction.into(),
                            prompt_name,
                            req.normalize,
                            permit,
                        )
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

            let counter = metrics::counter!("te_request_success", "method" => "batch");
            counter.increment(1);

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

/// Get Sparse Embeddings. Returns a 424 status code if the model is not an embedding model with SPLADE pooling.
#[utoipa::path(
post,
tag = "Text Embeddings Inference",
path = "/embed_sparse",
request_body = EmbedSparseRequest,
responses(
(status = 200, description = "Embeddings", body = EmbedSparseResponse),
(status = 424, description = "Embedding Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(
    skip_all,
    fields(total_time, tokenization_time, queue_time, inference_time,)
)]
async fn embed_sparse(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<EmbedSparseRequest>,
) -> Result<(HeaderMap, Json<EmbedSparseResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();

    let sparsify = |values: Vec<f32>| {
        let mut sparse_values = Vec::with_capacity(values.len());
        for (index, value) in values.into_iter().enumerate() {
            if value != 0.0 {
                sparse_values.push(SparseValue { index, value });
            }
        }
        sparse_values
    };
    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = match req.inputs {
        Input::Single(input) => {
            let counter = metrics::counter!("te_request_count", "method" => "single");
            counter.increment(1);

            let compute_chars = input.count_chars();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_sparse(
                    input,
                    truncate,
                    req.truncation_direction.into(),
                    req.prompt_name,
                    permit,
                )
                .await
                .map_err(ErrorResponse::from)?;

            let counter = metrics::counter!("te_request_count", "method" => "single");
            counter.increment(1);

            (
                EmbedSparseResponse(vec![sparsify(response.results)]),
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
            let counter = metrics::counter!("te_request_count", "method" => "batch");
            counter.increment(1);

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("te_request_failure", "err" => "validation");
                counter.increment(1);
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
                let counter = metrics::counter!("te_request_failure", "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.count_chars();

                let local_infer = infer.clone();
                let prompt_name = req.prompt_name.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    let response = local_infer
                        .embed_sparse(
                            input,
                            truncate,
                            req.truncation_direction.into(),
                            prompt_name,
                            permit,
                        )
                        .await?;
                    Ok((sparsify(response.results), response.metadata))
                })
            }
            let results = join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<(Vec<SparseValue>, InferMetadata)>, TextEmbeddingsError>>()
                .map_err(ErrorResponse::from)?;

            let mut embeddings = Vec::with_capacity(batch_size);
            let mut total_tokenization_time = 0;
            let mut total_queue_time = 0;
            let mut total_inference_time = 0;
            let mut total_compute_tokens = 0;

            for r in results {
                total_tokenization_time += r.1.tokenization.as_nanos() as u64;
                total_queue_time += r.1.queue.as_nanos() as u64;
                total_inference_time += r.1.inference.as_nanos() as u64;
                total_compute_tokens += r.1.prompt_tokens;
                embeddings.push(r.0);
            }
            let batch_size = batch_size as u64;

            let counter = metrics::counter!("te_request_success", "method" => "batch");
            counter.increment(1);

            (
                EmbedSparseResponse(embeddings),
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
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
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

    let truncate = req.truncate.unwrap_or(info.auto_truncate);

    let (response, metadata) = match req.inputs {
        Input::Single(input) => {
            let counter = metrics::counter!("te_request_count", "method" => "single");
            counter.increment(1);

            let compute_chars = input.count_chars();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_all(
                    input,
                    truncate,
                    req.truncation_direction.into(),
                    req.prompt_name,
                    permit,
                )
                .await
                .map_err(ErrorResponse::from)?;

            let counter = metrics::counter!("te_request_count", "method" => "single");
            counter.increment(1);

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
            let counter = metrics::counter!("te_request_count", "method" => "batch");
            counter.increment(1);

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("te_request_failure", "err" => "validation");
                counter.increment(1);
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
                let counter = metrics::counter!("te_request_failure", "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.count_chars();

                let local_infer = infer.clone();
                let prompt_name = req.prompt_name.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    local_infer
                        .embed_all(
                            input,
                            truncate,
                            req.truncation_direction.into(),
                            prompt_name,
                            permit,
                        )
                        .await
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

            let counter = metrics::counter!("te_request_success", "method" => "batch");
            counter.increment(1);

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
path = "/v1/embeddings",
request_body = OpenAICompatRequest,
responses(
(status = 200, description = "Embeddings", body = OpenAICompatResponse),
(status = 424, description = "Embedding Error", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Inference failed", "type": "backend"})),
(status = 429, description = "Model is overloaded", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Model is overloaded", "type": "overloaded"})),
(status = 422, description = "Tokenization error", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Tokenization error", "type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = OpenAICompatErrorResponse,
example = json ! ({"message": "Batch is empty", "type": "empty"})),
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
    let encode_embedding = |array: Vec<f32>| {
        match req.encoding_format {
            EncodingFormat::Float => Embedding::Float(array),
            EncodingFormat::Base64 => {
                // Unsafe is fine here since we do not violate memory ownership: bytes
                // is only used in this scope and we return an owned string
                let bytes = unsafe {
                    std::slice::from_raw_parts(array.as_ptr() as *const u8, array.len() * 4)
                };

                Embedding::Base64(BASE64_STANDARD.encode(bytes))
            }
        }
    };

    let span = tracing::Span::current();
    let start_time = Instant::now();

    let truncate = info.auto_truncate;

    let (embeddings, metadata) = match req.input {
        Input::Single(input) => {
            let counter = metrics::counter!("te_request_count", "method" => "single");
            counter.increment(1);

            let compute_chars = input.count_chars();

            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
            let response = infer
                .embed_pooled(
                    input,
                    truncate,
                    tokenizers::TruncationDirection::Right,
                    None,
                    true,
                    permit,
                )
                .await
                .map_err(ErrorResponse::from)?;

            let counter = metrics::counter!("te_request_count", "method" => "single");
            counter.increment(1);

            let embedding = encode_embedding(response.results);
            (
                vec![OpenAICompatEmbedding {
                    object: "embedding",
                    embedding,
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
            let counter = metrics::counter!("te_request_count", "method" => "batch");
            counter.increment(1);

            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("te_request_failure", "err" => "validation");
                counter.increment(1);
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
                let counter = metrics::counter!("te_request_failure", "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            let mut compute_chars = 0;

            for input in inputs {
                compute_chars += input.count_chars();

                let local_infer = infer.clone();
                futures.push(async move {
                    let permit = local_infer.acquire_permit().await;
                    local_infer
                        .embed_pooled(
                            input,
                            truncate,
                            tokenizers::TruncationDirection::Right,
                            None,
                            true,
                            permit,
                        )
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

            for (i, r) in results.into_iter().enumerate() {
                total_tokenization_time += r.metadata.tokenization.as_nanos() as u64;
                total_queue_time += r.metadata.queue.as_nanos() as u64;
                total_inference_time += r.metadata.inference.as_nanos() as u64;
                total_compute_tokens += r.metadata.prompt_tokens;
                let embedding = encode_embedding(r.results);
                embeddings.push(OpenAICompatEmbedding {
                    object: "embedding",
                    embedding,
                    index: i,
                });
            }
            let batch_size = batch_size as u64;

            let counter = metrics::counter!("te_request_success", "method" => "batch");
            counter.increment(1);

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
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
)
)]
#[instrument(skip_all)]
async fn tokenize(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let tokenize_inner = move |input: String,
                               add_special_tokens: bool,
                               prompt_name: Option<String>,
                               infer: Infer| async move {
        let (encoded_input, encoding) = infer
            .tokenize(input.clone(), add_special_tokens, prompt_name)
            .await
            .map_err(ErrorResponse::from)?;
        let input = encoded_input.unwrap_or(input);

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
        TokenizeInput::Single(input) => {
            vec![tokenize_inner(input, req.add_special_tokens, req.prompt_name, infer.0).await?]
        }
        TokenizeInput::Batch(inputs) => {
            if inputs.is_empty() {
                let message = "`inputs` cannot be empty".to_string();
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("te_request_failure", "err" => "validation");
                counter.increment(1);
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
                let counter = metrics::counter!("te_request_failure", "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            for input in inputs {
                futures.push(tokenize_inner(
                    input,
                    req.add_special_tokens,
                    req.prompt_name.clone(),
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

/// Decode input ids
#[utoipa::path(
post,
tag = "Text Embeddings Inference",
path = "/decode",
request_body = DecodeRequest,
responses(
(status = 200, description = "Decoded ids", body = DecodeResponse),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
)
)]
#[instrument(skip_all)]
async fn decode(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<DecodeRequest>,
) -> Result<Json<DecodeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let decode_inner = move |ids: Vec<u32>, skip_special_tokens: bool, infer: Infer| async move {
        let text = infer
            .decode(ids, skip_special_tokens)
            .await
            .map_err(ErrorResponse::from)?;
        Ok::<String, ErrorResponse>(text)
    };

    let texts = match req.ids {
        InputIds::Single(ids) => vec![decode_inner(ids, req.skip_special_tokens, infer.0).await?],
        InputIds::Batch(ids) => {
            if ids.is_empty() {
                let message = "`ids` cannot be empty".to_string();
                tracing::error!("{message}");
                let err = ErrorResponse {
                    error: message,
                    error_type: ErrorType::Empty,
                };
                let counter = metrics::counter!("te_request_failure", "err" => "validation");
                counter.increment(1);
                Err(err)?;
            }

            let batch_size = ids.len();
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
                let counter = metrics::counter!("te_request_failure", "err" => "batch_size");
                counter.increment(1);
                Err(err)?;
            }

            let mut futures = Vec::with_capacity(batch_size);
            for ids in ids {
                futures.push(decode_inner(ids, req.skip_special_tokens, infer.0.clone()));
            }

            join_all(futures)
                .await
                .into_iter()
                .collect::<Result<Vec<String>, ErrorResponse>>()?
        }
    };
    Ok(Json(DecodeResponse(texts)))
}

/// Generate embeddings from a Vertex request
#[utoipa::path(
post,
tag = "Text Embeddings Inference",
path = "/vertex",
request_body = VertexRequest,
responses(
(status = 200, description = "Results"),
(status = 424, description = "Error", body = ErrorResponse,
example = json ! ({"error": "Inference failed", "error_type": "backend"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Tokenization error", body = ErrorResponse,
example = json ! ({"error": "Tokenization error", "error_type": "tokenizer"})),
(status = 400, description = "Batch is empty", body = ErrorResponse,
example = json ! ({"error": "Batch is empty", "error_type": "empty"})),
(status = 413, description = "Batch size error", body = ErrorResponse,
example = json ! ({"error": "Batch size error", "error_type": "validation"})),
)
)]
#[instrument(skip_all)]
async fn vertex_compatibility(
    infer: Extension<Infer>,
    info: Extension<Info>,
    Json(req): Json<VertexRequest>,
) -> Result<Json<VertexResponse>, (StatusCode, Json<ErrorResponse>)> {
    let embed_future = move |infer: Extension<Infer>, info: Extension<Info>, req: EmbedRequest| async move {
        let result = embed(infer, info, Json(req)).await?;
        Ok(VertexPrediction::Embed(result.1 .0))
    };
    let embed_sparse_future =
        move |infer: Extension<Infer>, info: Extension<Info>, req: EmbedSparseRequest| async move {
            let result = embed_sparse(infer, info, Json(req)).await?;
            Ok(VertexPrediction::EmbedSparse(result.1 .0))
        };
    let predict_future =
        move |infer: Extension<Infer>, info: Extension<Info>, req: PredictRequest| async move {
            let result = predict(infer, info, Json(req)).await?;
            Ok(VertexPrediction::Predict(result.1 .0))
        };
    let rerank_future =
        move |infer: Extension<Infer>, info: Extension<Info>, req: RerankRequest| async move {
            let result = rerank(infer, info, Json(req)).await?;
            Ok(VertexPrediction::Rerank(result.1 .0))
        };

    let mut futures = Vec::with_capacity(req.instances.len());
    for instance in req.instances {
        let local_infer = infer.clone();
        let local_info = info.clone();

        // Rerank is the only payload that can me matched safely
        if let Ok(instance) = serde_json::from_value::<RerankRequest>(instance.clone()) {
            futures.push(rerank_future(local_infer, local_info, instance).boxed());
            continue;
        }

        match info.model_type {
            ModelType::Classifier(_) | ModelType::Reranker(_) => {
                let instance = serde_json::from_value::<PredictRequest>(instance)
                    .map_err(ErrorResponse::from)?;
                futures.push(predict_future(local_infer, local_info, instance).boxed());
            }
            ModelType::Embedding(_) => {
                if infer.is_splade() {
                    let instance = serde_json::from_value::<EmbedSparseRequest>(instance)
                        .map_err(ErrorResponse::from)?;
                    futures.push(embed_sparse_future(local_infer, local_info, instance).boxed());
                } else {
                    let instance = serde_json::from_value::<EmbedRequest>(instance)
                        .map_err(ErrorResponse::from)?;
                    futures.push(embed_future(local_infer, local_info, instance).boxed());
                }
            }
        }
    }

    let predictions = join_all(futures)
        .await
        .into_iter()
        .collect::<Result<Vec<VertexPrediction>, (StatusCode, Json<ErrorResponse>)>>()?;

    Ok(Json(VertexResponse { predictions }))
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
    payload_limit: usize,
    api_key: Option<String>,
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
    embed_sparse,
    openai_embed,
    similarity,
    tokenize,
    decode,
    metrics,
    ),
    components(
    schemas(
    PredictInput,
    Input,
    Info,
    ModelType,
    ClassifierModel,
    Embedding,
    EncodingFormat,
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
    EmbedSparseRequest,
    SparseValue,
    EmbedSparseResponse,
    RerankRequest,
    Rank,
    RerankResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    OpenAICompatErrorResponse,
    TokenizeInput,
    TokenizeRequest,
    TokenizeResponse,
    TruncationDirection,
    SimilarityInput,
    SimilarityParameters,
    SimilarityRequest,
    SimilarityResponse,
    SimpleToken,
    InputType,
    InputIds,
    DecodeRequest,
    DecodeResponse,
    ErrorType,
    )
    ),
    tags(
    (name = "Text Embeddings Inference", description = "Hugging Face Text Embeddings Inference API")
    ),
    info(
    title = "Text Embeddings Inference",
    license(
    name = "Apache 2.0",
    url = "https://www.apache.org/licenses/LICENSE-2.0"
    )
    )
    )]
    struct ApiDoc;

    // CORS allowed origins
    // map to go inside the option and then map to parse from String to HeaderValue
    // Finally, convert to AllowOrigin
    let allow_origin: Option<AllowOrigin> = cors_allow_origin.map(|cors_allow_origin| {
        if cors_allow_origin.iter().any(|origin| origin == "*") {
            AllowOrigin::any()
        } else {
            AllowOrigin::list(
                cors_allow_origin
                    .into_iter()
                    .map(|origin| origin.parse::<HeaderValue>().unwrap()),
            )
        }
    });

    // See: https://github.com/metrics-rs/metrics/issues/467#issuecomment-2022755151
    let (recorder, _) = prom_builder
        .build()
        .context("failed to build prometheus recorder")?;
    let prom_handle = recorder.handle();
    metrics::set_global_recorder(recorder).context("Failed to set global recorder")?;

    // CORS layer
    let allow_origin = allow_origin.unwrap_or(AllowOrigin::any());
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    // Define VertextApiDoc conditionally only if the "google" feature is enabled
    let doc = {
        // avoid `mut` if possible
        #[cfg(feature = "google")]
        {
            #[derive(OpenApi)]
            #[openapi(
                paths(vertex_compatibility),
                components(schemas(VertexRequest, VertexResponse, VertexPrediction))
            )]
            struct VertextApiDoc;

            // limiting mutability to the smallest scope necessary
            let mut doc = ApiDoc::openapi();
            doc.merge(VertextApiDoc::openapi());
            doc
        }
        #[cfg(not(feature = "google"))]
        ApiDoc::openapi()
    };

    let mut routes = Router::new()
        // Base routes
        .route("/info", get(get_model_info))
        .route("/embed", post(embed))
        .route("/embed_all", post(embed_all))
        .route("/embed_sparse", post(embed_sparse))
        .route("/predict", post(predict))
        .route("/rerank", post(rerank))
        .route("/similarity", post(similarity))
        .route("/tokenize", post(tokenize))
        .route("/decode", post(decode))
        // OpenAI compat route
        .route("/embeddings", post(openai_embed))
        .route("/v1/embeddings", post(openai_embed))
        // Vertex compat route
        .route("/vertex", post(vertex_compatibility));

    #[allow(unused_mut)]
    let mut public_routes = Router::new()
        // Base Health route
        .route("/health", get(health))
        // Inference API health route
        .route("/", get(health))
        // AWS Sagemaker health route
        .route("/ping", get(health))
        // Prometheus metrics route
        .route("/metrics", get(metrics));

    #[cfg(feature = "google")]
    {
        tracing::info!("Built with `google` feature");

        if let Ok(env_predict_route) = std::env::var("AIP_PREDICT_ROUTE") {
            tracing::info!("Serving Vertex compatible route on {env_predict_route}");
            routes = routes.route(&env_predict_route, post(vertex_compatibility));
        }

        if let Ok(env_health_route) = std::env::var("AIP_HEALTH_ROUTE") {
            tracing::info!("Serving Vertex compatible health route on {env_health_route}");
            public_routes = public_routes.route(&env_health_route, get(health));
        }
    }
    #[cfg(not(feature = "google"))]
    {
        // Set default routes
        routes = match &info.model_type {
            ModelType::Classifier(_) => {
                routes
                    .route("/", post(predict))
                    // AWS Sagemaker route
                    .route("/invocations", post(predict))
            }
            ModelType::Reranker(_) => {
                routes
                    .route("/", post(rerank))
                    // AWS Sagemaker route
                    .route("/invocations", post(rerank))
            }
            ModelType::Embedding(model) => {
                if std::env::var("TASK").ok() == Some("sentence-similarity".to_string()) {
                    routes
                        .route("/", post(similarity))
                        // AWS Sagemaker route
                        .route("/invocations", post(similarity))
                } else if model.pooling == "splade" {
                    routes
                        .route("/", post(embed_sparse))
                        // AWS Sagemaker route
                        .route("/invocations", post(embed_sparse))
                } else {
                    routes
                        .route("/", post(embed))
                        // AWS Sagemaker route
                        .route("/invocations", post(embed))
                }
            }
        };
    }

    if let Some(api_key) = api_key {
        let prefix = format!("Bearer {}", api_key);

        // Leak to allow FnMut
        let api_key: &'static str = prefix.leak();

        let auth = move |headers: HeaderMap,
                         request: axum::extract::Request,
                         next: axum::middleware::Next| async move {
            match headers.get(AUTHORIZATION) {
                Some(token) if token == api_key => {
                    let response = next.run(request).await;
                    Ok(response)
                }
                _ => Err(StatusCode::UNAUTHORIZED),
            }
        };

        routes = routes.layer(axum::middleware::from_fn(auth));
    }

    let app = Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", doc))
        .merge(routes)
        .merge(public_routes)
        .layer(Extension(infer))
        .layer(Extension(info))
        .layer(Extension(prom_handle.clone()))
        .layer(OtelAxumLayer::default())
        .layer(DefaultBodyLimit::max(payload_limit))
        .layer(cors_layer);

    // Run server
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context(format!("Could not bind TCP Listener on {addr}"))?;

    tracing::info!("Starting HTTP server: {}", &addr);
    tracing::info!("Ready");

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
            ErrorType::Empty => StatusCode::BAD_REQUEST,
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

impl From<serde_json::Error> for ErrorResponse {
    fn from(err: serde_json::Error) -> Self {
        ErrorResponse {
            error: err.to_string(),
            error_type: ErrorType::Validation,
        }
    }
}
