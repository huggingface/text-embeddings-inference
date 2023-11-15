/// HTTP Server logic
use crate::{
    ClassifierModel, EmbedRequest, EmbedResponse, EmbeddingModel, ErrorResponse, ErrorType, Info,
    Input, ModelType, OpenAICompatEmbedding, OpenAICompatErrorResponse, OpenAICompatRequest,
    OpenAICompatResponse, OpenAICompatUsage, PredictRequest, PredictResponse, Prediction, Sequence,
};
use axum::extract::Extension;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::routing::{get, post};
use axum::{http, Json, Router};
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;
use futures::future::join_all;
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use text_embeddings_core::infer::{Infer, InferResponse};
use text_embeddings_core::TextEmbeddingsError;
use tokio::signal;
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

    metrics::increment_counter!("te_request_count", "method" => "single");

    let compute_chars = req.inputs.count_chars();

    let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
    let mut response = infer
        .predict(req.inputs, req.truncate, permit)
        .await
        .map_err(ErrorResponse::from)?;

    let id2label = match &info.model_type {
        ModelType::Classifier(classifier) => &classifier.id2label,
        _ => panic!(),
    };

    let mut predictions: Vec<Prediction> = {
        if !req.raw_scores {
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

        // Map score to label
        response
            .results
            .into_iter()
            .enumerate()
            .map(|(i, s)| Prediction {
                score: s,
                label: id2label.get(&i.to_string()).unwrap().clone(),
            })
            .collect()
    };
    // Reverse sort
    predictions.sort_by(|x, y| x.score.partial_cmp(&y.score).unwrap());
    predictions.reverse();

    metrics::increment_counter!("te_request_success", "method" => "predict");

    let compute_tokens = response.prompt_tokens;
    let tokenization_time = response.tokenization;
    let queue_time = response.queue;
    let inference_time = response.inference;

    let total_time = start_time.elapsed();

    // Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("tokenization_time", format!("{tokenization_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-characters",
        compute_chars.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-tokens",
        compute_tokens.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-tokenization-time",
        tokenization_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );

    // Metrics
    metrics::histogram!("te_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "te_request_tokenization_duration",
        tokenization_time.as_secs_f64()
    );
    metrics::histogram!("te_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "te_request_inference_duration",
        inference_time.as_secs_f64()
    );

    tracing::info!("Success");

    Ok((headers, Json(PredictResponse(predictions))))
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

    let (compute_chars, compute_tokens, tokenization_time, queue_time, inference_time, response) =
        match req.inputs {
            Input::Single(input) => {
                metrics::increment_counter!("te_request_count", "method" => "single");

                let compute_chars = input.chars().count();

                let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
                let response = infer
                    .embed(input, req.truncate, req.normalize, permit)
                    .await
                    .map_err(ErrorResponse::from)?;

                metrics::increment_counter!("te_request_success", "method" => "single");

                (
                    compute_chars,
                    response.prompt_tokens,
                    response.tokenization,
                    response.queue,
                    response.inference,
                    EmbedResponse(vec![response.results]),
                )
            }
            Input::Batch(inputs) => {
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
                    compute_chars += input.chars().count();

                    let local_infer = infer.clone();
                    futures.push(async move {
                        let permit = local_infer.acquire_permit().await;
                        local_infer
                            .embed(input, req.truncate, req.normalize, permit)
                            .await
                    })
                }
                let results = join_all(futures)
                    .await
                    .into_iter()
                    .collect::<Result<Vec<InferResponse>, TextEmbeddingsError>>()
                    .map_err(ErrorResponse::from)?;

                let mut embeddings = Vec::with_capacity(batch_size);
                let mut total_tokenization_time = 0;
                let mut total_queue_time = 0;
                let mut total_inference_time = 0;
                let mut total_compute_tokens = 0;

                for r in results {
                    total_tokenization_time += r.tokenization.as_nanos() as u64;
                    total_queue_time += r.queue.as_nanos() as u64;
                    total_inference_time += r.inference.as_nanos() as u64;
                    total_compute_tokens += r.prompt_tokens;
                    embeddings.push(r.results);
                }
                let batch_size = batch_size as u64;

                metrics::increment_counter!("te_request_success", "method" => "batch");

                (
                    compute_chars,
                    total_compute_tokens,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                    EmbedResponse(embeddings),
                )
            }
        };

    let total_time = start_time.elapsed();

    // Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("tokenization_time", format!("{tokenization_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-characters",
        compute_chars.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-tokens",
        compute_tokens.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-tokenization-time",
        tokenization_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );

    // Metrics
    metrics::histogram!("te_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "te_request_tokenization_duration",
        tokenization_time.as_secs_f64()
    );
    metrics::histogram!("te_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "te_request_inference_duration",
        inference_time.as_secs_f64()
    );

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

    let (compute_chars, compute_tokens, tokenization_time, queue_time, inference_time, embeddings) =
        match req.input {
            Input::Single(input) => {
                metrics::increment_counter!("te_request_count", "method" => "single");

                let compute_chars = input.chars().count();

                let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;
                let response = infer
                    .embed(input, false, true, permit)
                    .await
                    .map_err(ErrorResponse::from)?;

                metrics::increment_counter!("te_request_success", "method" => "single");

                (
                    compute_chars,
                    response.prompt_tokens,
                    response.tokenization,
                    response.queue,
                    response.inference,
                    vec![OpenAICompatEmbedding {
                        object: "embedding",
                        embedding: response.results,
                        index: 0,
                    }],
                )
            }
            Input::Batch(inputs) => {
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
                    compute_chars += input.chars().count();

                    let local_infer = infer.clone();
                    futures.push(async move {
                        let permit = local_infer.acquire_permit().await;
                        local_infer.embed(input, false, true, permit).await
                    })
                }
                let results = join_all(futures)
                    .await
                    .into_iter()
                    .collect::<Result<Vec<InferResponse>, TextEmbeddingsError>>()
                    .map_err(ErrorResponse::from)?;

                let mut embeddings = Vec::with_capacity(batch_size);
                let mut total_tokenization_time = 0;
                let mut total_queue_time = 0;
                let mut total_inference_time = 0;
                let mut total_compute_tokens = 0;

                for (i, r) in results.into_iter().enumerate() {
                    total_tokenization_time += r.tokenization.as_nanos() as u64;
                    total_queue_time += r.queue.as_nanos() as u64;
                    total_inference_time += r.inference.as_nanos() as u64;
                    total_compute_tokens += r.prompt_tokens;
                    embeddings.push(OpenAICompatEmbedding {
                        object: "embedding",
                        embedding: r.results,
                        index: i,
                    });
                }
                let batch_size = batch_size as u64;

                metrics::increment_counter!("te_request_success", "method" => "batch");

                (
                    compute_chars,
                    total_compute_tokens,
                    Duration::from_nanos(total_tokenization_time / batch_size),
                    Duration::from_nanos(total_queue_time / batch_size),
                    Duration::from_nanos(total_inference_time / batch_size),
                    embeddings,
                )
            }
        };

    let total_time = start_time.elapsed();

    // Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("tokenization_time", format!("{tokenization_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-characters",
        compute_chars.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-tokens",
        compute_tokens.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-tokenization-time",
        tokenization_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );

    // Metrics
    metrics::histogram!("te_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "te_request_tokenization_duration",
        tokenization_time.as_secs_f64()
    );
    metrics::histogram!("te_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "te_request_inference_duration",
        inference_time.as_secs_f64()
    );

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
    allow_origin: Option<AllowOrigin>,
) -> Result<(), axum::BoxError> {
    // OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
    paths(
    get_model_info,
    health,
    predict,
    embed,
    openai_embed,
    metrics,
    ),
    components(
    schemas(
    Sequence,
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
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    OpenAICompatErrorResponse,
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

    // Duration buckets
    let duration_matcher = Matcher::Suffix(String::from("duration"));
    let n_duration_buckets = 35;
    let mut duration_buckets = Vec::with_capacity(n_duration_buckets);
    // Minimum duration in seconds
    let mut value = 0.00001;
    for _ in 0..n_duration_buckets {
        // geometric sequence
        value *= 1.5;
        duration_buckets.push(value);
    }

    // Input Length buckets
    let input_length_matcher = Matcher::Full(String::from("te_request_input_length"));
    let input_length_buckets: Vec<f64> = (0..100)
        .map(|x| (info.max_input_length as f64 / 100.0) * (x + 1) as f64)
        .collect();

    // Batch size buckets
    let batch_size_matcher = Matcher::Full(String::from("te_batch_next_size"));
    let batch_size_buckets: Vec<f64> = (0..2048).map(|x| (x + 1) as f64).collect();

    // Batch tokens buckets
    let batch_tokens_matcher = Matcher::Full(String::from("te_batch_next_tokens"));
    let batch_tokens_buckets: Vec<f64> = (0..100_000).map(|x| (x + 1) as f64).collect();

    // Prometheus handler
    let builder = PrometheusBuilder::new()
        .set_buckets_for_metric(duration_matcher, &duration_buckets)
        .unwrap()
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)
        .unwrap()
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)
        .unwrap()
        .set_buckets_for_metric(batch_tokens_matcher, &batch_tokens_buckets)
        .unwrap();

    let prom_handle = builder
        .install_recorder()
        .expect("failed to install metrics recorder");

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
        .route("/predict", post(predict))
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
    let app = match infer.is_classifier() {
        true => {
            app.route("/", post(predict))
                // AWS Sagemaker route
                .route("/invocations", post(predict))
        }
        false => {
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
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        // Wait until all requests are finished to shut down
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// Shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("signal received, starting graceful shutdown");
    opentelemetry::global::shutdown_tracer_provider();
}

impl From<TextEmbeddingsError> for ErrorResponse {
    fn from(err: TextEmbeddingsError) -> Self {
        let error_type = match err {
            TextEmbeddingsError::Tokenizer(_) => ErrorType::Tokenizer,
            TextEmbeddingsError::Validation(_) => ErrorType::Validation,
            TextEmbeddingsError::Overloaded(_) => ErrorType::Overloaded,
            TextEmbeddingsError::Backend(_) => ErrorType::Backend,
        };
        Self {
            error: err.to_string(),
            error_type,
        }
    }
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
