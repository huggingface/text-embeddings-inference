use crate::grpc::{
    EmbedRequest, EmbedResponse, InfoRequest, InfoResponse, PredictRequest, PredictResponse,
    Prediction, Rank, RerankRequest, RerankResponse,
};
use crate::{grpc, shutdown, ErrorResponse, ErrorType, Info, ModelType};
use futures::future::join_all;
use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use text_embeddings_core::infer::{Infer, InferResponse};
use tonic::codegen::http::HeaderMap;
use tonic::metadata::MetadataMap;
use tonic::server::NamedService;
use tonic::transport::Server;
use tonic::{Code, Extensions, Request, Response, Status};
use tonic_health::ServingStatus;
use tracing::{instrument, Span};

struct ResponseMetadata {
    compute_chars: usize,
    compute_tokens: usize,
    start_time: Instant,
    tokenization_time: Duration,
    queue_time: Duration,
    inference_time: Duration,
}

impl ResponseMetadata {
    fn new(compute_chars: usize, start_time: Instant, response: &InferResponse) -> Self {
        Self {
            compute_chars,
            compute_tokens: response.prompt_tokens,
            start_time,
            tokenization_time: response.tokenization,
            queue_time: response.queue,
            inference_time: response.inference,
        }
    }

    fn record_span(&self, span: &Span) {
        // Tracing metadata
        span.record("compute_chars", self.compute_chars);
        span.record("compute_tokens", self.compute_tokens);
        span.record("total_time", format!("{:?}", self.start_time.elapsed()));
        span.record("tokenization_time", format!("{:?}", self.tokenization_time));
        span.record("queue_time", format!("{:?}", self.queue_time));
        span.record("inference_time", format!("{:?}", self.inference_time));
    }

    fn record_metrics(&self) {
        // Metrics
        metrics::histogram!(
            "te_request_duration",
            self.start_time.elapsed().as_secs_f64()
        );
        metrics::histogram!(
            "te_request_tokenization_duration",
            self.tokenization_time.as_secs_f64()
        );
        metrics::histogram!("te_request_queue_duration", self.queue_time.as_secs_f64());
        metrics::histogram!(
            "te_request_inference_duration",
            self.inference_time.as_secs_f64()
        );
    }
}

impl From<ResponseMetadata> for HeaderMap {
    fn from(value: ResponseMetadata) -> Self {
        // Headers
        let mut headers = HeaderMap::new();
        headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
        headers.insert(
            "x-compute-time",
            value
                .start_time
                .elapsed()
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers.insert(
            "x-compute-characters",
            value.compute_chars.to_string().parse().unwrap(),
        );
        headers.insert(
            "x-compute-tokens",
            value.compute_tokens.to_string().parse().unwrap(),
        );
        headers.insert(
            "x-total-time",
            value
                .start_time
                .elapsed()
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers.insert(
            "x-tokenization-time",
            value
                .tokenization_time
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers.insert(
            "x-queue-time",
            value.queue_time.as_millis().to_string().parse().unwrap(),
        );
        headers.insert(
            "x-inference-time",
            value
                .inference_time
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers
    }
}

#[derive(Debug, Clone)]
struct TextEmbeddingsService {
    infer: Infer,
    info: Info,
}

impl TextEmbeddingsService {
    #[instrument(
        skip_all,
        fields(
            compute_chars,
            compute_tokens,
            total_time,
            tokenization_time,
            queue_time,
            inference_time,
        )
    )]
    async fn embed_inner(
        &self,
        request: EmbedRequest,
    ) -> Result<(EmbedResponse, ResponseMetadata), Status> {
        let span = tracing::Span::current();
        let start_time = Instant::now();

        metrics::increment_counter!("te_request_count", "method" => "single");

        let compute_chars = request.inputs.chars().count();

        let permit = self
            .infer
            .try_acquire_permit()
            .map_err(ErrorResponse::from)?;
        let response = self
            .infer
            .embed(request.inputs, request.truncate, request.normalize, permit)
            .await
            .map_err(ErrorResponse::from)?;

        metrics::increment_counter!("te_request_success", "method" => "single");

        let response_metadata = ResponseMetadata::new(compute_chars, start_time, &response);
        response_metadata.record_span(&span);
        response_metadata.record_metrics();

        tracing::info!("Success");

        Ok((
            EmbedResponse {
                embeddings: response.results,
            },
            response_metadata,
        ))
    }

    #[instrument(
        skip_all,
        fields(
            compute_chars,
            compute_tokens,
            total_time,
            tokenization_time,
            queue_time,
            inference_time,
        )
    )]
    async fn predict_inner(
        &self,
        request: PredictRequest,
    ) -> Result<(PredictResponse, ResponseMetadata), Status> {
        let span = tracing::Span::current();
        let start_time = Instant::now();

        metrics::increment_counter!("te_request_count", "method" => "single");

        let compute_chars = request.inputs.chars().count();

        let permit = self
            .infer
            .try_acquire_permit()
            .map_err(ErrorResponse::from)?;
        let response = self
            .infer
            .predict(request.inputs, request.truncate, request.raw_scores, permit)
            .await
            .map_err(ErrorResponse::from)?;

        let id2label = match &self.info.model_type {
            ModelType::Classifier(classifier) => &classifier.id2label,
            ModelType::Reranker(classifier) => &classifier.id2label,
            _ => panic!(),
        };

        let response_metadata = ResponseMetadata::new(compute_chars, start_time, &response);

        let mut predictions: Vec<Prediction> = {
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

        metrics::increment_counter!("te_request_success", "method" => "single");

        response_metadata.record_span(&span);
        response_metadata.record_metrics();

        tracing::info!("Success");

        Ok((PredictResponse { predictions }, response_metadata))
    }

    #[instrument(
        skip_all,
        fields(
            compute_chars,
            compute_tokens,
            total_time,
            tokenization_time,
            queue_time,
            inference_time,
        )
    )]
    async fn rerank_inner(
        &self,
        request: RerankRequest,
    ) -> Result<(RerankResponse, ResponseMetadata), Status> {
        let span = tracing::Span::current();
        let start_time = Instant::now();

        match &self.info.model_type {
            ModelType::Classifier(_) => {
                metrics::increment_counter!("te_request_failure", "err" => "model_type");
                let message = "model is not a re-ranker model".to_string();
                tracing::error!("{message}");
                Err(Status::new(Code::FailedPrecondition, message))
            }
            ModelType::Reranker(_) => Ok(()),
            ModelType::Embedding(_) => {
                metrics::increment_counter!("te_request_failure", "err" => "model_type");
                let message = "model is not a classifier model".to_string();
                tracing::error!("{message}");
                Err(Status::new(Code::FailedPrecondition, message))
            }
        }?;

        // Closure for rerank
        let rerank_inner = move |query: String,
                                 text: String,
                                 truncate: bool,
                                 raw_scores: bool,
                                 infer: Infer| async move {
            let permit = infer.try_acquire_permit().map_err(ErrorResponse::from)?;

            let response = infer
                .predict((query, text), truncate, raw_scores, permit)
                .await
                .map_err(ErrorResponse::from)?;

            let score = response.results[0];

            Ok::<(usize, Duration, Duration, Duration, f32), ErrorResponse>((
                response.prompt_tokens,
                response.tokenization,
                response.queue,
                response.inference,
                score,
            ))
        };

        metrics::increment_counter!("te_request_count", "method" => "batch");

        let batch_size = request.texts.len();
        if batch_size > self.info.max_client_batch_size {
            let message = format!(
                "batch size {batch_size} > maximum allowed batch size {}",
                self.info.max_client_batch_size
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
        let query_chars = request.query.chars().count();
        let mut total_compute_chars = query_chars * batch_size;

        for text in &request.texts {
            total_compute_chars += text.chars().count();
            let local_infer = self.infer.clone();
            futures.push(rerank_inner(
                request.query.clone(),
                text.clone(),
                request.truncate,
                request.raw_scores,
                local_infer,
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
            let text = if request.return_text {
                Some(request.texts[index].clone())
            } else {
                None
            };

            ranks.push(Rank {
                index: index as u32,
                text,
                score: r.4,
            })
        }

        // Reverse sort
        ranks.sort_by(|x, y| x.score.partial_cmp(&y.score).unwrap());
        ranks.reverse();

        let batch_size = batch_size as u64;

        metrics::increment_counter!("te_request_success", "method" => "batch");

        let response_metadata = ResponseMetadata {
            compute_chars: total_compute_chars,
            compute_tokens: total_compute_tokens,
            start_time,
            tokenization_time: Duration::from_nanos(total_tokenization_time / batch_size),
            queue_time: Duration::from_nanos(total_queue_time / batch_size),
            inference_time: Duration::from_nanos(total_inference_time / batch_size),
        };
        response_metadata.record_span(&span);
        response_metadata.record_metrics();

        tracing::info!("Success");

        Ok((RerankResponse { ranks }, response_metadata))
    }
}

#[tonic::async_trait]
impl grpc::info_server::Info for TextEmbeddingsService {
    async fn info(&self, _request: Request<InfoRequest>) -> Result<Response<InfoResponse>, Status> {
        let model_type = match self.info.model_type {
            ModelType::Classifier(_) => grpc::ModelType::Classifier,
            ModelType::Embedding(_) => grpc::ModelType::Embedding,
            ModelType::Reranker(_) => grpc::ModelType::Reranker,
        };

        Ok(Response::new(InfoResponse {
            version: self.info.version.to_string(),
            sha: self.info.sha.map(|s| s.to_string()),
            docker_label: self.info.docker_label.map(|s| s.to_string()),
            model_id: self.info.model_id.clone(),
            model_sha: self.info.model_sha.clone(),
            model_dtype: self.info.model_dtype.clone(),
            model_type: model_type.into(),
            max_concurrent_requests: self.info.max_concurrent_requests as u32,
            max_input_length: self.info.max_input_length as u32,
            max_batch_tokens: self.info.max_batch_tokens as u32,
            max_batch_requests: self.info.max_batch_requests.map(|v| v as u32),
            max_client_batch_size: self.info.max_client_batch_size as u32,
            tokenization_workers: self.info.tokenization_workers as u32,
        }))
    }
}

#[tonic::async_trait]
impl grpc::embed_server::Embed for TextEmbeddingsService {
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let request = request.into_inner();

        let (response, metadata) = self.embed_inner(request).await?;

        let headers = HeaderMap::from(metadata);

        Ok(Response::from_parts(
            MetadataMap::from_headers(headers),
            response,
            Extensions::default(),
        ))
    }
}

#[tonic::async_trait]
impl grpc::predict_server::Predict for TextEmbeddingsService {
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let request = request.into_inner();

        let (response, metadata) = self.predict_inner(request).await?;

        let headers = HeaderMap::from(metadata);

        Ok(Response::from_parts(
            MetadataMap::from_headers(headers),
            response,
            Extensions::default(),
        ))
    }
}

#[tonic::async_trait]
impl grpc::rerank_server::Rerank for TextEmbeddingsService {
    async fn rerank(
        &self,
        request: Request<RerankRequest>,
    ) -> Result<Response<RerankResponse>, Status> {
        let request = request.into_inner();

        let (response, metadata) = self.rerank_inner(request).await?;

        let headers = HeaderMap::from(metadata);

        Ok(Response::from_parts(
            MetadataMap::from_headers(headers),
            response,
            Extensions::default(),
        ))
    }
}

pub async fn run(
    infer: Infer,
    info: Info,
    addr: SocketAddr,
    prom_builder: PrometheusBuilder,
) -> Result<(), anyhow::Error> {
    prom_builder.install()?;
    tracing::info!("Serving Prometheus metrics: 0.0.0.0:9000");

    // Liveness service
    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    // Info is always serving
    health_reporter
        .set_serving::<grpc::InfoServer<TextEmbeddingsService>>()
        .await;
    // Set all other services to not serving
    // Their health will be updated in the task below
    health_reporter
        .set_not_serving::<grpc::EmbedServer<TextEmbeddingsService>>()
        .await;
    health_reporter
        .set_not_serving::<grpc::RerankServer<TextEmbeddingsService>>()
        .await;
    health_reporter
        .set_not_serving::<grpc::PredictServer<TextEmbeddingsService>>()
        .await;

    // Backend health watcher
    let mut health_watcher = infer.health_watcher();

    // Clone model_type and move it to the task
    let health_watcher_model_type = info.model_type.clone();

    // Update services health
    tokio::spawn(async move {
        while health_watcher.changed().await.is_ok() {
            let health = *health_watcher.borrow_and_update();
            let status = match health {
                true => ServingStatus::Serving,
                false => ServingStatus::NotServing,
            };

            // Match on model type and set the health of the correct service(s)
            //
            // If Reranker, we have both a predict and rerank service
            //
            // This logic hints back to the user that if they try using the wrong service
            // given the model type, it will always return an error.
            //
            // For example if the model type is `Embedding`, sending requests to `Rerank` will
            // always return an `UNIMPLEMENTED` Status and both the `Rerank` and `Predict` services
            // will have a `NOT_SERVING` ServingStatus.
            match health_watcher_model_type {
                ModelType::Classifier(_) => {
                    health_reporter
                        .set_service_status(
                            <grpc::PredictServer<TextEmbeddingsService>>::NAME,
                            status,
                        )
                        .await
                }
                ModelType::Embedding(_) => {
                    health_reporter
                        .set_service_status(
                            <grpc::EmbedServer<TextEmbeddingsService>>::NAME,
                            status,
                        )
                        .await
                }
                ModelType::Reranker(_) => {
                    // Reranker has both a predict and rerank service
                    health_reporter
                        .set_service_status(
                            <grpc::PredictServer<TextEmbeddingsService>>::NAME,
                            status,
                        )
                        .await;
                    health_reporter
                        .set_service_status(
                            <grpc::RerankServer<TextEmbeddingsService>>::NAME,
                            status,
                        )
                        .await;
                }
            };
        }
    });

    // gRPC reflection
    let file_descriptor_set: &[u8] = tonic::include_file_descriptor_set!("descriptor");
    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(file_descriptor_set)
        .build()?;

    // Main service
    let service = TextEmbeddingsService { infer, info };

    // Create gRPC server
    tracing::info!("Starting gRPC server: {}", &addr);
    Server::builder()
        .add_service(health_service)
        .add_service(reflection_service)
        .add_service(grpc::InfoServer::new(service.clone()))
        .add_service(grpc::EmbedServer::new(service.clone()))
        .add_service(grpc::PredictServer::new(service.clone()))
        .add_service(grpc::RerankServer::new(service))
        .serve_with_shutdown(addr, shutdown::shutdown_signal())
        .await?;

    Ok(())
}

impl From<ErrorResponse> for Status {
    fn from(value: ErrorResponse) -> Self {
        let code = match value.error_type {
            ErrorType::Unhealthy => Code::Unavailable,
            ErrorType::Backend => Code::FailedPrecondition,
            ErrorType::Overloaded => Code::ResourceExhausted,
            ErrorType::Validation => Code::InvalidArgument,
            ErrorType::Tokenizer => Code::FailedPrecondition,
        };

        Status::new(code, value.error)
    }
}
