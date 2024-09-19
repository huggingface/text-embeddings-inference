use crate::grpc::pb::tei::v1::{
    EmbedAllRequest, EmbedAllResponse, EmbedSparseRequest, EmbedSparseResponse, EncodeRequest,
    EncodeResponse, PredictPairRequest, RerankStreamRequest, SimpleToken, SparseValue,
    TokenEmbedding, TruncationDirection,
};
use crate::grpc::{
    DecodeRequest, DecodeResponse, EmbedRequest, EmbedResponse, InfoRequest, InfoResponse,
    PredictRequest, PredictResponse, Prediction, Rank, RerankRequest, RerankResponse,
};
use crate::ResponseMetadata;
use crate::{grpc, shutdown, ErrorResponse, ErrorType, Info, ModelType};
use futures::future::join_all;
use metrics_exporter_prometheus::PrometheusBuilder;
use std::future::Future;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use text_embeddings_core::infer::Infer;
use text_embeddings_core::tokenization::EncodingInput;
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tonic::codegen::http::HeaderMap;
use tonic::metadata::MetadataMap;
use tonic::server::NamedService;
use tonic::transport::Server;
use tonic::{Code, Extensions, Request, Response, Status, Streaming};
use tonic_health::ServingStatus;
use tracing::{instrument, Span};

impl From<&ResponseMetadata> for grpc::Metadata {
    fn from(value: &ResponseMetadata) -> Self {
        Self {
            compute_chars: value.compute_chars as u32,
            compute_tokens: value.compute_tokens as u32,
            total_time_ns: value.start_time.elapsed().as_nanos() as u64,
            tokenization_time_ns: value.tokenization_time.as_nanos() as u64,
            queue_time_ns: value.queue_time.as_nanos() as u64,
            inference_time_ns: value.inference_time.as_nanos() as u64,
        }
    }
}

#[derive(Debug, Clone)]
struct TextEmbeddingsService {
    infer: Infer,
    info: Info,
    max_parallel_stream_requests: usize,
}

impl TextEmbeddingsService {
    fn new(infer: Infer, info: Info) -> Self {
        let max_parallel_stream_requests = std::env::var("GRPC_MAX_PARALLEL_STREAM_REQUESTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1024);
        Self {
            infer,
            info,
            max_parallel_stream_requests,
        }
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
    async fn embed_pooled_inner(
        &self,
        request: EmbedRequest,
        permit: OwnedSemaphorePermit,
    ) -> Result<(EmbedResponse, ResponseMetadata), Status> {
        let span = Span::current();
        let start_time = Instant::now();

        let compute_chars = request.inputs.chars().count();
        let truncation_direction = convert_truncation_direction(request.truncation_direction);
        let response = self
            .infer
            .embed_pooled(
                request.inputs,
                request.truncate,
                truncation_direction,
                request.prompt_name,
                request.normalize,
                permit,
            )
            .await
            .map_err(ErrorResponse::from)?;

        let response_metadata = ResponseMetadata::new(
            compute_chars,
            response.metadata.prompt_tokens,
            start_time,
            response.metadata.tokenization,
            response.metadata.queue,
            response.metadata.inference,
        );
        response_metadata.record_span(&span);
        response_metadata.record_metrics();

        tracing::info!("Success");

        Ok((
            EmbedResponse {
                embeddings: response.results,
                metadata: Some(grpc::Metadata::from(&response_metadata)),
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
    async fn embed_sparse_inner(
        &self,
        request: EmbedSparseRequest,
        permit: OwnedSemaphorePermit,
    ) -> Result<(EmbedSparseResponse, ResponseMetadata), Status> {
        let span = Span::current();
        let start_time = Instant::now();

        let compute_chars = request.inputs.chars().count();
        let truncation_direction = convert_truncation_direction(request.truncation_direction);
        let response = self
            .infer
            .embed_sparse(
                request.inputs,
                request.truncate,
                truncation_direction,
                request.prompt_name,
                permit,
            )
            .await
            .map_err(ErrorResponse::from)?;

        let response_metadata = ResponseMetadata::new(
            compute_chars,
            response.metadata.prompt_tokens,
            start_time,
            response.metadata.tokenization,
            response.metadata.queue,
            response.metadata.inference,
        );

        let mut sparse_values = Vec::with_capacity(response.results.len());
        for (index, value) in response.results.into_iter().enumerate() {
            if value != 0.0 {
                sparse_values.push(SparseValue {
                    index: index as u32,
                    value,
                });
            }
        }

        response_metadata.record_span(&span);
        response_metadata.record_metrics();

        tracing::info!("Success");

        Ok((
            EmbedSparseResponse {
                sparse_embeddings: sparse_values,
                metadata: Some(grpc::Metadata::from(&response_metadata)),
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
    async fn embed_all_inner(
        &self,
        request: EmbedAllRequest,
        permit: OwnedSemaphorePermit,
    ) -> Result<(EmbedAllResponse, ResponseMetadata), Status> {
        let span = Span::current();
        let start_time = Instant::now();

        let compute_chars = request.inputs.chars().count();
        let truncation_direction = convert_truncation_direction(request.truncation_direction);
        let response = self
            .infer
            .embed_all(
                request.inputs,
                request.truncate,
                truncation_direction,
                request.prompt_name,
                permit,
            )
            .await
            .map_err(ErrorResponse::from)?;

        let response_metadata = ResponseMetadata::new(
            compute_chars,
            response.metadata.prompt_tokens,
            start_time,
            response.metadata.tokenization,
            response.metadata.queue,
            response.metadata.inference,
        );
        response_metadata.record_span(&span);
        response_metadata.record_metrics();

        tracing::info!("Success");

        let token_embeddings = response
            .results
            .into_iter()
            .map(|v| TokenEmbedding { embeddings: v })
            .collect();

        Ok((
            EmbedAllResponse {
                token_embeddings,
                metadata: Some(grpc::Metadata::from(&response_metadata)),
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
    async fn predict_inner<I: Into<EncodingInput> + std::fmt::Debug>(
        &self,
        inputs: I,
        truncate: bool,
        truncation_direction: tokenizers::TruncationDirection,
        raw_scores: bool,
        permit: OwnedSemaphorePermit,
    ) -> Result<(PredictResponse, ResponseMetadata), Status> {
        let span = Span::current();
        let start_time = Instant::now();

        let inputs = inputs.into();
        let compute_chars = match &inputs {
            EncodingInput::Single(s) => s.chars().count(),
            EncodingInput::Dual(s1, s2) => s1.chars().count() + s2.chars().count(),
            EncodingInput::Ids(_) => unreachable!(),
        };

        let response = self
            .infer
            .predict(inputs, truncate, truncation_direction, raw_scores, permit)
            .await
            .map_err(ErrorResponse::from)?;

        let id2label = match &self.info.model_type {
            ModelType::Classifier(classifier) => &classifier.id2label,
            ModelType::Reranker(classifier) => &classifier.id2label,
            _ => panic!(),
        };

        let response_metadata = ResponseMetadata::new(
            compute_chars,
            response.metadata.prompt_tokens,
            start_time,
            response.metadata.tokenization,
            response.metadata.queue,
            response.metadata.inference,
        );

        let mut predictions = Vec::with_capacity(response.results.len());
        for (i, s) in response.results.into_iter().enumerate() {
            // Check that s is not NaN or the partial_cmp below will panic
            if s.is_nan() {
                Err(ErrorResponse {
                    error: "score is NaN".to_string(),
                    error_type: ErrorType::Backend,
                })?;
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

        response_metadata.record_span(&span);
        response_metadata.record_metrics();

        tracing::info!("Success");

        Ok((
            PredictResponse {
                predictions,
                metadata: Some(grpc::Metadata::from(&response_metadata)),
            },
            response_metadata,
        ))
    }

    #[instrument(skip_all)]
    async fn tokenize_inner(&self, request: EncodeRequest) -> Result<EncodeResponse, Status> {
        let inputs = request.inputs;
        let (encoded_inputs, encoding) = self
            .infer
            .tokenize(
                inputs.clone(),
                request.add_special_tokens,
                request.prompt_name,
            )
            .await
            .map_err(ErrorResponse::from)?;
        let inputs = encoded_inputs.unwrap_or(inputs);

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
                        let text: String = inputs.chars().skip(start).take(stop - start).collect();
                        SimpleToken {
                            id,
                            text,
                            special,
                            start: Some(start as u32),
                            stop: Some(stop as u32),
                        }
                    }
                }
            })
            .collect();
        Ok(EncodeResponse { tokens })
    }

    #[instrument(skip_all)]
    async fn decode_inner(&self, request: DecodeRequest) -> Result<DecodeResponse, Status> {
        let ids = request.ids;
        let text = self
            .infer
            .decode(ids, request.skip_special_tokens)
            .await
            .map_err(ErrorResponse::from)?;
        Ok(DecodeResponse { text })
    }

    #[instrument(skip_all)]
    async fn stream<Req, Res, F, Fut>(
        &self,
        request: Request<Streaming<Req>>,
        function: F,
    ) -> Result<Response<UnboundedReceiverStream<Result<Res, Status>>>, Status>
    where
        Req: Send + 'static,
        Res: Send + 'static,
        F: FnOnce(Req, OwnedSemaphorePermit) -> Fut + Send + Clone + 'static,
        Fut: Future<Output = Result<(Res, ResponseMetadata), Status>> + Send,
    {
        let mut request_stream = request.into_inner();

        // Create bounded channel to have an upper bound of spawned tasks
        // We will have at most `max_parallel_stream_requests` messages from this stream in the queue
        let (internal_sender, mut internal_receiver) = mpsc::channel::<(
            Req,
            oneshot::Sender<Result<Res, Status>>,
        )>(self.max_parallel_stream_requests);

        // Required for the async move below
        let local = self.clone();

        // Background task that uses the bounded channel
        tokio::spawn(async move {
            while let Some((request, mut sender)) = internal_receiver.recv().await {
                // Wait on permit before spawning the task to avoid creating more tasks than needed
                let permit = local.infer.acquire_permit().await;

                // Required for the async move below
                let function_local = function.clone();

                // Create async task for this specific input
                tokio::spawn(async move {
                    // Select on closed to cancel work if the stream was closed
                    tokio::select! {
                    response = function_local(request, permit) => {
                        let _ = sender.send(response.map(|(r, _m)| r));
                    }
                    _ = sender.closed() => {}
                    }
                });
            }
        });

        // Intermediate channels
        // Required to keep the order of the requests
        let (intermediate_sender, mut intermediate_receiver) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            // Iterate on input
            while let Some(request) = request_stream.next().await {
                // Create return channel
                let (result_sender, result_receiver) = oneshot::channel();
                // Push to intermediate channel and preserve ordering
                intermediate_sender
                    .send(result_receiver)
                    .expect("`intermediate_receiver` was dropped. This is a bug.");

                match request {
                    Ok(request) => internal_sender
                        .send((request, result_sender))
                        .await
                        .expect("`internal_receiver` was dropped. This is a bug."),
                    Err(status) => {
                        // Request is malformed
                        let _ = result_sender.send(Err(status));
                    }
                };
            }
        });

        // Final channel for the outputs
        let (response_sender, response_receiver) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            while let Some(result_receiver) = intermediate_receiver.recv().await {
                // Select on closed to cancel work if the stream was closed
                tokio::select! {
                response = result_receiver => {
                    let _ = response_sender.send(response.expect("`result_sender` was dropped. This is a bug."));
                }
                _ = response_sender.closed() => {}
                }
            }
        });

        Ok(Response::new(UnboundedReceiverStream::new(
            response_receiver,
        )))
    }

    #[instrument(skip_all)]
    async fn stream_no_permit<Req, Res, F, Fut>(
        &self,
        request: Request<Streaming<Req>>,
        function: F,
    ) -> Result<Response<UnboundedReceiverStream<Result<Res, Status>>>, Status>
    where
        Req: Send + 'static,
        Res: Send + 'static,
        F: FnOnce(Req) -> Fut + Send + Clone + 'static,
        Fut: Future<Output = Result<Res, Status>> + Send,
    {
        let mut request_stream = request.into_inner();

        // Create bounded channel to have an upper bound of spawned tasks
        // We will have at most `max_parallel_stream_requests` messages from this stream in the queue
        let (internal_sender, mut internal_receiver) = mpsc::channel::<(
            Req,
            oneshot::Sender<Result<Res, Status>>,
        )>(self.max_parallel_stream_requests);

        // Background task that uses the bounded channel
        tokio::spawn(async move {
            while let Some((request, mut sender)) = internal_receiver.recv().await {
                // Required for the async move below
                let function_local = function.clone();

                // Create async task for this specific input
                tokio::spawn(async move {
                    // Select on closed to cancel work if the stream was closed
                    tokio::select! {
                    response = function_local(request) => {
                        let _ = sender.send(response);
                    }
                    _ = sender.closed() => {}
                    }
                });
            }
        });

        // Intermediate channels
        // Required to keep the order of the requests
        let (intermediate_sender, mut intermediate_receiver) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            // Iterate on input
            while let Some(request) = request_stream.next().await {
                // Create return channel
                let (result_sender, result_receiver) = oneshot::channel();
                // Push to intermediate channel and preserve ordering
                intermediate_sender
                    .send(result_receiver)
                    .expect("`intermediate_receiver` was dropped. This is a bug.");

                match request {
                    Ok(request) => internal_sender
                        .send((request, result_sender))
                        .await
                        .expect("`internal_receiver` was dropped. This is a bug."),
                    Err(status) => {
                        // Request is malformed
                        let _ = result_sender.send(Err(status));
                    }
                };
            }
        });

        // Final channel for the outputs
        let (response_sender, response_receiver) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            while let Some(result_receiver) = intermediate_receiver.recv().await {
                // Select on closed to cancel work if the stream was closed
                tokio::select! {
                response = result_receiver => {
                    let _ = response_sender.send(response.expect("`result_sender` was dropped. This is a bug."));
                }
                _ = response_sender.closed() => {}
                }
            }
        });

        Ok(Response::new(UnboundedReceiverStream::new(
            response_receiver,
        )))
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
    #[instrument(skip_all)]
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let counter = metrics::counter!("te_request_count", "method" => "single");
        counter.increment(1);

        let permit = self
            .infer
            .try_acquire_permit()
            .map_err(ErrorResponse::from)?;

        let request = request.into_inner();
        let (response, metadata) = self.embed_pooled_inner(request, permit).await?;
        let headers = HeaderMap::from(metadata);

        let counter = metrics::counter!("te_request_count", "method" => "single");
        counter.increment(1);

        Ok(Response::from_parts(
            MetadataMap::from_headers(headers),
            response,
            Extensions::default(),
        ))
    }

    type EmbedStreamStream = UnboundedReceiverStream<Result<EmbedResponse, Status>>;

    #[instrument(skip_all)]
    async fn embed_stream(
        &self,
        request: Request<Streaming<EmbedRequest>>,
    ) -> Result<Response<Self::EmbedStreamStream>, Status> {
        // Clone for move below
        let clone = self.clone();
        let function = |req: EmbedRequest, permit: OwnedSemaphorePermit| async move {
            clone.embed_pooled_inner(req, permit).await
        };

        self.stream(request, function).await
    }

    async fn embed_sparse(
        &self,
        request: Request<EmbedSparseRequest>,
    ) -> Result<Response<EmbedSparseResponse>, Status> {
        let counter = metrics::counter!("te_request_count", "method" => "single");
        counter.increment(1);

        let permit = self
            .infer
            .try_acquire_permit()
            .map_err(ErrorResponse::from)?;

        let request = request.into_inner();
        let (response, metadata) = self.embed_sparse_inner(request, permit).await?;
        let headers = HeaderMap::from(metadata);

        let counter = metrics::counter!("te_request_count", "method" => "single");
        counter.increment(1);

        Ok(Response::from_parts(
            MetadataMap::from_headers(headers),
            response,
            Extensions::default(),
        ))
    }

    type EmbedSparseStreamStream = UnboundedReceiverStream<Result<EmbedSparseResponse, Status>>;

    async fn embed_sparse_stream(
        &self,
        request: Request<Streaming<EmbedSparseRequest>>,
    ) -> Result<Response<Self::EmbedSparseStreamStream>, Status> {
        // Clone for move below
        let clone = self.clone();
        let function = |req: EmbedSparseRequest, permit: OwnedSemaphorePermit| async move {
            clone.embed_sparse_inner(req, permit).await
        };

        self.stream(request, function).await
    }

    #[instrument(skip_all)]
    async fn embed_all(
        &self,
        request: Request<EmbedAllRequest>,
    ) -> Result<Response<EmbedAllResponse>, Status> {
        let counter = metrics::counter!("te_request_count", "method" => "single");
        counter.increment(1);

        let permit = self
            .infer
            .try_acquire_permit()
            .map_err(ErrorResponse::from)?;

        let request = request.into_inner();
        let (response, metadata) = self.embed_all_inner(request, permit).await?;
        let headers = HeaderMap::from(metadata);

        let counter = metrics::counter!("te_request_count", "method" => "single");
        counter.increment(1);

        Ok(Response::from_parts(
            MetadataMap::from_headers(headers),
            response,
            Extensions::default(),
        ))
    }

    type EmbedAllStreamStream = UnboundedReceiverStream<Result<EmbedAllResponse, Status>>;

    #[instrument(skip_all)]
    async fn embed_all_stream(
        &self,
        request: Request<Streaming<EmbedAllRequest>>,
    ) -> Result<Response<Self::EmbedAllStreamStream>, Status> {
        // Clone for move below
        let clone = self.clone();
        let function = |req: EmbedAllRequest, permit: OwnedSemaphorePermit| async move {
            clone.embed_all_inner(req, permit).await
        };

        self.stream(request, function).await
    }
}

#[tonic::async_trait]
impl grpc::predict_server::Predict for TextEmbeddingsService {
    #[instrument(skip_all)]
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let counter = metrics::counter!("te_request_count", "method" => "single");
        counter.increment(1);

        let permit = self
            .infer
            .try_acquire_permit()
            .map_err(ErrorResponse::from)?;

        let request = request.into_inner();
        let truncation_direction = convert_truncation_direction(request.truncation_direction);
        let (response, metadata) = self
            .predict_inner(
                request.inputs,
                request.truncate,
                truncation_direction,
                request.raw_scores,
                permit,
            )
            .await?;
        let headers = HeaderMap::from(metadata);

        let counter = metrics::counter!("te_request_count", "method" => "single");
        counter.increment(1);

        Ok(Response::from_parts(
            MetadataMap::from_headers(headers),
            response,
            Extensions::default(),
        ))
    }

    async fn predict_pair(
        &self,
        request: Request<PredictPairRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let counter = metrics::counter!("te_request_count", "method" => "single");
        counter.increment(1);
        let request = request.into_inner();

        let mut inputs = request.inputs;

        let inputs = match inputs.len() {
            1 => EncodingInput::Single(inputs.pop().unwrap()),
            2 => EncodingInput::Dual(inputs.swap_remove(0), inputs.pop().unwrap()),
            _ => {
                return Err(Status::from(ErrorResponse {
                    error: format!(
                        "`inputs` must have a single or two elements. Given: {}",
                        inputs.len()
                    ),
                    error_type: ErrorType::Validation,
                }))
            }
        };

        let permit = self
            .infer
            .try_acquire_permit()
            .map_err(ErrorResponse::from)?;

        let truncation_direction = convert_truncation_direction(request.truncation_direction);
        let (response, metadata) = self
            .predict_inner(
                inputs,
                request.truncate,
                truncation_direction,
                request.raw_scores,
                permit,
            )
            .await?;
        let headers = HeaderMap::from(metadata);

        let counter = metrics::counter!("te_request_count", "method" => "single");
        counter.increment(1);

        Ok(Response::from_parts(
            MetadataMap::from_headers(headers),
            response,
            Extensions::default(),
        ))
    }

    type PredictStreamStream = UnboundedReceiverStream<Result<PredictResponse, Status>>;

    #[instrument(skip_all)]
    async fn predict_stream(
        &self,
        request: Request<Streaming<PredictRequest>>,
    ) -> Result<Response<Self::PredictStreamStream>, Status> {
        // Clone for move below
        let clone = self.clone();
        let function = |req: PredictRequest, permit: OwnedSemaphorePermit| async move {
            let truncation_direction = convert_truncation_direction(req.truncation_direction);
            clone
                .predict_inner(
                    req.inputs,
                    req.truncate,
                    truncation_direction,
                    req.raw_scores,
                    permit,
                )
                .await
        };

        self.stream(request, function).await
    }

    type PredictPairStreamStream = UnboundedReceiverStream<Result<PredictResponse, Status>>;

    async fn predict_pair_stream(
        &self,
        request: Request<Streaming<PredictPairRequest>>,
    ) -> Result<Response<Self::PredictPairStreamStream>, Status> {
        // Clone for move below
        let clone = self.clone();
        let function = |req: PredictPairRequest, permit: OwnedSemaphorePermit| async move {
            let mut inputs = req.inputs;

            let inputs = match inputs.len() {
                1 => EncodingInput::Single(inputs.pop().unwrap()),
                2 => EncodingInput::Dual(inputs.swap_remove(0), inputs.pop().unwrap()),
                _ => {
                    return Err(Status::from(ErrorResponse {
                        error: format!(
                            "`inputs` must have a single or two elements. Given: {}",
                            inputs.len()
                        ),
                        error_type: ErrorType::Validation,
                    }))
                }
            };

            let truncation_direction = convert_truncation_direction(req.truncation_direction);
            clone
                .predict_inner(
                    inputs,
                    req.truncate,
                    truncation_direction,
                    req.raw_scores,
                    permit,
                )
                .await
        };

        self.stream(request, function).await
    }
}

#[tonic::async_trait]
impl grpc::rerank_server::Rerank for TextEmbeddingsService {
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
    async fn rerank(
        &self,
        request: Request<RerankRequest>,
    ) -> Result<Response<RerankResponse>, Status> {
        let span = Span::current();
        let start_time = Instant::now();

        let request = request.into_inner();

        if request.texts.is_empty() {
            let message = "`texts` cannot be empty".to_string();
            tracing::error!("{message}");
            let err = ErrorResponse {
                error: message,
                error_type: ErrorType::Validation,
            };
            let counter = metrics::counter!("te_request_failure", "err" => "validation");
            counter.increment(1);
            Err(err)?;
        }

        match &self.info.model_type {
            ModelType::Classifier(_) => {
                let counter = metrics::counter!("te_request_failure", "err" => "model_type");
                counter.increment(1);
                let message = "model is not a re-ranker model".to_string();
                tracing::error!("{message}");
                Err(Status::new(Code::FailedPrecondition, message))
            }
            ModelType::Reranker(_) => Ok(()),
            ModelType::Embedding(_) => {
                let counter = metrics::counter!("te_request_failure", "err" => "model_type");
                counter.increment(1);
                let message = "model is not a classifier model".to_string();
                tracing::error!("{message}");
                Err(Status::new(Code::FailedPrecondition, message))
            }
        }?;

        // Closure for rerank
        let rerank_inner = move |query: String,
                                 text: String,
                                 truncate: bool,
                                 truncation_direction: tokenizers::TruncationDirection,
                                 raw_scores: bool,
                                 infer: Infer| async move {
            let permit = infer.acquire_permit().await;

            let response = infer
                .predict(
                    (query, text),
                    truncate,
                    truncation_direction,
                    raw_scores,
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

        let counter = metrics::counter!("te_request_count", "method" => "batch");
        counter.increment(1);

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
            let counter = metrics::counter!("te_request_failure", "err" => "batch_size");
            counter.increment(1);
            Err(err)?;
        }

        let mut futures = Vec::with_capacity(batch_size);
        let query_chars = request.query.chars().count();
        let mut total_compute_chars = query_chars * batch_size;
        let truncation_direction = convert_truncation_direction(request.truncation_direction);

        for text in &request.texts {
            total_compute_chars += text.chars().count();
            let local_infer = self.infer.clone();
            futures.push(rerank_inner(
                request.query.clone(),
                text.clone(),
                request.truncate,
                truncation_direction,
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

            let score = r.4;
            // Check that s is not NaN or the partial_cmp below will panic
            if score.is_nan() {
                Err(ErrorResponse {
                    error: "score is NaN".to_string(),
                    error_type: ErrorType::Backend,
                })?;
            }

            ranks.push(Rank {
                index: index as u32,
                text,
                score,
            })
        }

        // Reverse sort
        ranks.sort_by(|x, y| x.score.partial_cmp(&y.score).unwrap());
        ranks.reverse();

        let batch_size = batch_size as u64;

        let counter = metrics::counter!("te_request_success", "method" => "batch");
        counter.increment(1);

        let response_metadata = ResponseMetadata::new(
            total_compute_chars,
            total_compute_tokens,
            start_time,
            Duration::from_nanos(total_tokenization_time / batch_size),
            Duration::from_nanos(total_queue_time / batch_size),
            Duration::from_nanos(total_inference_time / batch_size),
        );
        response_metadata.record_span(&span);
        response_metadata.record_metrics();

        let message = RerankResponse {
            ranks,
            metadata: Some(grpc::Metadata::from(&response_metadata)),
        };

        let headers = HeaderMap::from(response_metadata);

        tracing::info!("Success");

        Ok(Response::from_parts(
            MetadataMap::from_headers(headers),
            message,
            Extensions::default(),
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
    async fn rerank_stream(
        &self,
        request: Request<Streaming<RerankStreamRequest>>,
    ) -> Result<Response<RerankResponse>, Status> {
        let span = Span::current();
        let start_time = Instant::now();

        // Check model type
        match &self.info.model_type {
            ModelType::Classifier(_) => {
                let counter = metrics::counter!("te_request_failure", "err" => "model_type");
                counter.increment(1);
                let message = "model is not a re-ranker model".to_string();
                tracing::error!("{message}");
                Err(Status::new(Code::FailedPrecondition, message))
            }
            ModelType::Reranker(_) => Ok(()),
            ModelType::Embedding(_) => {
                let counter = metrics::counter!("te_request_failure", "err" => "model_type");
                counter.increment(1);
                let message = "model is not a classifier model".to_string();
                tracing::error!("{message}");
                Err(Status::new(Code::FailedPrecondition, message))
            }
        }?;

        // Closure for rerank
        let rerank_inner = move |index: usize,
                                 query: String,
                                 text: String,
                                 truncate: bool,
                                 truncation_direction: tokenizers::TruncationDirection,
                                 raw_scores: bool,
                                 infer: Infer,
                                 permit: OwnedSemaphorePermit| async move {
            let response = infer
                .predict(
                    (query, text.clone()),
                    truncate,
                    truncation_direction,
                    raw_scores,
                    permit,
                )
                .await
                .map_err(ErrorResponse::from)?;

            let score = response.results[0];

            Ok::<(usize, usize, Duration, Duration, Duration, f32, String), ErrorResponse>((
                index,
                response.metadata.prompt_tokens,
                response.metadata.tokenization,
                response.metadata.queue,
                response.metadata.inference,
                score,
                text,
            ))
        };

        let counter = metrics::counter!("te_request_count", "method" => "batch");
        counter.increment(1);

        let mut request_stream = request.into_inner();

        // Create bounded channel to have an upper bound of spawned tasks
        // We will have at most `max_parallel_stream_requests` messages from this stream in the queue
        let (rerank_sender, mut rerank_receiver) = mpsc::channel::<(
            (
                usize,
                String,
                String,
                bool,
                tokenizers::TruncationDirection,
                bool,
            ),
            oneshot::Sender<
                Result<(usize, usize, Duration, Duration, Duration, f32, String), ErrorResponse>,
            >,
        )>(self.max_parallel_stream_requests);

        // Required for the async move below
        let local_infer = self.infer.clone();

        // Background task that uses the bounded channel
        tokio::spawn(async move {
            while let Some((
                (index, query, text, truncate, truncation_direction, raw_scores),
                mut sender,
            )) = rerank_receiver.recv().await
            {
                // Wait on permit before spawning the task to avoid creating more tasks than needed
                let permit = local_infer.acquire_permit().await;

                // Required for the async move below
                let task_infer = local_infer.clone();

                // Create async task for this specific input
                tokio::spawn(async move {
                    // Select on closed to cancel work if the stream was closed
                    tokio::select! {
                    result = rerank_inner(index, query, text, truncate, truncation_direction, raw_scores, task_infer, permit) => {
                        let _ = sender.send(result);
                    }
                    _ = sender.closed() => {}
                    }
                });
            }
        });

        let mut index = 0;
        let mut total_compute_chars = 0;

        // Set by first request
        let mut raw_scores = None;
        let mut return_text = None;

        // Intermediate channels
        // Required to keep the order of the requests
        let (intermediate_sender, mut intermediate_receiver) = mpsc::unbounded_channel();

        while let Some(request) = request_stream.next().await {
            let request = request?;

            // Create return channel
            let (result_sender, result_receiver) = oneshot::channel();
            // Push to intermediate channel and preserve ordering
            intermediate_sender
                .send(result_receiver)
                .expect("`intermediate_receiver` was dropped. This is a bug.");

            // Set `raw_scores` and `return_text` using the values in the first request
            if raw_scores.is_none() && return_text.is_none() {
                raw_scores = Some(request.raw_scores);
                return_text = Some(request.return_text);
            }

            total_compute_chars += request.query.chars().count();
            total_compute_chars += request.text.chars().count();

            let truncation_direction = convert_truncation_direction(request.truncation_direction);
            rerank_sender
                .send((
                    (
                        index,
                        request.query,
                        request.text,
                        request.truncate,
                        truncation_direction,
                        raw_scores.unwrap(),
                    ),
                    result_sender,
                ))
                .await
                .expect("`rerank_receiver` was dropped. This is a bug.");

            index += 1;
        }

        // Drop the sender to signal to the underlying task that we are done
        drop(rerank_sender);

        let batch_size = index;

        let mut ranks = Vec::with_capacity(batch_size);
        let mut total_tokenization_time = 0;
        let mut total_queue_time = 0;
        let mut total_inference_time = 0;
        let mut total_compute_tokens = 0;

        // Iterate on result stream
        while let Some(result_receiver) = intermediate_receiver.recv().await {
            let r = result_receiver
                .await
                .expect("`result_sender` was dropped. This is a bug.")?;

            total_compute_tokens += r.1;
            total_tokenization_time += r.2.as_nanos() as u64;
            total_queue_time += r.3.as_nanos() as u64;
            total_inference_time += r.4.as_nanos() as u64;
            let text = if return_text.unwrap() {
                Some(r.6)
            } else {
                None
            };

            let score = r.5;
            // Check that s is not NaN or the partial_cmp below will panic
            if score.is_nan() {
                Err(ErrorResponse {
                    error: "score is NaN".to_string(),
                    error_type: ErrorType::Backend,
                })?;
            }

            ranks.push(Rank {
                index: r.0 as u32,
                text,
                score,
            })
        }

        // Check that the outputs have the correct size
        if ranks.len() < batch_size {
            let message = "rerank results is missing values".to_string();
            tracing::error!("{message}");
            let err = ErrorResponse {
                error: message,
                error_type: ErrorType::Backend,
            };
            let counter = metrics::counter!("te_request_failure", "err" => "missing_values");
            counter.increment(1);
            Err(err)?;
        }

        // Reverse sort
        ranks.sort_by(|x, y| x.score.partial_cmp(&y.score).unwrap());
        ranks.reverse();

        let batch_size = batch_size as u64;

        let counter = metrics::counter!("te_request_success", "method" => "batch");
        counter.increment(1);

        let response_metadata = ResponseMetadata::new(
            total_compute_chars,
            total_compute_tokens,
            start_time,
            Duration::from_nanos(total_tokenization_time / batch_size),
            Duration::from_nanos(total_queue_time / batch_size),
            Duration::from_nanos(total_inference_time / batch_size),
        );
        response_metadata.record_span(&span);
        response_metadata.record_metrics();

        let message = RerankResponse {
            ranks,
            metadata: Some(grpc::Metadata::from(&response_metadata)),
        };

        let headers = HeaderMap::from(response_metadata);

        tracing::info!("Success");

        Ok(Response::from_parts(
            MetadataMap::from_headers(headers),
            message,
            Extensions::default(),
        ))
    }
}

#[tonic::async_trait]
impl grpc::tokenize_server::Tokenize for TextEmbeddingsService {
    async fn tokenize(
        &self,
        request: Request<EncodeRequest>,
    ) -> Result<Response<EncodeResponse>, Status> {
        let request = request.into_inner();
        let tokens = self.tokenize_inner(request).await?;
        Ok(Response::new(tokens))
    }

    type TokenizeStreamStream = UnboundedReceiverStream<Result<EncodeResponse, Status>>;

    async fn tokenize_stream(
        &self,
        request: Request<Streaming<EncodeRequest>>,
    ) -> Result<Response<Self::TokenizeStreamStream>, Status> {
        // Clone for move below
        let clone = self.clone();
        let function = |req: EncodeRequest| async move { clone.tokenize_inner(req).await };

        self.stream_no_permit(request, function).await
    }

    async fn decode(
        &self,
        request: Request<DecodeRequest>,
    ) -> Result<Response<DecodeResponse>, Status> {
        let request = request.into_inner();
        let tokens = self.decode_inner(request).await?;
        Ok(Response::new(tokens))
    }

    type DecodeStreamStream = UnboundedReceiverStream<Result<DecodeResponse, Status>>;

    async fn decode_stream(
        &self,
        request: Request<Streaming<DecodeRequest>>,
    ) -> Result<Response<Self::DecodeStreamStream>, Status> {
        // Clone for move below
        let clone = self.clone();
        let function = |req: DecodeRequest| async move { clone.decode_inner(req).await };

        self.stream_no_permit(request, function).await
    }
}

pub async fn run(
    infer: Infer,
    info: Info,
    addr: SocketAddr,
    prom_builder: PrometheusBuilder,
    api_key: Option<String>,
) -> Result<(), anyhow::Error> {
    prom_builder.install()?;
    tracing::info!("Serving Prometheus metrics: 0.0.0.0:9000");

    // Liveness service
    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    // Info is always serving
    health_reporter
        .set_serving::<grpc::InfoServer<TextEmbeddingsService>>()
        .await;
    // Tokenize is always serving
    health_reporter
        .set_serving::<grpc::TokenizeServer<TextEmbeddingsService>>()
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
    let service = TextEmbeddingsService::new(infer, info);

    // Create gRPC server
    let server = if let Some(api_key) = api_key {
        let mut prefix = "Bearer ".to_string();
        prefix.push_str(&api_key);

        // Leak to allow FnMut
        let api_key: &'static str = prefix.leak();

        let auth = move |req: Request<()>| -> Result<Request<()>, Status> {
            match req.metadata().get("authorization") {
                Some(t) if t == api_key => Ok(req),
                _ => Err(Status::unauthenticated("No valid auth token")),
            }
        };

        Server::builder()
            .add_service(health_service)
            .add_service(reflection_service)
            .add_service(grpc::InfoServer::with_interceptor(service.clone(), auth))
            .add_service(grpc::TokenizeServer::with_interceptor(
                service.clone(),
                auth,
            ))
            .add_service(grpc::EmbedServer::with_interceptor(service.clone(), auth))
            .add_service(grpc::PredictServer::with_interceptor(service.clone(), auth))
            .add_service(grpc::RerankServer::with_interceptor(service, auth))
            .serve_with_shutdown(addr, shutdown::shutdown_signal())
    } else {
        Server::builder()
            .add_service(health_service)
            .add_service(reflection_service)
            .add_service(grpc::InfoServer::new(service.clone()))
            .add_service(grpc::TokenizeServer::new(service.clone()))
            .add_service(grpc::EmbedServer::new(service.clone()))
            .add_service(grpc::PredictServer::new(service.clone()))
            .add_service(grpc::RerankServer::new(service))
            .serve_with_shutdown(addr, shutdown::shutdown_signal())
    };

    tracing::info!("Starting gRPC server: {}", &addr);
    tracing::info!("Ready");
    server.await?;

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
            ErrorType::Empty => Code::InvalidArgument,
        };

        Status::new(code, value.error)
    }
}

fn convert_truncation_direction(value: i32) -> tokenizers::TruncationDirection {
    match TruncationDirection::try_from(value).expect("Unexpected enum value") {
        TruncationDirection::Right => tokenizers::TruncationDirection::Right,
        TruncationDirection::Left => tokenizers::TruncationDirection::Left,
    }
}
