/// Single shard Client
use crate::pb::embedding::v1::embedding_service_client::EmbeddingServiceClient;
use crate::pb::embedding::v1::*;
use crate::Result;
use grpc_metadata::InjectTelemetryContext;
use tonic::transport::{Channel, Uri};
use tracing::instrument;

/// Text Generation Inference gRPC client
#[derive(Debug, Clone)]
pub struct Client {
    stub: EmbeddingServiceClient<Channel>,
}

impl Client {
    /// Returns a client connected to the given url
    pub async fn connect(uri: Uri) -> Result<Self> {
        let channel = Channel::builder(uri).connect().await?;

        Ok(Self {
            stub: EmbeddingServiceClient::new(channel),
        })
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let channel = Channel::from_shared("http://[::]:50051".to_string())
            .unwrap()
            .connect_with_connector(tower::service_fn(move |_: Uri| {
                tokio::net::UnixStream::connect(path.clone())
            }))
            .await?;

        Ok(Self {
            stub: EmbeddingServiceClient::new(channel),
        })
    }

    /// Get backend health
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let request = tonic::Request::new(HealthRequest {}).inject_context();
        let response = self.stub.health(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn embed(
        &mut self,
        input_ids: Vec<u32>,
        token_type_ids: Vec<u32>,
        position_ids: Vec<u32>,
        cu_seq_lengths: Vec<u32>,
        max_length: u32,
    ) -> Result<Vec<Embedding>> {
        let request = tonic::Request::new(EmbedRequest {
            input_ids,
            token_type_ids,
            position_ids,
            max_length,
            cu_seq_lengths,
        })
        .inject_context();
        let response = self.stub.embed(request).await?.into_inner();
        Ok(response.embeddings)
    }

    #[instrument(skip_all)]
    pub async fn predict(
        &mut self,
        input_ids: Vec<u32>,
        token_type_ids: Vec<u32>,
        position_ids: Vec<u32>,
        cu_seq_lengths: Vec<u32>,
        max_length: u32,
    ) -> Result<Vec<Score>> {
        let request = tonic::Request::new(EmbedRequest {
            input_ids,
            token_type_ids,
            position_ids,
            max_length,
            cu_seq_lengths,
        })
        .inject_context();
        let response = self.stub.predict(request).await?.into_inner();
        Ok(response.scores)
    }
}
