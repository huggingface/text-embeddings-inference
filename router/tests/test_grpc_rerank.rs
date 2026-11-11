#![cfg(feature = "grpc")]

use anyhow::Result;
use std::time::Duration;
use text_embeddings_backend::{DType, OtlpProtocol};
use text_embeddings_router::run;
use tonic::codec::ProstCodec;
use tonic::codegen::http::uri::PathAndQuery;
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Request, Status};

// Subset of the `tei.v1` messages used by `Rerank/RerankStream`
#[derive(Clone, PartialEq, prost::Message)]
struct RerankStreamRequest {
    #[prost(string, tag = "1")]
    query: String,
    #[prost(string, tag = "2")]
    text: String,
}

#[derive(Clone, PartialEq, prost::Message)]
struct Rank {
    #[prost(uint32, tag = "1")]
    index: u32,
    #[prost(float, tag = "3")]
    score: f32,
}

#[derive(Clone, PartialEq, prost::Message)]
struct RerankResponse {
    #[prost(message, repeated, tag = "1")]
    ranks: Vec<Rank>,
}

async fn start_grpc_server(port: u16, prometheus_port: u16) -> Result<Channel> {
    let server_task = tokio::spawn(run(
        "cross-encoder/ms-marco-MiniLM-L2-v2".to_string(),
        Some("1b5cd67b15209f24824c50370e0397743aa9b787".to_string()),
        Some(1),
        Some(DType::Float32),
        "cross-encoder/ms-marco-MiniLM-L2-v2".to_string(),
        None,
        4,
        1024,
        None,
        32,
        false,
        None,
        None,
        None,
        None,
        Some("127.0.0.1".to_string()),
        port,
        None,
        None,
        None,
        2_000_000,
        None,
        None,
        "text-embeddings-inference.server".to_owned(),
        OtlpProtocol::Grpc,
        Some(prometheus_port),
        None,
    ));

    // The server only listens once the model is loaded
    let endpoint = Endpoint::from_shared(format!("http://127.0.0.1:{port}"))?;
    let connect = async {
        loop {
            if let Ok(channel) = endpoint.connect().await {
                return channel;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    };

    tokio::select! {
        result = server_task => anyhow::bail!("server stopped: {:?}", result),
        channel = tokio::time::timeout(Duration::from_secs(60), connect) => Ok(channel?),
    }
}

async fn rerank_stream(
    channel: Channel,
    requests: Vec<RerankStreamRequest>,
) -> Result<RerankResponse, Status> {
    let mut client = tonic::client::Grpc::new(channel);
    client
        .ready()
        .await
        .map_err(|err| Status::unknown(err.to_string()))?;
    let codec: ProstCodec<RerankStreamRequest, RerankResponse> = ProstCodec::default();
    let response = client
        .client_streaming(
            Request::new(tokio_stream::iter(requests)),
            PathAndQuery::from_static("/tei.v1.Rerank/RerankStream"),
            codec,
        )
        .await?;
    Ok(response.into_inner())
}

#[tokio::test]
#[serial_test::serial]
async fn test_rerank_stream() -> Result<()> {
    let channel = start_grpc_server(8093, 9003).await?;

    let query = "What is deep learning?".to_string();
    let requests = [
        "Deep learning is a subset of machine learning that uses neural networks.",
        "A kitten plays with yarn in the garden.",
    ]
    .into_iter()
    .map(|text| RerankStreamRequest {
        query: query.clone(),
        text: text.to_string(),
    })
    .collect();

    // The response is sent once the client closes the stream
    let response = tokio::time::timeout(
        Duration::from_secs(30),
        rerank_stream(channel.clone(), requests),
    )
    .await
    .expect("`RerankStream` did not respond")?;

    assert_eq!(response.ranks.len(), 2);
    assert_eq!(response.ranks[0].index, 0);
    assert_eq!(response.ranks[1].index, 1);
    assert!(response.ranks[0].score > response.ranks[1].score);

    // An empty stream is rejected
    let status = tokio::time::timeout(Duration::from_secs(30), rerank_stream(channel, vec![]))
        .await
        .expect("`RerankStream` did not respond")
        .unwrap_err();
    assert_eq!(status.code(), Code::InvalidArgument);

    Ok(())
}
