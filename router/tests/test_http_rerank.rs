mod common;

use crate::common::start_server_with_ports;
use anyhow::Result;
use insta::internals::YamlMatcher;
use serde::{Deserialize, Serialize};
use serde_json::json;
use text_embeddings_backend::DType;

#[derive(Deserialize, Debug)]
pub struct Rank {
    index: usize,
    score: f32,
    text: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SnapshotRank {
    index: usize,
    text: String,
}

#[tokio::test]
#[serial_test::serial]
#[cfg(feature = "http")]
async fn test_rerank() -> Result<()> {
    let port = 8091;
    start_server_with_ports(
        "cross-encoder/ms-marco-MiniLM-L2-v2".to_string(),
        None,
        DType::Float32,
        port,
        9001,
    )
    .await?;

    let request = json!({
        "query": "What is deep learning?",
        "texts": [
            "Deep learning is a subset of machine learning that uses neural networks.",
            "A kitten plays with yarn in the garden."
        ],
        "return_text": true
    });

    let client = reqwest::Client::new();
    let res = client
        .post(format!("http://0.0.0.0:{port}/rerank"))
        .json(&request)
        .send()
        .await?
        .error_for_status()?;

    let ranks = res.json::<Vec<Rank>>().await?;

    assert_eq!(ranks.len(), 2);
    assert_eq!(ranks[0].index, 0);
    assert_eq!(
        ranks[0].text,
        "Deep learning is a subset of machine learning that uses neural networks."
    );
    assert!(ranks[0].score.is_finite());
    assert!(ranks[1].score.is_finite());
    assert!(ranks[0].score > ranks[1].score);

    let snapshot_ranks = ranks
        .into_iter()
        .map(|rank| SnapshotRank {
            index: rank.index,
            text: rank.text,
        })
        .collect::<Vec<_>>();
    let matcher = YamlMatcher::<Vec<SnapshotRank>>::new();
    insta::assert_yaml_snapshot!("ranks", snapshot_ranks, &matcher);

    Ok(())
}
