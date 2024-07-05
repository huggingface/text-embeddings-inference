mod common;

use crate::common::{start_server, Score};
use anyhow::Result;
use insta::internals::YamlMatcher;
use serde::{Deserialize, Serialize};
use serde_json::json;
use text_embeddings_backend::DType;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct SnapshotRank {
    index: usize,
    score: Score,
    text: String,
}

#[tokio::test]
#[cfg(feature = "http")]
async fn test_rerank() -> Result<()> {
    start_server("BAAI/bge-reranker-base".to_string(), None, DType::Float32).await?;

    let request = json!({
        "query": "test",
        "texts": vec!["test", "other", "test"],
        "return_text": true
    });

    let client = reqwest::Client::new();
    let res = client
        .post("http://0.0.0.0:8090/rerank")
        .json(&request)
        .send()
        .await?;

    let ranks = res.json::<Vec<SnapshotRank>>().await?;
    let matcher = YamlMatcher::<Vec<SnapshotRank>>::new();
    insta::assert_yaml_snapshot!("ranks", ranks, &matcher);

    assert_eq!(ranks[0].index, 2);
    assert_eq!(ranks[1].index, 0);
    assert_eq!(ranks[0].score, ranks[1].score);

    Ok(())
}
