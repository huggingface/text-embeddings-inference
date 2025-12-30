mod common;

use crate::common::{start_server, Score};
use anyhow::Result;
use insta::internals::YamlMatcher;
use serde_json::json;
use text_embeddings_backend::DType;

#[tokio::test]
#[cfg(feature = "http")]
async fn test_mrl_embeddings() -> Result<()> {
    start_server(
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        None,
        DType::Float32,
    )
    .await?;

    let request = json!({
        "inputs": "test",
        "dimensions": 128,
    });

    let client = reqwest::Client::new();
    let res = client
        .post("http://0.0.0.0:8090/embed")
        .json(&request)
        .send()
        .await?;

    let embeddings_single = res.json::<Vec<Vec<Score>>>().await?;
    let matcher = YamlMatcher::<Vec<Vec<Score>>>::new();
    insta::assert_yaml_snapshot!("mrl_embeddings_single", embeddings_single, &matcher);

    let request = json!({
        "inputs": vec!["test", "test", "test"],
        "dimensions": 128,
    });

    let client = reqwest::Client::new();
    let res = client
        .post("http://0.0.0.0:8090/embed")
        .json(&request)
        .send()
        .await?;

    let embeddings_batch = res.json::<Vec<Vec<Score>>>().await?;
    insta::assert_yaml_snapshot!("mrl_embeddings_batch", embeddings_batch, &matcher);

    for embeddings in &embeddings_batch {
        assert_eq!(embeddings, &embeddings_single[0]);
    }

    Ok(())
}
