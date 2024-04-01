mod common;

use crate::common::{start_server, Score};
use anyhow::Result;
use insta::internals::YamlMatcher;
use serde_json::json;
use text_embeddings_backend::DType;

#[tokio::test]
#[cfg(feature = "http")]
async fn test_embeddings() -> Result<()> {
    start_server(
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        None,
        DType::Float32,
    )
    .await?;

    let request = json!({
        "inputs": "test"
    });
    let client = reqwest::Client::new();
    let res = client
        .post("http://0.0.0.0:8090/embed")
        .json(&request)
        .send()
        .await?;

    let embeddings_single = res.json::<Vec<Vec<Score>>>().await?;
    let matcher = YamlMatcher::<Vec<Vec<Score>>>::new();
    insta::assert_yaml_snapshot!("embeddings_single", embeddings_single, &matcher);

    let test_tokens = vec![[101, 3231, 102]]; // tokenized "test"
    let request = json!({"inputs": &test_tokens});
    let res = client
        .post("http://0.0.0.0:8090/embed")
        .json(&request)
        .send()
        .await?;

    let embeddings_single = res.json::<Vec<Vec<Score>>>().await?;
    let matcher = YamlMatcher::<Vec<Vec<Score>>>::new();
    insta::assert_yaml_snapshot!("embeddings_single", embeddings_single, &matcher);

    let request = json!({
        "inputs": vec!["test", "test", "test", "test", "test"],
    });

    let client = reqwest::Client::new();
    let res = client
        .post("http://0.0.0.0:8090/embed")
        .json(&request)
        .send()
        .await?;
    let embeddings_batch = res.json::<Vec<Vec<Score>>>().await?;
    insta::assert_yaml_snapshot!("embeddings_batch", embeddings_batch, &matcher);
    for embeddings in &embeddings_batch {
        assert_eq!(embeddings, &embeddings_single[0]);
    }

    let request =
        json!({"inputs": &test_tokens.repeat(request["inputs"].as_array().unwrap().len())});
    let res = client
        .post("http://0.0.0.0:8090/embed")
        .json(&request)
        .send()
        .await?;

    let embeddings_batch = res.json::<Vec<Vec<Score>>>().await?;
    insta::assert_yaml_snapshot!("embeddings_batch", embeddings_batch, &matcher);
    for embeddings in &embeddings_batch {
        assert_eq!(embeddings, &embeddings_single[0]);
    }

    let request = json!({
        "inputs": "test"
    });

    let client = reqwest::Client::new();
    let res = client
        .post("http://0.0.0.0:8090/embed_all")
        .json(&request)
        .send()
        .await?;

    let embeddings_raw = res.json::<Vec<Vec<Vec<Score>>>>().await?;
    let matcher = YamlMatcher::<Vec<Vec<Vec<Score>>>>::new();
    insta::assert_yaml_snapshot!("embeddings_raw", embeddings_raw, &matcher);

    Ok(())
}
