mod common;

use crate::common::{start_server, Score};
use anyhow::Result;
use insta::internals::YamlMatcher;
use serde::{Deserialize, Serialize};
use serde_json::json;
use text_embeddings_backend::DType;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct SnapshotPrediction {
    score: Score,
    label: String,
}

#[tokio::test]
#[cfg(feature = "http")]
async fn test_predict() -> Result<()> {
    let model_id = if cfg!(feature = "ort") {
        "SamLowe/roberta-base-go_emotions-onnx"
    } else {
        "SamLowe/roberta-base-go_emotions"
    };

    start_server(model_id.to_string(), None, DType::Float32).await?;

    let request = json!({
        "inputs": "test"
    });

    let client = reqwest::Client::new();
    let res = client
        .post("http://0.0.0.0:8090/predict")
        .json(&request)
        .send()
        .await?;

    let predictions_single = res.json::<Vec<SnapshotPrediction>>().await?;
    let matcher = YamlMatcher::<Vec<SnapshotPrediction>>::new();
    insta::assert_yaml_snapshot!("predictions_single", predictions_single, &matcher);

    let request = json!({
        "inputs": vec![
            vec!["test"],
            vec!["test"],
            vec!["test"],
            vec!["test"],
            vec!["test"],
        ],
    });

    let client = reqwest::Client::new();
    let res = client
        .post("http://0.0.0.0:8090/predict")
        .json(&request)
        .send()
        .await?;

    let predictions_batch = res.json::<Vec<Vec<SnapshotPrediction>>>().await?;
    let matcher = YamlMatcher::<Vec<Vec<SnapshotPrediction>>>::new();
    insta::assert_yaml_snapshot!("predictions_batch", predictions_batch, &matcher);

    for predictions in &predictions_batch {
        assert_eq!(predictions, &predictions_single);
    }

    Ok(())
}
