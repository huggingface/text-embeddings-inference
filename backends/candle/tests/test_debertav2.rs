mod common;

use anyhow::Result;

use crate::common::{batch, download_artifacts, load_tokenizer, relative_matcher, SnapshotScores};

use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType};

#[test]
fn test_debertav2_rerank() -> Result<()> {
    let (model_root, _) = download_artifacts("mixedbread-ai/mxbai-rerank-xsmall-v1", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Classifier,
        None,
    )?;

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let predictions: Vec<Vec<f32>> = backend.predict(input_single)?.into_values().collect();

    let predictions = SnapshotScores::from(predictions);
    insta::assert_yaml_snapshot!(
        "debertav2_reranker_single",
        predictions,
        &relative_matcher()
    );

    Ok(())
}
