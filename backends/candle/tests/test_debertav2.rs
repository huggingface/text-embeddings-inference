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

#[test]
fn test_debertav2_rerank_batch_matches_single() -> Result<()> {
    let (model_root, _) = download_artifacts("mixedbread-ai/mxbai-rerank-xsmall-v1", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Classifier,
        None,
    )?;

    // Pairs of very different lengths, so the short ones are padded inside the batch.
    let pairs = [
        ("What is Deep Learning?", "Definitions"),
        (
            "What is Deep Learning?",
            "Deep Learning is a subset of machine learning that uses neural networks with many \
             layers to learn representations of data. It is used for computer vision, speech \
             recognition, natural language processing and many other tasks, and it typically \
             needs large datasets and a lot of compute to train well.",
        ),
        ("What is Deep Learning?", "Deep Learning is not..."),
    ];
    let encodings: Vec<_> = pairs
        .iter()
        .map(|pair| tokenizer.encode(*pair, true).unwrap())
        .collect();

    let batched = backend.predict(batch(encodings.clone(), vec![0, 1, 2], vec![]))?;
    for (i, encoding) in encodings.into_iter().enumerate() {
        let single = backend.predict(batch(vec![encoding], vec![0], vec![]))?;
        for (s, b) in single[&0].iter().zip(&batched[&i]) {
            assert!(
                (s - b).abs() < 1e-3,
                "pair {i}: score {b} in a padded batch, {s} alone"
            );
        }
    }

    Ok(())
}
