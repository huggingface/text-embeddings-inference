mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings, SnapshotScores};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer, relative_matcher};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
fn test_qwen3() -> Result<()> {
    let (model_root, _) = download_artifacts("Qwen/Qwen3-Embedding-0.6B", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::LastToken),
        None,
    )?;

    let input_batch = batch(
        vec![
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            tokenizer.encode("Deep Learning is...", true).unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
        ],
        [0, 1, 2].to_vec(),
        vec![],
    );

    let matcher = cosine_matcher();

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    let embeddings_batch = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("qwen3_cpu_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("qwen3_cpu_single", embeddings_single, &matcher);
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_qwen3_reranker() -> Result<()> {
    let (model_root, _) = download_artifacts("Qwen/Qwen3-Reranker-0.6B", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Classifier,
        None,
    )?;

    let input_single = batch(
        vec![tokenizer
            .encode(("What is Deep Learning?", "Deep Learning is not..."), true)
            .unwrap()],
        [0].to_vec(),
        vec![],
    );

    let predictions: Vec<Vec<f32>> = backend.predict(input_single)?.into_values().collect();
    let predictions_single = SnapshotScores::from(predictions);

    let matcher = relative_matcher();
    insta::assert_yaml_snapshot!("qwen3_reranker_single", predictions_single, &matcher);

    // Batched reranking. Positions 0 and 2 hold the SHORT pair (identical to `input_single`);
    // position 1 holds a longer, clearly-distinct document that sets the batch `max_length`, so
    // the short pair at 0/2 is actually left-padded. This exercises the left-padded, multi-row
    // last-token pooling + classifier path (batch_size > 1) that the single-input test does not
    // cover, and lets us assert that padding reproduces the unpadded single-input score.
    let input_batch = batch(
        vec![
            tokenizer
                .encode(("What is Deep Learning?", "Deep Learning is not..."), true)
                .unwrap(),
            tokenizer
                .encode(
                    (
                        "What is Deep Learning?",
                        "The weather in Paris is pleasant and sunny throughout the afternoon.",
                    ),
                    true,
                )
                .unwrap(),
            tokenizer
                .encode(("What is Deep Learning?", "Deep Learning is not..."), true)
                .unwrap(),
        ],
        [0, 1, 2].to_vec(),
        vec![],
    );

    // `predict()` keys the returned map by the 0..n output-row index (not by `pooled_indices`),
    // so sorting by key recovers request order here because `pooled_indices` is [0, 1, 2].
    let mut predictions: Vec<(usize, Vec<f32>)> =
        backend.predict(input_batch)?.into_iter().collect();
    predictions.sort_by_key(|(i, _)| *i);
    let predictions_batch =
        SnapshotScores::from(predictions.into_iter().map(|(_, v)| v).collect::<Vec<_>>());

    // Identical (left-padded) pairs -> identical scores (positions 0 and 2).
    assert_eq!(predictions_batch[0], predictions_batch[2]);
    // The left-padded batch score reproduces the unpadded single-input score for the same pair.
    assert_eq!(predictions_batch[0], predictions_single[0]);
    // The distinct (longer, off-topic) document must NOT collapse to the same score; guards
    // against a pooling/indexing regression that returns one row for every request.
    assert_ne!(predictions_batch[1], predictions_batch[0]);

    Ok(())
}
