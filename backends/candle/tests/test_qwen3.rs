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
    let (model_root, _) = download_artifacts("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Classifier,
        None,
    )?;

    let input_single = batch(
        vec![tokenizer
            .encode(
                "What is Deep Learning?",
                true,
            )
            .unwrap()],
        [0].to_vec(),
        vec![],
    );

    let predictions: Vec<Vec<f32>> = backend
        .predict(input_single)?
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    let predictions_single = SnapshotScores::from(predictions);

    let matcher = relative_matcher();
    insta::assert_yaml_snapshot!("qwen3_reranker_single", predictions_single, &matcher);

    Ok(())
}
