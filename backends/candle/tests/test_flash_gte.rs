#![allow(dead_code, unused_imports)]
mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings, SnapshotScores};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer, relative_matcher};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
fn test_flash_gte() -> Result<()> {
    let model_root = download_artifacts("Alibaba-NLP/gte-base-en-v1.5", None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float16".to_string(),
        ModelType::Embedding(Pool::Cls),
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
    insta::assert_yaml_snapshot!("gte_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("gte_single", embeddings_single, &matcher);
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}

#[test]
#[serial_test::serial]
#[cfg(all(
    feature = "cuda",
    any(feature = "flash-attn", feature = "flash-attn-v1")
))]
fn test_flash_gte_classification() -> Result<()> {
    let model_root = download_artifacts("Alibaba-NLP/gte-multilingual-reranker-base", None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(&model_root, "float16".to_string(), ModelType::Classifier)?;

    let input_single = batch(
        vec![tokenizer
            .encode(("What is Deep Learning?", "Deep Learning is not..."), true)
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
    insta::assert_yaml_snapshot!("gte_classification_single", predictions_single, &matcher);

    Ok(())
}
