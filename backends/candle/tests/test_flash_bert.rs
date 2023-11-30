#![allow(dead_code, unused_imports)]
mod common;

use crate::common::SnapshotScores;
use anyhow::Result;
use common::{batch, download_artifacts, load_tokenizer, relative_matcher};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
#[cfg(all(
    feature = "cuda",
    any(feature = "flash-attn", feature = "flash-attn-v1")
))]
fn test_flash_mini() -> Result<()> {
    let model_root = download_artifacts("sentence-transformers/all-MiniLM-L6-v2")?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        model_root,
        "float16".to_string(),
        ModelType::Embedding(Pool::Mean),
    )?;

    let input_batch = batch(vec![
        tokenizer.encode("What is Deep Learning?", true).unwrap(),
        tokenizer.encode("Deep Learning is...", true).unwrap(),
        tokenizer.encode("What is Deep Learning?", true).unwrap(),
    ]);

    let matcher = relative_matcher();

    let embeddings_batch = SnapshotScores::from(backend.embed(input_batch)?);
    insta::assert_yaml_snapshot!("mini_batch", embeddings_batch, &matcher);

    let input_single = batch(vec![tokenizer
        .encode("What is Deep Learning?", true)
        .unwrap()]);

    let embeddings_single = SnapshotScores::from(backend.embed(input_single)?);

    insta::assert_yaml_snapshot!("mini_single", embeddings_single, &matcher);
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
fn test_flash_emotions() -> Result<()> {
    let model_root = download_artifacts("SamLowe/roberta-base-go_emotions")?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(model_root, "float16".to_string(), ModelType::Classifier)?;

    let input_batch = batch(vec![
        tokenizer.encode("I like you.", true).unwrap(),
        tokenizer
            .encode("I am not having a great day.", true)
            .unwrap(),
        tokenizer.encode("I like you.", true).unwrap(),
    ]);

    let matcher = relative_matcher();

    let predictions_batch = SnapshotScores::from(backend.predict(input_batch)?);
    insta::assert_yaml_snapshot!("emotions_batch", predictions_batch, &matcher);

    let input_single = batch(vec![tokenizer.encode("I like you.", true).unwrap()]);

    let predictions_single = SnapshotScores::from(backend.predict(input_single)?);

    insta::assert_yaml_snapshot!("emotions_single", predictions_single, &matcher);
    assert_eq!(predictions_batch[0], predictions_single[0]);
    assert_eq!(predictions_batch[2], predictions_single[0]);

    Ok(())
}
