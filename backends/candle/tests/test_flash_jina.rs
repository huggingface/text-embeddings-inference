#![allow(dead_code, unused_imports)]
mod common;

use crate::common::SnapshotScores;
use anyhow::Result;
use common::{batch, download_artifacts, load_tokenizer, relative_matcher};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
fn test_flash_jina_small() -> Result<()> {
    let model_root = download_artifacts("jinaai/jina-embeddings-v2-small-en")?;
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
    insta::assert_yaml_snapshot!("jina_batch", embeddings_batch, &matcher);

    let input_single = batch(vec![tokenizer
        .encode("What is Deep Learning?", true)
        .unwrap()]);

    let embeddings_single = SnapshotScores::from(backend.embed(input_single)?);

    insta::assert_yaml_snapshot!("jina_single", embeddings_single, &matcher);
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}
