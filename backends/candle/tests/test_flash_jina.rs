#![allow(dead_code, unused_imports)]
mod common;

use crate::common::{sort_embeddings, SnapshotScores};
use anyhow::Result;
use common::{batch, download_artifacts, load_tokenizer, relative_matcher};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
fn test_flash_jina_small() -> Result<()> {
    let model_root = download_artifacts("jinaai/jina-embeddings-v2-small-en", None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        model_root,
        "float16".to_string(),
        ModelType::Embedding(Pool::Mean),
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

    let matcher = relative_matcher();

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    let embeddings_batch = SnapshotScores::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("jina_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotScores::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("jina_single", embeddings_single, &matcher);
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}
