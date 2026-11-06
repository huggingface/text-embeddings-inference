#![allow(dead_code, unused_imports)]
mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
fn test_flash_embedding_gemma2() -> Result<()> {
    let (model_root, _) = download_artifacts("google/embeddinggemma-2", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float16".to_string(),
        ModelType::Embedding(Pool::Mean),
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
    insta::assert_yaml_snapshot!("embedding_gemma2_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("embedding_gemma2_single", embeddings_single, &matcher);
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    // NOTE: Inputs longer than the bidirectional sliding window (i.e. `2 * 512 + 1` tokens) are
    // batched with a short one, so that both the Flash Attention window and the padded attention
    // of the full attention layers are exercised
    let long_input = "Deep Learning is a subset of Machine Learning. ".repeat(150);
    let input_long_batch = batch(
        vec![
            tokenizer.encode(long_input.as_str(), true).unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
        ],
        [0, 1].to_vec(),
        vec![],
    );
    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_long_batch)?);
    let embeddings_long_batch = SnapshotEmbeddings::from(pooled_embeddings);

    let input_long_single = batch(
        vec![tokenizer.encode(long_input.as_str(), true).unwrap()],
        [0].to_vec(),
        vec![],
    );
    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_long_single)?);
    let embeddings_long_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!(
        "embedding_gemma2_long_single",
        embeddings_long_single,
        &matcher
    );
    assert_eq!(embeddings_long_batch[0], embeddings_long_single[0]);
    assert_eq!(embeddings_long_batch[1], embeddings_single[0]);

    Ok(())
}
