mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
fn test_stella_en_400m_v5_default_dense() -> Result<()> {
    let (model_root, dense_paths) = download_artifacts("dunzhang/stella_en_400M_v5", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        dense_paths, // This will default to `2_Dense_1024/` as defined in `modules.json`
        0,
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
    insta::assert_yaml_snapshot!(
        "stella_en_400m_v5_default_dense_batch",
        embeddings_batch,
        &matcher
    );

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!(
        "stella_en_400m_v5_default_dense_single",
        embeddings_single,
        &matcher
    );
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_stella_en_400m_v5_dense_768() -> Result<()> {
    let (model_root, dense_paths) = download_artifacts(
        "dunzhang/stella_en_400M_v5",
        None,
        Some("2_Dense_768"),
    )?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        dense_paths,
        0,
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

    // Check that embeddings have 768 dimensions
    assert!(!pooled_embeddings.is_empty());
    assert_eq!(pooled_embeddings[0].len(), 768);
    assert_eq!(pooled_embeddings[1].len(), 768);
    assert_eq!(pooled_embeddings[2].len(), 768);

    let embeddings_batch = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!(
        "stella_en_400m_v5_dense_768_batch",
        embeddings_batch,
        &matcher
    );

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);

    // Check that single embedding also has 768 dimensions
    assert!(!pooled_embeddings.is_empty());
    assert_eq!(pooled_embeddings[0].len(), 768);

    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!(
        "stella_en_400m_v5_dense_768_single",
        embeddings_single,
        &matcher
    );
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}
