mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings};
use anyhow::Result;
use common::{
    batch, cosine_matcher, download_artifacts, download_dense_modules, get_api_repo, load_tokenizer,
};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
fn test_stella_en_400m_v5_default_dense() -> Result<()> {
    let api_repo = get_api_repo("dunzhang/stella_en_400M_v5", None);
    let model_root = download_artifacts(&api_repo).unwrap();
    let tokenizer = load_tokenizer(&model_root)?;
    let dense_paths = download_dense_modules(&api_repo, None)
        .ok()
        .filter(|paths| !paths.is_empty())
        .map(|paths| paths.into_iter().map(|path| path.to_string()).collect());

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        dense_paths, // This will default to `2_Dense_1024/` as defined in `modules.json`
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
    let api_repo = get_api_repo("dunzhang/stella_en_400M_v5", None);
    let model_root = download_artifacts(&api_repo).unwrap();
    let tokenizer = load_tokenizer(&model_root)?;
    let dense_paths = download_dense_modules(&api_repo, Some("2_Dense_768".to_string()))
        .ok()
        .filter(|paths| !paths.is_empty())
        .map(|paths| paths.into_iter().map(|path| path.to_string()).collect());

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        dense_paths,
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
