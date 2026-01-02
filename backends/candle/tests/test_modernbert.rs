mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings};
use anyhow::Result;
use common::{
    batch, cosine_matcher, download_artifacts, load_tokenizer, relative_matcher, SnapshotScores,
};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
fn test_modernbert() -> Result<()> {
    let (model_root, _) = download_artifacts("answerdotai/ModernBERT-base", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
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
    if cfg!(all(feature = "cuda", feature = "flash-attn")) {
        insta::assert_yaml_snapshot!("modernbert_batch_flash", embeddings_batch, &matcher);
    } else {
        insta::assert_yaml_snapshot!("modernbert_batch", embeddings_batch, &matcher);
    }

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    if cfg!(all(feature = "cuda", feature = "flash-attn")) {
        insta::assert_yaml_snapshot!("modernbert_single_flash", embeddings_single, &matcher);
    } else {
        insta::assert_yaml_snapshot!("modernbert_single", embeddings_single, &matcher);
    }
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    let input_batch = batch(
        vec![
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            tokenizer.encode("Deep Learning is...", true).unwrap(),
        ],
        [0].to_vec(),
        [1].to_vec(),
    );

    let (pooled_embeddings, raw_embeddings) = sort_embeddings(backend.embed(input_batch)?);
    let pooled_embeddings = SnapshotEmbeddings::from(pooled_embeddings);
    let raw_embeddings = SnapshotEmbeddings::from(raw_embeddings);

    assert_eq!(embeddings_batch[0], pooled_embeddings[0]);
    assert_eq!(raw_embeddings.len(), 6);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_modernbert_pooled_raw() -> Result<()> {
    let (model_root, _) = download_artifacts("answerdotai/ModernBERT-base", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Cls),
        None,
        0,
    )?;

    let input_batch = batch(
        vec![
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            tokenizer.encode("Deep Learning is...", true).unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            tokenizer.encode("Deep Learning is...", true).unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
        ],
        [0, 2, 3].to_vec(),
        [1, 4, 5].to_vec(),
    );

    let matcher = cosine_matcher();

    let (pooled_embeddings, raw_embeddings) = sort_embeddings(backend.embed(input_batch)?);
    let pooled_embeddings_batch = SnapshotEmbeddings::from(pooled_embeddings);
    if cfg!(all(feature = "cuda", feature = "flash-attn")) {
        insta::assert_yaml_snapshot!(
            "modernbert_batch_pooled_flash",
            pooled_embeddings_batch,
            &matcher
        );
    } else {
        insta::assert_yaml_snapshot!("modernbert_batch_pooled", pooled_embeddings_batch, &matcher);
    }

    let raw_embeddings_batch = SnapshotEmbeddings::from(raw_embeddings);
    if cfg!(all(feature = "cuda", feature = "flash-attn")) {
        insta::assert_yaml_snapshot!("modernbert_batch_raw_flash", raw_embeddings_batch, &matcher);
    } else {
        insta::assert_yaml_snapshot!("modernbert_batch_raw", raw_embeddings_batch, &matcher);
    }

    // Check that the first token of each raw embeddings member is the same as the cls pooling ones
    assert_eq!(pooled_embeddings_batch[0], raw_embeddings_batch[0]);
    assert_eq!(pooled_embeddings_batch[1], raw_embeddings_batch[7]);
    assert_eq!(pooled_embeddings_batch[2], raw_embeddings_batch[13]);
    assert_eq!(raw_embeddings_batch.len(), 20);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);
    if cfg!(all(feature = "cuda", feature = "flash-attn")) {
        insta::assert_yaml_snapshot!(
            "modernbert_single_pooled_flash",
            embeddings_single,
            &matcher
        );
    } else {
        insta::assert_yaml_snapshot!("modernbert_single_pooled", embeddings_single, &matcher);
    }

    assert_eq!(pooled_embeddings_batch[0], embeddings_single[0]);
    assert_eq!(pooled_embeddings_batch[2], embeddings_single[0]);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        vec![],
        [0].to_vec(),
    );

    let (_, raw_embeddings) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(raw_embeddings);
    if cfg!(all(feature = "cuda", feature = "flash-attn")) {
        insta::assert_yaml_snapshot!("modernbert_single_raw_flash", embeddings_single, &matcher);
    } else {
        insta::assert_yaml_snapshot!("modernbert_single_raw", embeddings_single, &matcher);
    }

    assert_eq!(raw_embeddings_batch[0], embeddings_single[0]);
    assert_eq!(raw_embeddings_batch[13], embeddings_single[0]);
    assert_eq!(embeddings_single.len(), 7);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_modernbert_classification() -> Result<()> {
    let (model_root, _) =
        download_artifacts("Alibaba-NLP/gte-reranker-modernbert-base", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Classifier,
        None,
        0,
    )?;

    let input_single = batch(
        vec![tokenizer
            .encode(("What is Deep Learning?", "Deep Learning is not..."), true)
            .unwrap()],
        [0].to_vec(),
        vec![],
    );

    let predictions: Vec<Vec<f32>> = backend
        .predict(input_single)?.into_values()
        .collect();
    let predictions_single = SnapshotScores::from(predictions);

    let matcher = relative_matcher();
    insta::assert_yaml_snapshot!(
        "modernbert_classification_single",
        predictions_single,
        &matcher
    );

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_modernbert_classification_mean_pooling() -> Result<()> {
    let (model_root, _) =
        download_artifacts("tomaarsen/reranker-ModernBERT-large-gooaq-bce", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Classifier,
        None,
        0,
    )?;

    let input_single = batch(
        vec![tokenizer
            .encode(("What is Deep Learning?", "Deep Learning is not..."), true)
            .unwrap()],
        [0].to_vec(),
        vec![],
    );

    let predictions: Vec<Vec<f32>> = backend
        .predict(input_single)?.into_values()
        .collect();
    let predictions_single = SnapshotScores::from(predictions);

    let matcher = relative_matcher();
    insta::assert_yaml_snapshot!(
        "modernbert_classification_mean_pooling",
        predictions_single,
        &matcher
    );

    Ok(())
}
