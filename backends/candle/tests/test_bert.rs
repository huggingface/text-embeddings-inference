mod common;

use crate::common::SnapshotScores;
use anyhow::Result;
use common::{batch, download_artifacts, load_tokenizer, relative_matcher};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
fn test_mini() -> Result<()> {
    let model_root = download_artifacts("sentence-transformers/all-MiniLM-L6-v2")?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        model_root,
        "float32".to_string(),
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

    let embeddings_batch = SnapshotScores::from(backend.embed(input_batch)?.pooled_embeddings);
    insta::assert_yaml_snapshot!("mini_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let embeddings_single = SnapshotScores::from(backend.embed(input_single)?.pooled_embeddings);

    insta::assert_yaml_snapshot!("mini_single", embeddings_single, &matcher);
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

    let embeddings = backend.embed(input_batch)?;
    let pooled_embeddings = SnapshotScores::from(embeddings.pooled_embeddings);
    let raw_embeddings = SnapshotScores::from(embeddings.raw_embeddings);

    assert_eq!(embeddings_batch[0], pooled_embeddings[0]);
    assert_eq!(raw_embeddings.len(), 8);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_mini_pooled_raw() -> Result<()> {
    let model_root = download_artifacts("sentence-transformers/all-MiniLM-L6-v2")?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Cls),
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

    let matcher = relative_matcher();

    let embeddings = backend.embed(input_batch)?;
    let pooled_embeddings_batch = SnapshotScores::from(embeddings.pooled_embeddings);
    insta::assert_yaml_snapshot!("mini_batch_pooled", pooled_embeddings_batch, &matcher);

    let raw_embeddings_batch = SnapshotScores::from(embeddings.raw_embeddings);
    insta::assert_yaml_snapshot!("mini_batch_raw", raw_embeddings_batch, &matcher);

    // Check that the first token of each raw embeddings member is the same as the cls pooling ones
    assert_eq!(pooled_embeddings_batch[0], raw_embeddings_batch[0]);
    assert_eq!(pooled_embeddings_batch[1], raw_embeddings_batch[7]);
    assert_eq!(pooled_embeddings_batch[2], raw_embeddings_batch[15]);
    assert_eq!(raw_embeddings_batch.len(), 22);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let embeddings_single = backend.embed(input_single)?;
    let embeddings_single = SnapshotScores::from(embeddings_single.pooled_embeddings);
    insta::assert_yaml_snapshot!("mini_single_pooled", embeddings_single, &matcher);

    assert_eq!(pooled_embeddings_batch[0], embeddings_single[0]);
    assert_eq!(pooled_embeddings_batch[2], embeddings_single[0]);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        vec![],
        [0].to_vec(),
    );

    let embeddings_single = backend.embed(input_single)?;

    let embeddings_single = SnapshotScores::from(embeddings_single.raw_embeddings);
    insta::assert_yaml_snapshot!("mini_single_raw", embeddings_single, &matcher);

    assert_eq!(raw_embeddings_batch[0], embeddings_single[0]);
    assert_eq!(raw_embeddings_batch[15], embeddings_single[0]);
    assert_eq!(embeddings_single.len(), 7);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_emotions() -> Result<()> {
    let model_root = download_artifacts("SamLowe/roberta-base-go_emotions")?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(model_root, "float32".to_string(), ModelType::Classifier)?;

    let input_batch = batch(
        vec![
            tokenizer.encode("I like you.", true).unwrap(),
            tokenizer
                .encode("I am not having a great day.", true)
                .unwrap(),
            tokenizer.encode("I like you.", true).unwrap(),
        ],
        [0, 1, 2].to_vec(),
        vec![],
    );

    let matcher = relative_matcher();

    let predictions_batch = SnapshotScores::from(backend.predict(input_batch)?);
    insta::assert_yaml_snapshot!("emotions_batch", predictions_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("I like you.", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let predictions_single = SnapshotScores::from(backend.predict(input_single)?);

    insta::assert_yaml_snapshot!("emotions_single", predictions_single, &matcher);
    assert_eq!(predictions_batch[0], predictions_single[0]);
    assert_eq!(predictions_batch[2], predictions_single[0]);

    Ok(())
}
