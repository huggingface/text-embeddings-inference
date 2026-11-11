mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings, SnapshotScores};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer, relative_matcher};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
fn test_bert() -> Result<()> {
    let (model_root, _) = download_artifacts("sentence-transformers/all-MiniLM-L6-v2", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
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
    insta::assert_yaml_snapshot!("bert_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("bert_single", embeddings_single, &matcher);
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
    assert_eq!(raw_embeddings.len(), 8);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_bert_pooled_raw() -> Result<()> {
    let (model_root, _) = download_artifacts("sentence-transformers/all-MiniLM-L6-v2", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Cls),
        None,
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
    insta::assert_yaml_snapshot!("bert_batch_pooled", pooled_embeddings_batch, &matcher);

    let raw_embeddings_batch = SnapshotEmbeddings::from(raw_embeddings);
    insta::assert_yaml_snapshot!("bert_batch_raw", raw_embeddings_batch, &matcher);

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

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("bert_single_pooled", embeddings_single, &matcher);

    assert_eq!(pooled_embeddings_batch[0], embeddings_single[0]);
    assert_eq!(pooled_embeddings_batch[2], embeddings_single[0]);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        vec![],
        [0].to_vec(),
    );

    let (_, raw_embeddings) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(raw_embeddings);
    insta::assert_yaml_snapshot!("bert_single_raw", embeddings_single, &matcher);

    assert_eq!(raw_embeddings_batch[0], embeddings_single[0]);
    assert_eq!(raw_embeddings_batch[15], embeddings_single[0]);
    assert_eq!(embeddings_single.len(), 7);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_emotions() -> Result<()> {
    let (model_root, _) = download_artifacts("SamLowe/roberta-base-go_emotions", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Classifier,
        None,
    )?;

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

    let predictions: Vec<Vec<f32>> = backend.predict(input_batch)?.into_values().collect();
    let predictions_batch = SnapshotScores::from(predictions);
    insta::assert_yaml_snapshot!("emotions_batch", predictions_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("I like you.", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let predictions: Vec<Vec<f32>> = backend.predict(input_single)?.into_values().collect();
    let predictions_single = SnapshotScores::from(predictions);

    insta::assert_yaml_snapshot!("emotions_single", predictions_single, &matcher);
    assert_eq!(predictions_batch[0], predictions_single[0]);
    assert_eq!(predictions_batch[2], predictions_single[0]);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_bert_classification() -> Result<()> {
    let (model_root, _) =
        download_artifacts("ibm-research/re2g-reranker-nq", Some("refs/pr/3"), None)?;
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
                (
                    "PrimeTime is a timing signoff tool",
                    "PrimeTime can perform most accurate timing analysis",
                ),
                true,
            )
            .unwrap()],
        [0].to_vec(),
        vec![],
    );

    let predictions: Vec<Vec<f32>> = backend.predict(input_single)?.into_values().collect();
    let predictions_single = SnapshotScores::from(predictions);

    let matcher = relative_matcher();
    insta::assert_yaml_snapshot!("bert_classification_single", predictions_single, &matcher);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_bert_splade_padding() -> Result<()> {
    let (model_root, _) = download_artifacts(
        "sparse-encoder-testing/splade-bert-tiny-nq",
        Some("c080c2cff606c1ee9379e883895274367f2c29e3"),
        None,
    )?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Splade),
        None,
    )?;

    let short = tokenizer.encode("What is Deep Learning?", true).unwrap();
    let long = tokenizer
        .encode(
            "Deep Learning is a subset of machine learning based on neural networks with many layers.",
            true,
        )
        .unwrap();
    assert!(short.len() < long.len());

    let input_single = batch(vec![short.clone()], [0].to_vec(), vec![]);
    let (single, _) = sort_embeddings(backend.embed(input_single)?);

    // The short input is padded to the length of the long one
    let input_batch = batch(vec![short, long], [0, 1].to_vec(), vec![]);
    let (batched, _) = sort_embeddings(backend.embed(input_batch)?);

    // Padding positions must not contribute to the max pooling
    let max_diff = batched[0]
        .iter()
        .zip(&single[0])
        .map(|(b, s)| (b - s).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff < 1e-4, "max abs difference: {max_diff}");

    Ok(())
}
