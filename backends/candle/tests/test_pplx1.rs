mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
fn test_pplx1() -> Result<()> {
    let (model_root, _) = download_artifacts("perplexity-ai/pplx-embed-v1-0.6b", None, None)?;
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
    insta::assert_yaml_snapshot!("pplx1_cpu_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("pplx1_cpu_single", embeddings_single, &matcher);
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_pplx1_quantization() -> Result<()> {
    let (model_root, _) = download_artifacts("perplexity-ai/pplx-embed-v1-0.6b", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    let input = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let embeddings_map = backend.embed(input)?;
    let (pooled_embeddings, _) = sort_embeddings(embeddings_map);
    let embeddings = &pooled_embeddings[0];

    // Verify quantization: all values should be in [-127, 127] and be whole numbers
    for value in embeddings.iter() {
        assert!(
            *value >= -127.0 && *value <= 127.0,
            "Value {} is outside [-127, 127] range",
            value
        );
        // Check values are approximately integers (within small epsilon for floating point)
        let rounded = value.round();
        assert!(
            (value - rounded).abs() < 0.01,
            "Value {} is not close to an integer (rounded: {})",
            value,
            rounded
        );
    }

    Ok(())
}
