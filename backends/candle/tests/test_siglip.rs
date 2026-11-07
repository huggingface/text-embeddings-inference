mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
fn test_siglip() -> Result<()> {
    let (model_root, _) =
        download_artifacts("Veritone/siglip-base-patch16-224-text", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    // SigLIP always pools the final position and applies the projection head;
    // the configured pooling mode is ignored by the model.
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
    insta::assert_yaml_snapshot!("siglip_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("siglip_single", embeddings_single, &matcher);
    // Identical inputs must produce identical embeddings.
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_siglip_all() -> Result<()> {
    let (model_root, _) =
        download_artifacts("Veritone/siglip-base-patch16-224-text", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    // Request raw (per-token) embeddings. This exercises the pad-to-64 padding and
    // the padding-drop logic in `SiglipTextModel::forward`, which returns the
    // pre-head hidden states for the real tokens only.
    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        vec![],
        [0].to_vec(),
    );

    let (_, raw_embeddings) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_raw = SnapshotEmbeddings::from(raw_embeddings);

    let matcher = cosine_matcher();
    insta::assert_yaml_snapshot!("siglip_single_raw", embeddings_raw, &matcher);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_siglip_classifier_unsupported() -> Result<()> {
    let (model_root, _) =
        download_artifacts("Veritone/siglip-base-patch16-224-text", None, None)?;

    // SigLIP is embedding-only; classification is explicitly rejected at load time.
    let result = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Classifier,
        None,
    );

    assert!(
        result.is_err(),
        "expected SigLIP to reject `ModelType::Classifier`"
    );

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_siglip_long_input_truncated() -> Result<()> {
    let (model_root, _) =
        download_artifacts("Veritone/siglip-base-patch16-224-text", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    // An input longer than the fixed 64-token window must be truncated inside the model
    // rather than error out. Batch it with a short input to assert the over-long sequence
    // does not poison the rest of the batch (previously the reshape failed for the whole
    // batch with a shape mismatch).
    let long_text = "What is Deep Learning? ".repeat(30);
    let long_encoding = tokenizer.encode(long_text.as_str(), true).unwrap();
    assert!(
        long_encoding.len() > 64,
        "test input should exceed the 64-token window; got {}",
        long_encoding.len()
    );

    let input_batch = batch(
        vec![
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            long_encoding,
        ],
        [0, 1].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    assert_eq!(pooled_embeddings.len(), 2);
    for embedding in &pooled_embeddings {
        assert_eq!(embedding.len(), 768);
    }

    Ok(())
}

// SigLIP2 uses the Gemma tokenizer, whose pad token is `<pad>` = 0 (not `</s>` = 1 like
// v1). Because SigLIP runs without an attention mask and pools the final — padding —
// position, the pad token id is load-bearing: this test only passes if the model pads
// with 0, which comes from `pad_token_id` in the checkpoint's `text_config`. The v1 tests
// above cannot catch a wrong pad id because v1's pad token really is 1.
#[test]
#[serial_test::serial]
fn test_siglip2() -> Result<()> {
    let (model_root, _) =
        download_artifacts("Veritone/siglip2-base-patch16-224-text", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    let input_batch = batch(
        vec![
            tokenizer.encode("a dog running on grass", true).unwrap(),
            tokenizer
                .encode("the quick brown fox jumps over the lazy dog", true)
                .unwrap(),
            tokenizer.encode("a dog running on grass", true).unwrap(),
        ],
        [0, 1, 2].to_vec(),
        vec![],
    );

    let matcher = cosine_matcher();

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    let embeddings_batch = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("siglip2_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("a dog running on grass", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("siglip2_single", embeddings_single, &matcher);
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}

#[test]
#[serial_test::serial]
fn test_siglip2_all() -> Result<()> {
    let (model_root, _) =
        download_artifacts("Veritone/siglip2-base-patch16-224-text", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    let input_single = batch(
        vec![tokenizer.encode("a dog running on grass", true).unwrap()],
        vec![],
        [0].to_vec(),
    );

    let (_, raw_embeddings) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_raw = SnapshotEmbeddings::from(raw_embeddings);

    let matcher = cosine_matcher();
    insta::assert_yaml_snapshot!("siglip2_single_raw", embeddings_raw, &matcher);

    Ok(())
}
