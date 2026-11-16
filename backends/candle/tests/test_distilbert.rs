mod common;

use anyhow::Result;
use common::{download_artifacts, load_tokenizer};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{ModelType, Pool};

#[test]
fn test_distilbert_splade_padding() -> Result<()> {
    let (model_root, _) = download_artifacts(
        "hf-internal-testing/tiny-random-DistilBertForMaskedLM",
        Some("b2dfda30b012821996e6e603729562d9c900bc0f"),
        None,
    )?;
    let tokenizer = load_tokenizer(&model_root)?;
    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Splade),
        None,
    )?;
    common::assert_splade_batch_matches_single(&backend, &tokenizer)
}
