// This binary only needs the download/tokenizer helpers; it builds its own batches.
#[allow(dead_code)]
mod common;

use crate::common::{download_artifacts, load_tokenizer};
use anyhow::Result;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, Embedding, ModelType, Pool};
use tokenizers::Encoding;

/// XLM-RoBERTa position ids start at `pad_token_id + 1`; the router applies this offset in
/// `text_embeddings_core::tokenization`, so a parity check against FlagEmbedding has to apply it
/// too. The shared `common::batch` helper always starts at 0, which is only self-consistent with
/// the snapshots it produced.
const XLM_ROBERTA_POSITION_OFFSET: u32 = 2;

fn m3_batch(encodings: Vec<Encoding>) -> Batch {
    let mut input_ids = Vec::new();
    let mut token_type_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut cumulative_seq_lengths = vec![0];

    let mut max_length = 0;
    let mut cumulative_length = 0;

    for encoding in encodings.iter() {
        let encoding_length = encoding.len() as u32;
        input_ids.extend(encoding.get_ids().to_vec());
        token_type_ids.extend(encoding.get_type_ids().to_vec());
        position_ids
            .extend(XLM_ROBERTA_POSITION_OFFSET..(encoding_length + XLM_ROBERTA_POSITION_OFFSET));
        cumulative_length += encoding_length;
        cumulative_seq_lengths.push(cumulative_length);
        max_length = max_length.max(encoding_length);
    }

    let pooled_indices = (0..encodings.len() as u32).collect();

    Batch {
        input_ids,
        token_type_ids,
        position_ids,
        cumulative_seq_lengths,
        max_length,
        pooled_indices,
        raw_indices: vec![],
    }
}

/// Reference bge-m3 sparse output, captured from FlagEmbedding — see
/// `tests/fixtures/README.md` for provenance.
#[derive(Deserialize)]
struct Golden {
    cases: Vec<GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    text: String,
    /// `{token_id: weight}`
    lexical_weights: HashMap<String, f32>,
}

fn golden() -> Result<Golden> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/m3_golden.json");
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

/// Mirrors the router's `sparsify`: the pooled vector is vocabulary-sized and dense-typed, and
/// everything non-zero is a lexical weight.
fn sparsify(embedding: Embedding) -> HashMap<u32, f32> {
    let Embedding::Pooled(values) = embedding else {
        panic!("expected a pooled embedding")
    };
    values
        .into_iter()
        .enumerate()
        .filter(|(_, value)| *value != 0.0)
        .map(|(index, value)| (index as u32, value))
        .collect()
}

#[test]
#[serial_test::serial]
fn test_m3_sparse_matches_flagembedding() -> Result<()> {
    let golden = golden()?;
    let (model_root, _) = download_artifacts("BAAI/bge-m3", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::M3Sparse),
        None,
    )?;

    for (i, case) in golden.cases.iter().enumerate() {
        let encoding = tokenizer.encode(case.text.as_str(), true).unwrap();
        let input_batch = m3_batch(vec![encoding]);

        let mut embeddings = backend.embed(input_batch)?;
        let got = sparsify(embeddings.remove(&0).unwrap());

        let expected: HashMap<u32, f32> = case
            .lexical_weights
            .iter()
            .map(|(token_id, weight)| (token_id.parse().unwrap(), *weight))
            .collect();

        // The index set must match exactly: it is decided by the tokenizer plus the
        // special-token/`w > 0` filter, none of which may drift.
        let got_indices: HashSet<&u32> = got.keys().collect();
        let expected_indices: HashSet<&u32> = expected.keys().collect();
        let intersection = got_indices.intersection(&expected_indices).count();
        let union = got_indices.union(&expected_indices).count();
        let jaccard = intersection as f64 / union as f64;
        assert_eq!(
            jaccard, 1.0,
            "case {i} ({:?}): index set drifted — got {got_indices:?}, expected {expected_indices:?}",
            case.text
        );

        // Values are allowed a small drift: the golden was captured against exact GeLU, while
        // candle's bge-m3 uses the GeLU + tanh approximation.
        let max_delta = expected
            .iter()
            .map(|(token_id, want)| (got[token_id] - want).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta <= 1e-3,
            "case {i} ({:?}): max |delta| = {max_delta} exceeds 1e-3",
            case.text
        );

        println!(
            "case {i}: {:?} -> {} weights, max |delta| = {max_delta:.3e}",
            case.text,
            got.len()
        );
    }

    Ok(())
}

/// The head must refuse a model that does not ship `sparse_linear.pt`, rather than panic.
#[test]
#[serial_test::serial]
fn test_m3_sparse_rejects_model_without_sparse_head() -> Result<()> {
    let (model_root, _) = download_artifacts("sentence-transformers/all-MiniLM-L6-v2", None, None)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::M3Sparse),
        None,
    );

    match backend {
        Ok(_) => panic!("`m3_sparse` must not load on a model without a sparse head"),
        Err(err) => assert!(
            err.to_string().contains("m3_sparse"),
            "error should name the pooling mode, got: {err}"
        ),
    }

    Ok(())
}
