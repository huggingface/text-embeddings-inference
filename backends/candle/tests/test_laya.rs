mod common;

use anyhow::Result;
use common::{download_laya_artifacts, load_tokenizer};
use serde::Serialize;
use std::path::PathBuf;
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, DecisionInput, DecisionResult, ModelType};
use tokenizers::Tokenizer;

#[derive(Debug, Serialize)]
struct DecisionSnapshot {
    probabilities: Vec<f32>,
    confidence: f32,
    action: usize,
}

impl From<&DecisionResult> for DecisionSnapshot {
    fn from(decision: &DecisionResult) -> Self {
        Self {
            probabilities: decision.probabilities.clone(),
            confidence: decision.confidence,
            action: decision.action,
        }
    }
}

fn load_laya() -> Result<(CandleBackend, Tokenizer)> {
    let model_path = std::env::var("LAYA_MODEL_PATH")?;
    let model_root = {
        let local_path = PathBuf::from(&model_path);
        if local_path.is_dir() {
            local_path
        } else {
            download_laya_artifacts(&model_path)?
        }
    };
    let tokenizer_root = if model_root.join("tokenizer.json").exists() {
        model_root.clone()
    } else {
        model_root.join("tokenizer")
    };
    let tokenizer = load_tokenizer(&tokenizer_root)?;
    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Decision,
        None,
    )?;

    Ok((backend, tokenizer))
}

fn decision_input(
    tokenizer: &Tokenizer,
    prompt: &str,
    qtype: u32,
    question_index: usize,
    option_count: usize,
) -> DecisionInput {
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let mask_id = tokenizer.token_to_id("[MASK]").unwrap();
    let marker_positions = encoding
        .get_ids()
        .iter()
        .enumerate()
        .filter_map(|(index, id)| (*id == mask_id).then_some(index as u32))
        .collect::<Vec<_>>();
    assert_eq!(marker_positions.len(), option_count);

    DecisionInput {
        input_ids: encoding.get_ids().to_vec(),
        attention_mask: Some(encoding.get_attention_mask().to_vec()),
        qtype,
        marker_positions,
        question_index,
    }
}

fn assert_decision(decision: &text_embeddings_backend_core::DecisionResult, option_count: usize) {
    assert_eq!(decision.probabilities.len(), option_count);
    assert!((decision.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert!((0.0..=1.0).contains(&decision.confidence));
    assert!(decision.action < 2);
}

#[test]
#[ignore = "requires a local Laya checkpoint in LAYA_MODEL_PATH"]
#[serial_test::serial]
fn test_laya_choice() -> Result<()> {
    let (backend, tokenizer) = load_laya()?;
    let input = decision_input(
        &tokenizer,
        "[CLS] choice question: Which team should handle this request? [SEP] [MASK] billing: invoices and refunds [MASK] technical: bugs and outages [SEP] I was charged twice for my invoice. [SEP]",
        0,
        0,
        2,
    );
    let decisions = backend.decide(vec![input])?;

    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].question_index, 0);
    assert_decision(&decisions[0], 2);
    insta::assert_yaml_snapshot!("laya_choice", DecisionSnapshot::from(&decisions[0]));

    Ok(())
}

#[test]
#[ignore = "requires a local Laya checkpoint in LAYA_MODEL_PATH"]
#[serial_test::serial]
fn test_laya_score() -> Result<()> {
    let (backend, tokenizer) = load_laya()?;
    let input = decision_input(
        &tokenizer,
        "[CLS] score question: How urgent is this incident? [SEP] [MASK] low [MASK] medium [MASK] high [SEP] The service is degraded for some users. [SEP]",
        1,
        1,
        3,
    );
    let decisions = backend.decide(vec![input])?;

    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].question_index, 1);
    assert_decision(&decisions[0], 3);
    insta::assert_yaml_snapshot!("laya_score", DecisionSnapshot::from(&decisions[0]));

    Ok(())
}

#[test]
#[ignore = "requires a local Laya checkpoint in LAYA_MODEL_PATH"]
#[serial_test::serial]
fn test_laya_noul() -> Result<()> {
    let (backend, tokenizer) = load_laya()?;
    let input = decision_input(
        &tokenizer,
        "[CLS] noul question: Is this request ready to ship? [SEP] [MASK] false [MASK] true [SEP] The implementation has passing tests and review approval. [SEP]",
        2,
        2,
        2,
    );
    let decisions = backend.decide(vec![input])?;

    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].question_index, 2);
    assert_decision(&decisions[0], 2);
    insta::assert_yaml_snapshot!("laya_noul", DecisionSnapshot::from(&decisions[0]));

    Ok(())
}

#[test]
#[ignore = "requires a local Laya checkpoint in LAYA_MODEL_PATH"]
#[serial_test::serial]
fn test_laya_all_types_batch() -> Result<()> {
    let (backend, tokenizer) = load_laya()?;
    let inputs = vec![
        decision_input(
            &tokenizer,
            "[CLS] choice question: Which team should handle this request? [SEP] [MASK] billing: invoices and refunds [MASK] technical: bugs and outages [SEP] I was charged twice for my invoice. [SEP]",
            0,
            0,
            2,
        ),
        decision_input(
            &tokenizer,
            "[CLS] score question: How urgent is this incident? [SEP] [MASK] low [MASK] medium [MASK] high [SEP] The service is degraded for some users. [SEP]",
            1,
            1,
            3,
        ),
        decision_input(
            &tokenizer,
            "[CLS] noul question: Is this request ready to ship? [SEP] [MASK] false [MASK] true [SEP] The implementation has passing tests and review approval. [SEP]",
            2,
            2,
            2,
        ),
    ];
    let decisions = backend.decide(inputs)?;

    assert_eq!(decisions.len(), 3);
    for (question_index, (decision, option_count)) in
        decisions.iter().zip([2, 3, 2]).enumerate()
    {
        assert_eq!(decision.question_index, question_index);
        assert_decision(decision, option_count);
    }
    let snapshots = decisions.iter().map(DecisionSnapshot::from).collect::<Vec<_>>();
    insta::assert_yaml_snapshot!("laya_all_types_batch", snapshots);

    Ok(())
}
