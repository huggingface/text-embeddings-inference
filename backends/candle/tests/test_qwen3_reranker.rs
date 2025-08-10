mod common;

use anyhow::Result;
use common::{batch, download_artifacts, load_tokenizer};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType};

#[test]
#[serial_test::serial]
fn test_qwen3_reranker() -> Result<()> {
    if std::env::var("SKIP_DOWNLOAD_TESTS").is_ok() {
        return Ok(());
    }

    let model_root = download_artifacts("Qwen/Qwen3-Reranker-0.6B", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::ListwiseReranker,
        None,
    )?;

    let prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
    let suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let instruct = "Given a web search query, retrieve relevant passages that answer the query";

    let format_input = |query: &str, document: &str| -> String {
        format!("{prefix}<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {document}{suffix}")
    };

    let texts = [
        format_input(
            "What is the capital of China?",
            "The capital of China is Beijing.",
        ),
        format_input(
            "What is the capital of China?",
            "The capital of France is Paris.",
        ),
        format_input(
            "What is the capital of China?",
            "China is a large country in Asia.",
        ),
    ];

    let input_batch = batch(
        texts
            .iter()
            .map(|t| tokenizer.encode(t.as_str(), true).unwrap())
            .collect(),
        vec![0, 1, 2],
        vec![],
    );

    let predictions = backend.predict(input_batch)?;
    let scores_vec: Vec<f32> = predictions.into_iter().flat_map(|(_, v)| v).collect();

    assert_eq!(scores_vec.len(), 3, "Should return 3 scores for 3 inputs");

    for (i, &score) in scores_vec.iter().enumerate() {
        assert!(
            score.is_finite() && (0.0..=1.0).contains(&score),
            "Score[{}] = {} should be a valid probability",
            i,
            score
        );
    }

    assert!(
        scores_vec[0] > scores_vec[1],
        "Beijing document (score={}) should score higher than Paris document (score={})",
        scores_vec[0],
        scores_vec[1]
    );

    assert!(
        scores_vec[0] > scores_vec[2],
        "Beijing document (score={}) should score higher than generic China document (score={})",
        scores_vec[0],
        scores_vec[2]
    );

    let single_text = format_input(
        "What is machine learning?",
        "Machine learning is a subset of artificial intelligence.",
    );
    let input_single = batch(
        vec![tokenizer.encode(single_text.as_str(), true).unwrap()],
        vec![0],
        vec![],
    );

    let single_predictions = backend.predict(input_single)?;
    let single_score_vec: Vec<f32> = single_predictions
        .into_iter()
        .flat_map(|(_, v)| v)
        .collect();

    assert_eq!(
        single_score_vec.len(),
        1,
        "Should return 1 score for 1 input"
    );
    assert!(
        single_score_vec[0].is_finite() && (0.0..=1.0).contains(&single_score_vec[0]),
        "Single score should be a valid probability"
    );

    Ok(())
}

#[test]
fn test_qwen3_reranker_model_detection() {
    // Test that model names containing "reranker" are properly detected
    let reranker_models = vec![
        "Qwen/Qwen3-Reranker-0.6B",
        "Qwen/Qwen3-Reranker-4B",
        "custom/qwen3-reranker-finetuned",
        "org/model-qwen3-reranker",
    ];

    let non_reranker_models = vec![
        "Qwen/Qwen3-0.5B",
        "Qwen/Qwen3-7B-Instruct",
        "Qwen/Qwen3-Embedding-0.6B",
    ];

    for model_name in reranker_models {
        assert!(
            model_name.to_lowercase().contains("reranker"),
            "Model {} should be detected as reranker",
            model_name
        );
    }

    for model_name in non_reranker_models {
        assert!(
            !model_name.to_lowercase().contains("reranker"),
            "Model {} should NOT be detected as reranker",
            model_name
        );
    }
}
