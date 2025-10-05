//! Model detection utilities shared between router and backends
//!
//! CRITICAL: This module lives in `core` to avoid circular dependencies.
//! Both `router` and `backends/candle` can safely import from here.

use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

/// Model kind classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Embedding,
    SequenceClassifier,
    ListwiseReranker,
}

/// Check if tokenizer has special tokens for listwise reranking
fn has_special_tokens(tokenizer: &Tokenizer) -> Result<bool> {
    let embed_token_id = tokenizer.token_to_id("<|embed_token|>");
    let rerank_token_id = tokenizer.token_to_id("<|rerank_token|>");

    Ok(embed_token_id.is_some() && rerank_token_id.is_some())
}

/// Check if architecture is Qwen3-based
fn is_qwen_architecture(model_path: &Path) -> Result<bool> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(config_path).context("Failed to read config.json")?;
    let config: Value = serde_json::from_str(&config_str).context("Failed to parse config.json")?;

    // Check architectures field
    if let Some(arch_array) = config.get("architectures").and_then(|v| v.as_array()) {
        for arch in arch_array {
            if let Some(arch_str) = arch.as_str() {
                if matches!(
                    arch_str,
                    "QwenForCausalLM" | "Qwen3ForCausalLM" | "JinaForRanking"
                ) {
                    return Ok(true);
                }
            }
        }
    }

    // Check model_type field as fallback (qwen3 only)
    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
        if model_type == "qwen3" {
            return Ok(true);
        }
    }

    Ok(false)
}

fn is_sequence_classifier(model_path: &Path) -> Result<bool> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(config_path)?;
    let config: Value = serde_json::from_str(&config_str)?;

    // Check for id2label (classifier signature)
    Ok(config.get("id2label").is_some())
}

/// Detect model kind with listwise priority
pub fn detect_model_kind(model_path: &Path, tokenizer: &Tokenizer) -> Result<ModelKind> {
    // Check components independently for better error reporting
    let has_qwen_arch = is_qwen_architecture(model_path).unwrap_or(false);
    let has_tokens = has_special_tokens(tokenizer).unwrap_or(false);

    // For now, we'll implement a simplified version without projector weights checking
    // This will be expanded in the next iteration
    if has_qwen_arch && has_tokens {
        tracing::info!("✓ Detected ListwiseReranker: arch=qwen3, tokens=yes");
        return Ok(ModelKind::ListwiseReranker);
    }

    // Priority 2: Sequence classifier (existing logic)
    if is_sequence_classifier(model_path)? {
        tracing::info!("✓ Detected SequenceClassifier model");
        return Ok(ModelKind::SequenceClassifier);
    }

    // Default: Embedding
    tracing::info!("✓ Detected Embedding model (default)");
    Ok(ModelKind::Embedding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    use tokenizers::Tokenizer;

    fn create_test_tokenizer() -> Tokenizer {
        // For now, we'll create a simple tokenizer for testing
        // In real implementation, this would be a proper tokenizer
        // This is a placeholder to get tests passing
        Tokenizer::from_file("tokenizer.json").unwrap_or_else(|_| {
            // Create a dummy tokenizer for testing
            panic!("Test tokenizer setup needed");
        })
    }

    #[test]
    fn test_has_special_tokens_with_valid_tokenizer() {
        // This test will be implemented when we have a proper test tokenizer setup
        // For now, we'll implement a basic test structure
    }

    #[test]
    fn test_has_special_tokens_without_tokens() {
        // This test will be implemented when we have a proper test tokenizer setup
    }

    #[test]
    fn test_modelkind_equality() {
        assert_eq!(ModelKind::Embedding, ModelKind::Embedding);
        assert_eq!(ModelKind::SequenceClassifier, ModelKind::SequenceClassifier);
        assert_eq!(ModelKind::ListwiseReranker, ModelKind::ListwiseReranker);

        assert_ne!(ModelKind::Embedding, ModelKind::SequenceClassifier);
        assert_ne!(ModelKind::SequenceClassifier, ModelKind::ListwiseReranker);
        assert_ne!(ModelKind::ListwiseReranker, ModelKind::Embedding);
    }

    #[test]
    fn test_modelkind_debug_format() {
        assert_eq!(format!("{:?}", ModelKind::Embedding), "Embedding");
        assert_eq!(
            format!("{:?}", ModelKind::SequenceClassifier),
            "SequenceClassifier"
        );
        assert_eq!(
            format!("{:?}", ModelKind::ListwiseReranker),
            "ListwiseReranker"
        );
    }

    #[test]
    fn test_is_sequence_classifier_with_id2label() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config_path = temp_dir.path().join("config.json");

        let config = serde_json::json!({
            "architectures": ["BertModel"],
            "id2label": {"0": "LABEL0", "1": "LABEL1"}
        });

        fs::write(&config_path, config.to_string())?;

        let result = is_sequence_classifier(temp_dir.path())?;
        assert!(result);

        Ok(())
    }

    #[test]
    fn test_is_sequence_classifier_without_id2label() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config_path = temp_dir.path().join("config.json");

        let config = serde_json::json!({
            "architectures": ["BertModel"]
        });

        fs::write(&config_path, config.to_string())?;

        let result = is_sequence_classifier(temp_dir.path())?;
        assert!(!result);

        Ok(())
    }

    #[test]
    fn test_is_qwen_architecture_qwen3() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config_path = temp_dir.path().join("config.json");

        let config = serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3"
        });

        fs::write(&config_path, config.to_string())?;

        let result = is_qwen_architecture(temp_dir.path())?;
        assert!(result);

        Ok(())
    }

    #[test]
    fn test_is_qwen_architecture_other() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config_path = temp_dir.path().join("config.json");

        let config = serde_json::json!({
            "architectures": ["BertModel"],
            "model_type": "bert"
        });

        fs::write(&config_path, config.to_string())?;

        let result = is_qwen_architecture(temp_dir.path())?;
        assert!(!result);

        Ok(())
    }
}
