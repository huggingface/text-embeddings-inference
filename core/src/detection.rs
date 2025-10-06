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

/// Check if model has projector weights (LBNL signature)
fn has_projector_weights(model_path: &Path) -> Result<bool> {
    // Priority 1: Check index.json for sharded models
    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        return check_projector_in_index(&index_path);
    }

    // Priority 2: Check single safetensors file header
    let single_file = model_path.join("model.safetensors");
    if single_file.exists() {
        return check_projector_in_safetensors(&single_file);
    }

    // Priority 3: Fallback for non-standard layouts (pytorch_model.bin, etc.)
    let mut has_proj0 = false;
    let mut has_proj2 = false;
    for entry in std::fs::read_dir(model_path)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.contains("projector.0.weight") {
            has_proj0 = true;
        }
        if name.contains("projector.2.weight") {
            has_proj2 = true;
        }
    }
    Ok(has_proj0 && has_proj2)
}

/// Check projector weights in safetensors index file
fn check_projector_in_index(index_path: &Path) -> Result<bool> {
    let index_str = std::fs::read_to_string(index_path)?;
    let index: Value = serde_json::from_str(&index_str)?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("Invalid weight_map in index")?;

    let has_proj0_weight = weight_map.contains_key("projector.0.weight");
    let has_proj2_weight = weight_map.contains_key("projector.2.weight");
    let has_proj0_bias = weight_map.contains_key("projector.0.bias");
    let has_proj2_bias = weight_map.contains_key("projector.2.bias");

    // Must have both weights, no biases (per Jina v3 spec: bias=False)
    Ok(has_proj0_weight && has_proj2_weight && !has_proj0_bias && !has_proj2_bias)
}

/// Check projector weights in single safetensors file
fn check_projector_in_safetensors(path: &Path) -> Result<bool> {
    use std::fs::File;
    use std::io::Read;

    // Read just the header (first 8 bytes is header size, then JSON header)
    let mut file = File::open(path)?;
    let mut header_size_buf = [0u8; 8];
    file.read_exact(&mut header_size_buf)?;
    let header_size = u64::from_le_bytes(header_size_buf);

    // Read the JSON header
    let mut header_buf = vec![0u8; header_size as usize];
    file.read_exact(&mut header_buf)?;
    let header_str =
        String::from_utf8(header_buf).context("Invalid UTF-8 in safetensors header")?;
    let header: Value =
        serde_json::from_str(&header_str).context("Failed to parse safetensors header")?;

    // Check for required projector weights
    let has_w0 = header.get("projector.0.weight").is_some();
    let has_w2 = header.get("projector.2.weight").is_some();
    let has_b0 = header.get("projector.0.bias").is_some();
    let has_b2 = header.get("projector.2.bias").is_some();

    if has_b0 || has_b2 {
        tracing::warn!(
            "Projector bias detected in {:?} (Jina v3 requires bias=False). Model may be incompatible.",
            path
        );
        return Ok(false);
    }

    Ok(has_w0 && has_w2)
}

fn is_sequence_classifier(model_path: &Path) -> Result<bool> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(config_path)?;
    let config: Value = serde_json::from_str(&config_str)?;

    // Check for id2label (classifier signature)
    Ok(config.get("id2label").is_some())
}

/// Detect model kind with listwise priority
pub fn detect_model_kind(tokenizer: &Tokenizer, model_path: &Path) -> Result<ModelKind> {
    // Check components independently for better error reporting
    let has_qwen_arch = is_qwen_architecture(model_path)?;
    let has_projector = has_projector_weights(model_path)?;
    let has_tokens = has_special_tokens(tokenizer)?;

    // Priority 1: Listwise reranker (ALL three conditions must be true)
    if has_qwen_arch && has_projector && has_tokens {
        tracing::info!("✓ Detected ListwiseReranker: arch=qwen3, projector=yes, tokens=yes");
        return Ok(ModelKind::ListwiseReranker);
    }

    // Log partial matches for debugging
    if has_tokens && !has_projector {
        tracing::warn!(
            "⚠ Model has listwise special tokens but NO projector weights detected. \
             Falling back to pairwise mode. This may be a detection error."
        );
    } else if has_projector && !has_tokens {
        tracing::warn!(
            "⚠ Model has projector weights but NO listwise special tokens. \
             Falling back to pairwise mode. Verify tokenizer_config.json."
        );
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

    fn create_test_tokenizer(with_special_tokens: bool) -> Tokenizer {
        use tokenizers::models::bpe::BPE;
        use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

        // Create a minimal BPE tokenizer for testing
        let mut tokenizer =
            TokenizerBuilder::new(BPE::builder().build().expect("Failed to build BPE model"));

        if with_special_tokens {
            // Add the LBNL special tokens
            tokenizer.add_special_tokens(&[
                AddedToken::from("<|embed_token|>", true),
                AddedToken::from("<|rerank_token|>", true),
            ]);
        }

        tokenizer
    }

    #[test]
    fn test_has_special_tokens_with_valid_tokenizer() {
        let tokenizer = create_test_tokenizer(true);
        let result = has_special_tokens(&tokenizer);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_has_special_tokens_without_tokens() {
        let tokenizer = create_test_tokenizer(false);
        let result = has_special_tokens(&tokenizer);
        assert!(result.is_ok());
        assert!(!result.unwrap());
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
