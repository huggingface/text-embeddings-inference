//! Reranking strategy types and CLI argument parsing
//!
//! This module contains router-level enums for controlling
//! listwise vs pairwise reranking behavior.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Runtime reranking strategy (determined at request time)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RerankStrategy {
    Pairwise,
    Listwise,
}

/// CLI mode for reranker selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankMode {
    Auto,
    Pairwise,
    Listwise,
}

impl std::str::FromStr for RerankMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "pairwise" => Ok(Self::Pairwise),
            "listwise" => Ok(Self::Listwise),
            _ => Err(anyhow!(
                "Invalid reranker mode: {}. Valid values: auto, pairwise, listwise",
                s
            )),
        }
    }
}

/// Document ordering strategy for listwise processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankOrdering {
    Input,
    Random,
}

impl std::str::FromStr for RerankOrdering {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "input" => Ok(Self::Input),
            "random" => Ok(Self::Random),
            _ => Err(anyhow!(
                "Invalid rerank ordering: {}. Valid values: input, random",
                s
            )),
        }
    }
}

/// Determine the runtime reranking strategy from CLI mode and model capabilities
///
/// # Arguments
/// * `mode` - The CLI reranker mode (Auto/Pairwise/Listwise)
/// * `kind` - The detected model kind (ModelKind from core)
///
/// # Returns
/// The determined RerankStrategy (Pairwise or Listwise)
///
/// # Errors
/// Returns error if:
/// - Listwise mode is requested but model doesn't support it
/// - Pairwise mode is requested but model only supports listwise
pub fn determine_strategy(
    mode: &RerankMode,
    kind: &text_embeddings_core::detection::ModelKind,
) -> Result<RerankStrategy> {
    use text_embeddings_core::detection::ModelKind;

    match (mode, kind) {
        // Auto mode: Choose based on model capabilities
        (RerankMode::Auto, ModelKind::ListwiseReranker) => Ok(RerankStrategy::Listwise),
        (RerankMode::Auto, _) => Ok(RerankStrategy::Pairwise),

        // Pairwise mode: Reject listwise-only models
        (RerankMode::Pairwise, ModelKind::ListwiseReranker) => Err(anyhow!(
            "This model only supports listwise reranking. \
             Use --reranker-mode auto or --reranker-mode listwise."
        )),
        (RerankMode::Pairwise, _) => Ok(RerankStrategy::Pairwise),

        // Listwise mode: Only allow if model supports it
        (RerankMode::Listwise, ModelKind::ListwiseReranker) => Ok(RerankStrategy::Listwise),
        (RerankMode::Listwise, kind) => Err(anyhow!(
            "Model kind {:?} does not support listwise reranking. \
             Model must have projector weights and special tokens. \
             Use --reranker-mode auto or --reranker-mode pairwise.",
            kind
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_mode_from_str() {
        // Test valid values
        assert_eq!("auto".parse::<RerankMode>().unwrap(), RerankMode::Auto);
        assert_eq!(
            "pairwise".parse::<RerankMode>().unwrap(),
            RerankMode::Pairwise
        );
        assert_eq!(
            "listwise".parse::<RerankMode>().unwrap(),
            RerankMode::Listwise
        );

        // Test case insensitivity
        assert_eq!("AUTO".parse::<RerankMode>().unwrap(), RerankMode::Auto);
        assert_eq!(
            "Pairwise".parse::<RerankMode>().unwrap(),
            RerankMode::Pairwise
        );
        assert_eq!(
            "LISTWISE".parse::<RerankMode>().unwrap(),
            RerankMode::Listwise
        );
    }

    #[test]
    fn test_rerank_mode_invalid() {
        let result = "invalid".parse::<RerankMode>();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid reranker mode"));
    }

    #[test]
    fn test_rerank_ordering_from_str() {
        // Test valid values
        assert_eq!(
            "input".parse::<RerankOrdering>().unwrap(),
            RerankOrdering::Input
        );
        assert_eq!(
            "random".parse::<RerankOrdering>().unwrap(),
            RerankOrdering::Random
        );

        // Test case insensitivity
        assert_eq!(
            "INPUT".parse::<RerankOrdering>().unwrap(),
            RerankOrdering::Input
        );
        assert_eq!(
            "RANDOM".parse::<RerankOrdering>().unwrap(),
            RerankOrdering::Random
        );
    }

    #[test]
    fn test_rerank_ordering_invalid() {
        let result = "invalid".parse::<RerankOrdering>();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid rerank ordering"));
    }
}
