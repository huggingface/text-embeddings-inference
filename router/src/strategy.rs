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
