//! Listwise reranking orchestration
//!
//! This module contains the implementation for listwise reranking,
//! including vector math utilities and future handler logic.
//!
//! # Queue Isolation Policy
//!
//! ## Current Design (V1)
//!
//! - Listwise reranking uses a separate `BackendCommand::EmbedListwise` variant
//! - **No cross-request batching**: Each request's blocks are processed independently
//! - **Shared worker queue**: Both pairwise and listwise commands go through the same
//!   `BackendThread` worker
//! - Execution order follows arrival order; pairwise and listwise requests may interleave
//!
//! ## Rationale
//!
//! - **Privacy/Accuracy**: Prevents different users' documents from interacting in the
//!   same context window
//! - **Simplicity**: No request grouping logic needed
//! - **Acceptable latency**: Most requests have <125 docs = 1 block
//!
//! ## Future Optimizations (V2)
//!
//! If listwise requests dominate and cause pairwise latency spikes, consider:
//! - Separate worker thread pool for listwise (execution isolation)
//! - Priority queue (pairwise gets higher priority for low latency)
//! - Per-model worker pools (already planned for multi-model serving)
//!
//! ## Important Note
//!
//! Cross-request batching is not supported for listwise reranking. Each request is
//! processed independently, but pairwise and listwise requests share the same backend
//! worker queue and may interleave.

pub mod math;

// Re-export commonly used items
pub use math::{add_scaled, cosine_similarity, normalize, normalize_new, weighted_average};
