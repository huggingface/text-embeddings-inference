// SPDX-License-Identifier: MIT
// Published under RadixMLP by Michael Feil
// Copyright (c) 2025 michaelfeil

use crate::layers::index_select::index_select;
use candle::{Device, Result, Tensor};
use text_embeddings_backend_core::Batch;

/// Helper struct to manage compact/unfold tensor operations for RadixMLP.

/// if is not compact, all are operations are a no-op.
#[allow(dead_code)]
pub struct CompactUnfoldTensors {
    pub scatter_unfold: Option<Tensor>,
    pub fold_gather: Option<Tensor>,
    pub position_ids_compact: Tensor,
}

#[allow(dead_code)]
impl CompactUnfoldTensors {
    /// Create compact/unfold tensors from batch data
    // returning the input_ids tensor and the compact/unfold tensors if applicable.
    pub fn from_batch(batch: &Batch, device: &Device) -> Result<(Tensor, Self)> {
        let shape = batch.input_ids.len();

        let (input_ids, compact_tensors) =
            if let (Some(compact_ids), Some(compact_pos), Some(scatter), Some(fold)) = (
                batch.compact_input_ids.as_ref(),
                batch.compact_position_ids.as_ref(),
                batch.scatter_unfold.as_ref(),
                batch.fold_gather.as_ref(),
            ) {
                let m = compact_ids.len();
                let compact_ids_t = Tensor::from_vec(compact_ids.clone(), m, device)?;
                let scatter_t = Tensor::from_vec(scatter.clone(), shape, device)?;
                let fold_t = Tensor::from_vec(fold.clone(), m, device)?;
                let position_ids_compact = Tensor::from_vec(compact_pos.clone(), m, device)?;

                (
                    compact_ids_t,
                    CompactUnfoldTensors {
                        scatter_unfold: Some(scatter_t),
                        fold_gather: Some(fold_t),
                        position_ids_compact,
                    },
                )
            } else {
                let input_ids = Tensor::from_vec(batch.input_ids.clone(), shape, device)?;
                let position_ids = Tensor::from_vec(batch.position_ids.clone(), shape, device)?;
                (
                    input_ids,
                    CompactUnfoldTensors {
                        scatter_unfold: None,
                        fold_gather: None,
                        position_ids_compact: position_ids,
                    },
                )
            };

        Ok((input_ids, compact_tensors))
    }

    /// Expand compact → original using `scatter_unfold`, if present.
    #[inline]
    pub fn scatter_unfold(&self, tensor: &Tensor) -> Result<Tensor> {
        if let Some(scatter) = &self.scatter_unfold {
            index_select(tensor, scatter, 0)
        } else {
            Ok(tensor.clone())
        }
    }

    /// Gather original → compact using `fold_gather`, if present.
    /// Identity path: returns a shallow handle clone (no device copy).
    #[inline]
    pub fn fold_gather(&self, tensor: &Tensor) -> Result<Tensor> {
        if let Some(gather) = &self.fold_gather {
            Ok(index_select(tensor, gather, 0)?)
        } else {
            Ok(tensor.clone())
        }
    }
}
