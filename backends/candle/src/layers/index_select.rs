// SPDX-License-Identifier: MIT
// Published under RadixMLP by Michael Feil
// Copyright (c) 2025 michaelfeil

// First Published under RadixMLP and https://github.com/michaelfeil/candle-index-select-cu by Michael Feil

use candle::{Result, Tensor};
#[cfg(feature = "cuda")]
use candle_index_select_cu;

#[inline]
#[allow(dead_code)]
pub fn index_select(tensor: &Tensor, ids: &Tensor, dim: usize) -> Result<Tensor> {
    #[cfg(not(feature = "cuda"))]
    {
        tensor.index_select(ids, dim)
    }
    #[cfg(feature = "cuda")]
    {
        candle_index_select_cu::index_select(tensor, ids, dim)
    }
}
