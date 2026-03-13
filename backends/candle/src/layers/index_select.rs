// SPDX-License-Identifier: MIT or Apache-2.0
// First Published under RadixMLP and https://github.com/michaelfeil/candle-index-select-cu by Michael Feil

use candle::{Result, Tensor};

#[inline]
#[allow(dead_code)]
pub fn index_select(tensor: &Tensor, ids: &Tensor, dim: usize) -> Result<Tensor> {
    tensor.index_select(ids, dim)
}
