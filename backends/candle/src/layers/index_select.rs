// SPDX-License-Identifier: MIT or Apache-2.0
// First Published under RadixMLP and https://github.com/michaelfeil/candle-index-select-cu by Michael Feil

use candle::{Result, Tensor};

#[cfg(feature = "cuda")]
use candle::DType;
#[cfg(feature = "cuda")]
use candle_index_select_cu;

#[inline]
#[allow(dead_code)]
pub fn index_select(tensor: &Tensor, ids: &Tensor, dim: usize) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if matches!(tensor.dtype(), DType::F16 | DType::F32) && matches!(ids.dtype(), DType::U32) {
            // NOTE: `candle-index-select-cu` supports f16/f32 data and u32 indices
            candle_index_select_cu::index_select(tensor, ids, dim)
        } else {
            tensor.index_select(ids, dim)
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        tensor.index_select(ids, dim)
    }
}
