use candle::{Result, Tensor};

#[inline]
#[allow(dead_code)]
pub fn index_select(tensor: &Tensor, ids: &Tensor, dim: usize) -> Result<Tensor> {
    tensor.index_select(ids, dim)
}
