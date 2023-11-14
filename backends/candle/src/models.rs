#[cfg(any(feature = "mkl", feature = "mkl-dynamic"))]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod bert;

pub use bert::{BertModel, Config, PositionEmbeddingType};
use candle::{Result, Tensor};
pub use jina::JinaBertModel;
use text_embeddings_backend_core::Batch;

#[cfg(feature = "cuda")]
mod flash_bert;
mod jina;

#[cfg(feature = "cuda")]
pub use flash_bert::FlashBertModel;

pub(crate) trait Model {
    fn embed(&self, _batch: Batch) -> Result<Tensor> {
        candle::bail!("`embed` is not implemented for this model");
    }

    fn predict(&self, _batch: Batch) -> Result<Tensor> {
        candle::bail!("`predict is not implemented for this model");
    }
}
