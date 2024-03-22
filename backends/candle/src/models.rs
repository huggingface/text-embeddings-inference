#[cfg(any(feature = "mkl", feature = "mkl-dynamic"))]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod bert;
mod distilbert;
mod jina;
mod nomic;

#[cfg(feature = "cuda")]
mod flash_bert;

#[cfg(feature = "cuda")]
mod flash_jina;

#[cfg(feature = "cuda")]
mod flash_nomic;

#[cfg(feature = "cuda")]
mod flash_distilbert;

pub use bert::{BertConfig, BertModel, PositionEmbeddingType};
use candle::{Result, Tensor};
pub use distilbert::{DistilBertConfig, DistilBertModel};
pub use jina::JinaBertModel;
pub use nomic::{NomicBertModel, NomicConfig};
use text_embeddings_backend_core::Batch;

#[cfg(feature = "cuda")]
pub use flash_bert::FlashBertModel;

#[cfg(feature = "cuda")]
pub use flash_jina::FlashJinaBertModel;

#[cfg(feature = "cuda")]
pub use flash_nomic::FlashNomicBertModel;

#[cfg(feature = "cuda")]
pub use flash_distilbert::FlashDistilBertModel;

pub(crate) trait Model {
    fn is_padded(&self) -> bool;

    fn embed(&self, _batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        candle::bail!("`embed` is not implemented for this model");
    }

    fn predict(&self, _batch: Batch) -> Result<Tensor> {
        candle::bail!("`predict is not implemented for this model");
    }
}
