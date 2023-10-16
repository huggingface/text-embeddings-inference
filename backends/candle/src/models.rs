#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod bert;
mod bert_quant;

pub use bert::{BertModel, Config};
pub use bert_quant::QuantBertModel;
use candle::{Result, Tensor};
use text_embeddings_backend_core::Batch;

#[cfg(feature = "cuda")]
mod flash_bert;
#[cfg(feature = "cuda")]
pub use flash_bert::FlashBertModel;

pub(crate) trait EmbeddingModel {
    fn embed(&self, batch: Batch) -> Result<Tensor>;
}
