//! LBNL Reranker model: Qwen3 + MLP Projector
use crate::layers::projector::Projector;
use crate::models::{qwen3::Qwen3Model, Model};
use candle::{DType, Device, IndexOp, Result as CResult, Tensor};
use candle_nn::VarBuilder;
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, ListwiseBlockInput, ListwiseBlockOutput,
};

pub struct LbnlReranker {
    qwen3: Qwen3Model,
    projector: Projector,
    device: Device,
    dtype: DType, // Track model's native dtype (BF16/FP16/F32)
}

impl LbnlReranker {
    pub fn new(
        vb: VarBuilder,
        qwen3: Qwen3Model,
        device: Device,
        hidden_size: usize,
        dtype: DType, // Model's loaded dtype
    ) -> CResult<Self> {
        // Load projector with same dtype as Qwen3 model to prevent mixed-precision issues
        // VarBuilder already has the correct dtype from initialization
        let projector = Projector::load(vb, hidden_size)?;
        Ok(Self {
            qwen3,
            projector,
            device,
            dtype,
        })
    }

    pub fn forward(&self, input: &ListwiseBlockInput) -> anyhow::Result<ListwiseBlockOutput> {
        let t = input.input_ids.len();
        let ids = Tensor::from_vec(input.input_ids.clone(), (1, t), &self.device)?;

        // Use attention mask from input (preserves left-padding structure from tokenizer)
        // Python reference: tokenizer creates mask with 0 for padding, 1 for real tokens
        // Ignoring this causes padding tokens to be treated as real tokens!
        let mask = Tensor::from_vec(input.attention_mask.clone(), (1, t), &self.device)?;

        // Use forward_with_tensors for hidden states extraction
        let hs = self.qwen3.forward_with_tensors(&ids, &mask)?;

        // Verify dtype matches expectation
        let hs = if hs.dtype() != self.dtype {
            tracing::warn!(
                "Hidden states dtype mismatch: got {:?}, expected {:?}",
                hs.dtype(),
                self.dtype
            );
            hs.to_dtype(self.dtype)?
        } else {
            hs
        };

        // Find special token positions
        let mut doc_pos = Vec::with_capacity(input.doc_count);
        let mut rerank_pos = None;
        for (i, &tid) in input.input_ids.iter().enumerate() {
            if tid == input.embed_token_id {
                doc_pos.push(i);
            }
            if tid == input.rerank_token_id {
                rerank_pos = Some(i);
            }
        }
        let qpos = rerank_pos.ok_or_else(|| anyhow::anyhow!("No rerank token found"))?;

        // Extract hidden states at positions → native dtype [1, H]
        let hq = hs.i((0, qpos, ..))?.unsqueeze(0)?;

        // Process documents: projector in native dtype, convert to F32 only for Vec extraction
        let mut doc_embs = Vec::with_capacity(doc_pos.len());
        for &p in &doc_pos {
            let hd = hs.i((0, p, ..))?.unsqueeze(0)?;
            // Projector operates in native dtype (BF16/FP16) - faster and more memory efficient
            let zd_native = self.projector.forward(&hd)?;
            // Convert to F32 only for Vec<f32> extraction
            let zd_f32 = zd_native.to_dtype(DType::F32)?;
            doc_embs.push(zd_f32.to_vec2::<f32>()?.remove(0));
        }

        // Process query: same dtype policy
        let zq_native = self.projector.forward(&hq)?;
        let zq_f32 = zq_native.to_dtype(DType::F32)?;
        let zq_vec = zq_f32.to_vec2::<f32>()?.remove(0);

        // Important normalization policy (modeling.py parity):
        // - Projector outputs are returned WITHOUT L2 normalization
        // - Router handler performs normalization inside cosine_similarity()
        // - This matches Python reference where normalize() is called in compute_scores()
        // - Normalizing here would cause double normalization!

        Ok(ListwiseBlockOutput {
            query_embedding: zq_vec,
            doc_embeddings: doc_embs,
        })
    }
}

// Implement Model trait for Candle backend integration
impl Model for LbnlReranker {
    fn is_padded(&self) -> bool {
        true // Qwen3 uses left padding
    }

    // LBNL reranker doesn't support standard embedding
    fn embed(&self, _batch: Batch) -> candle::Result<(Option<Tensor>, Option<Tensor>)> {
        candle::bail!("LBNL reranker only supports listwise reranking, not standard embedding")
    }

    // LBNL reranker doesn't support pairwise prediction
    fn predict(&self, _batch: Batch) -> candle::Result<Tensor> {
        candle::bail!("LBNL reranker only supports listwise reranking, not pairwise prediction")
    }
}

// Implement Backend trait (not a separate ListwiseBackend)
// This allows dispatch via Box<dyn Backend> without downcasting
impl Backend for LbnlReranker {
    fn health(&self) -> Result<(), BackendError> {
        Ok(()) // Model loaded successfully
    }

    fn is_padded(&self) -> bool {
        true // Qwen3 uses left padding
    }

    fn embed(
        &self,
        _batch: Batch,
    ) -> Result<text_embeddings_backend_core::Embeddings, BackendError> {
        Err(BackendError::Inference(
            "LBNL reranker only supports embed_listwise_block, not standard embedding".into(),
        ))
    }

    fn predict(
        &self,
        _batch: Batch,
    ) -> Result<text_embeddings_backend_core::Predictions, BackendError> {
        Err(BackendError::Inference(
            "LBNL reranker only supports embed_listwise_block, not pairwise prediction".into(),
        ))
    }

    // Override default implementation to provide listwise support
    fn embed_listwise_block(
        &self,
        input: ListwiseBlockInput,
    ) -> Result<ListwiseBlockOutput, BackendError> {
        self.forward(&input)
            .map_err(|e| BackendError::Inference(e.to_string()))
    }
}
