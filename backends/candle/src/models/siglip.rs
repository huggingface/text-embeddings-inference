use crate::layers::{HiddenAct, LayerNorm, Linear};
use crate::models::Model;
use candle::{Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType};

// To be compatible with the original google repository
// handle the full config but we only care about the text part
#[derive(Debug, Clone, Deserialize)]
pub struct SiglipConfig {
    pub text_config: SiglipTextConfig,
}

fn default_text_vocab_size() -> usize {
    32000
}

fn default_text_hidden_size() -> usize {
    768
}

fn default_text_intermediate_size() -> usize {
    3072
}

fn default_text_num_hidden_layers() -> usize {
    12
}

fn default_text_num_attention_heads() -> usize {
    12
}

fn default_text_max_position_embeddings() -> usize {
    64
}

fn default_text_layer_norm_eps() -> f64 {
    1e-6
}


fn default_text_hidden_act() -> HiddenAct {
    HiddenAct::Gelu
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L27
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SiglipTextConfig {
    #[serde(default = "default_text_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_text_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_text_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_text_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_text_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_text_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_text_layer_norm_eps")]
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: HiddenAct,
}

#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(config: &SiglipTextConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = config.hidden_size;

        let query_weight = vb.pp("q_proj").get((embed_dim, embed_dim), "weight")?;
        let query_bias = vb.pp("q_proj").get(embed_dim, "bias")?;
        let q_proj = Linear::new(query_weight, Some(query_bias), None);

        let key_weight = vb.pp("k_proj").get((embed_dim, embed_dim), "weight")?;
        let key_bias = vb.pp("k_proj").get(embed_dim, "bias")?;
        let k_proj = Linear::new(key_weight, Some(key_bias), None);

        let value_weight = vb.pp("v_proj").get((embed_dim, embed_dim), "weight")?;
        let value_bias = vb.pp("v_proj").get(embed_dim, "bias")?;
        let v_proj = Linear::new(value_weight, Some(value_bias), None);

        let out_weight = vb.pp("out_proj").get((embed_dim, embed_dim), "weight")?;
        let out_bias = vb.pp("out_proj").get(embed_dim, "bias")?;
        let out_proj = Linear::new(out_weight, Some(out_bias), None);

        let num_heads = config.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let shape = (batch_size, q_len, self.num_heads, self.head_dim);
        let query_states = query_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let attn_weights = (query_states.matmul(&key_states.t()?)? * self.scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        // The original implementation upcasts to f32 but candle_nn::ops::softmax should handle this properly.
        let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_outputs = attn_scores
            .matmul(&value_states)?
            .transpose(1, 2)?
            .reshape((batch_size, q_len, ()))?;
        let attn_outputs = self.out_proj.forward(&attn_outputs)?;
        Ok(attn_outputs)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/modeling_siglip.py#L599
#[derive(Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    activation_fn: HiddenAct,
}

impl Mlp {
    fn new(config: &SiglipTextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let fc1_weight = vb
            .pp("fc1")
            .get((intermediate_size, hidden_size), "weight")?;
        let fc1_bias = vb.pp("fc1").get(intermediate_size, "bias")?;
        let fc1 = Linear::new(fc1_weight, Some(fc1_bias), None);
        let fc2_weight = vb
            .pp("fc2")
            .get((hidden_size, intermediate_size), "weight")?;
        let fc2_bias = vb.pp("fc2").get(hidden_size, "bias")?;
        // TODO: pass hidden_act directly to fc1
        let fc2 = Linear::new(fc2_weight, Some(fc2_bias), None);
        Ok(Self {
            fc1,
            fc2,
            activation_fn: config.hidden_act.clone(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &candle::Tensor) -> Result<candle::Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        Ok(xs)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/modeling_siglip.py#L614
#[derive(Debug)]
struct EncoderLayer {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: Mlp,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    fn new(config: &SiglipTextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let layer_norm_eps = config.layer_norm_eps;
        let self_attn = Attention::new(config, vb.pp("self_attn"))?;

        let layer_norm1 =
            LayerNorm::load(vb.pp("layer_norm1"), hidden_size, layer_norm_eps as f32)?;
        let mlp = Mlp::new(config, vb.pp("mlp"))?;
        let layer_norm2 =
            LayerNorm::load(vb.pp("layer_norm2"), hidden_size, layer_norm_eps as f32)?;
        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs, None)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs, None)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = (xs + residual)?;
        Ok(xs)
    }
}

#[derive(Debug)]
struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new(config: &SiglipTextConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = vec![];
        let vb = vb.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = EncoderLayer::new(config, vb.pp(layer_idx))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, attention_mask)?
        }
        Ok(xs)
    }
}

pub struct SiglipTextModel {
    embeddings: SiglipTextEmbeddings,
    encoder: Encoder,
    final_layer_norm: LayerNorm,
    pub head: Linear,
    max_position_embeddings: usize,
    pad_token_id: u32,
    device: Device,
}

impl SiglipTextModel {
    pub fn load(vb: VarBuilder, config: &SiglipTextConfig, model_type: ModelType) -> Result<Self> {
        // SigLIP always pools the final position and applies the projection head;
        // the configured pooling mode is irrelevant, but classification is unsupported.
        if let ModelType::Classifier = model_type {
            candle::bail!("SiglipTextModel only supports embedding mode")
        }

        let embeddings = SiglipTextEmbeddings::new(config, vb.pp("embeddings"))?;
        let encoder = Encoder::new(config, vb.pp("encoder"))?;
        let final_layer_norm = LayerNorm::load(
            vb.pp("final_layer_norm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        let head_weight = vb
            .pp("head")
            .get((config.hidden_size, config.hidden_size), "weight")?;
        let head_bias = vb.pp("head").get(config.hidden_size, "bias")?;
        let head = Linear::new(head_weight, Some(head_bias), None);

        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            head,
            max_position_embeddings: config.max_position_embeddings,
            pad_token_id: config.pad_token_id,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let batch_size = batch.len();
        // SigLIP is trained with inputs padded to a fixed length (`max_position_embeddings`,
        // 64) and no attention mask: it attends to every position, including padding.
        // Pooling then takes the final position (</s>/pad token), so the padded
        // width must be exactly 64 for the position embedding at index 63 to be correct.
        let padded_len = self.max_position_embeddings;

        let mut input_ids = Vec::with_capacity(batch_size * padded_len);
        for i in 0..batch_size {
            let start = batch.cumulative_seq_lengths[i] as usize;
            let end = batch.cumulative_seq_lengths[i + 1] as usize;
            // SigLIP operates on a fixed `padded_len` (64) token window. Truncate any longer
            // input to that width: an over-long sequence would otherwise leave the flattened
            // `input_ids` the wrong size and make the `Tensor::from_vec` reshape below fail with
            // a shape mismatch, taking down every request dynamically batched alongside it.
            let seq_len = (end - start).min(padded_len);

            for j in start..start + seq_len {
                input_ids.push(batch.input_ids[j]);
            }
            // Pad up to `padded_len` with pad token </s>.
            for _ in seq_len..padded_len {
                input_ids.push(self.pad_token_id);
            }
        }

        let input_ids = Tensor::from_vec(input_ids, (batch_size, padded_len), &self.device)?;

        let embedding_output = self.embeddings.forward(&input_ids)?;
        // No attention mask: SigLIP attends to all positions.
        let encoder_output = self.encoder.forward(&embedding_output, None)?;
        let last_hidden_state = self.final_layer_norm.forward(&encoder_output, None)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            // Pool the final position (index 63 under pad-to-64), then project with the head.
            let mut results = Vec::with_capacity(batch.pooled_indices.len());
            for &i in &batch.pooled_indices {
                results.push(
                    last_hidden_state
                        .i((i as usize, padded_len - 1))?
                        .unsqueeze(0)?,
                );
            }
            let pooled_tokens = Tensor::cat(&results, 0)?;
            Some(self.head.forward(&pooled_tokens)?)
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            // Flatten and drop padding tokens so the backend can slice by real seq length.
            let (b, l, h) = last_hidden_state.shape().dims3()?;
            let outputs = last_hidden_state.reshape((b * l, h))?;

            let mut final_indices: Vec<u32> = Vec::new();
            for &i in &batch.raw_indices {
                let i = i as usize;
                let start = i * padded_len;
                // Clamp to the padded width; longer sequences are truncated to `padded_len`
                // above, so reading their full original length would run past the row and pull
                // in the next sequence's tokens (or index out of bounds on the last row).
                let length = ((batch.cumulative_seq_lengths[i + 1]
                    - batch.cumulative_seq_lengths[i]) as usize)
                    .min(padded_len);
                for j in start..start + length {
                    final_indices.push(j as u32);
                }
            }

            let n = final_indices.len();
            let final_indices = Tensor::from_vec(final_indices, n, &self.device)?;
            Some(outputs.index_select(&final_indices, 0)?)
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for SiglipTextModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}

#[derive(Debug, Clone)]
struct SiglipTextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
    position_ids: Tensor,
}

impl SiglipTextEmbeddings {
    fn new(config: &SiglipTextConfig, vb: VarBuilder) -> Result<Self> {
        let token_embedding = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("token_embedding"),
        )?;
        let position_embedding = candle_nn::embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embedding"),
        )?;
        let position_ids =
            Tensor::arange(0u32, config.max_position_embeddings as u32, vb.device())?
                .unsqueeze(0)?;
        Ok(Self {
            token_embedding,
            position_embedding,
            position_ids,
        })
    }
}

impl Module for SiglipTextEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.dim(D::Minus1)?;
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        let position_ids = self.position_ids.narrow(1, 0, seq_length)?;
        let position_embedding = self.position_embedding.forward(&position_ids)?;
        inputs_embeds.broadcast_add(&position_embedding)
    }
}
