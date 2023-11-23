use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Dropout, Embedding, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, Pool};

use crate::alibi::build_alibi_tensor;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub layer_norm_epsilon: f64,
    pub initializer_range: f64,
    pub use_cache: bool,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub apply_residual_connection_post_layernorm: bool,
    pub hidden_dropout: f64,
    pub attention_dropout: f64,
    pub pretraining_tp: usize,
    pub slow_but_exact: bool,
    pub architectures: Vec<String>,
    pub attention_softmax_in_fp32: bool,
    pub seq_length: usize,
    pub pad_token_id: usize,
    pub masked_softmax_fusion: bool,
    pub model_type: Option<String>,
    pub n_embed: usize,
    pub num_attention_heads: usize,
}

pub struct BloomAttention {
    pretraining_tp: usize,
    slow_but_exact: bool,

    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    split_size: usize,
    hidden_dropout: f64,

    inv_norm_factor: f64,
    beta: f64,

    query_key_value: Linear,
    dense: Linear,
    attention_dropout: Dropout,
}

impl BloomAttention {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let pretraining_tp = config.pretraining_tp;
        let slow_but_exact = config.slow_but_exact;

        let hidden_size = config.hidden_size;
        let num_heads = config.n_head;
        let head_dim = config.hidden_size / config.n_head;
        let split_size = config.hidden_size;
        let hidden_dropout = config.hidden_dropout;

        if head_dim * num_heads != hidden_size {
            candle::bail!(
                "hidden_size must be divisible by num_heads (got hidden_size: {hidden_size} and num_heads: {num_heads}",
            )
        }

        let inv_norm_factor = 1. / (head_dim as f64).sqrt();
        let beta = 1.;

        let qkv_weight = vb
            .pp("query_key_value")
            .get((hidden_size * 3, hidden_size), "weight")?;
        let qkv_bias = vb.pp("query_key_value").get(hidden_size * 3, "bias")?;
        let query_key_value = Linear::new(qkv_weight, Some(qkv_bias));

        let dense_weight = vb
            .pp("query_key_value")
            .get((hidden_size, hidden_size), "weight")?;
        let dense_bias = vb.pp("dense").get(hidden_size, "bias")?;
        let dense = Linear::new(dense_weight, Some(dense_bias));

        let attention_dropout = Dropout::new(config.attention_dropout as f32);

        Ok(Self {
            pretraining_tp,
            slow_but_exact,
            hidden_size,
            num_heads,
            head_dim,
            split_size,
            hidden_dropout,
            inv_norm_factor,
            beta,
            query_key_value,
            dense,
            attention_dropout,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residual: &Tensor,
        alibi: &Tensor,
        attention_mask: &Tensor,
        layer_past: Option<&(Tensor, Tensor)>,
        head_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let fused_qkv = self.query_key_value.forward(hidden_states)?;

        let (batch_size, seq_length, _) = fused_qkv.dims3()?;
        let fused_qkv =
            fused_qkv.reshape((batch_size, seq_length, self.num_heads, 3, self.head_dim))?;
        let query_layer = fused_qkv.i((.., .., .., 0, ..))?;
        let key_layer = fused_qkv.i((.., .., .., 1, ..))?;
        let value_layer = fused_qkv.i((.., .., .., 2, ..))?;

        let (batch_size, q_length, _, _) = query_layer.dims4()?;

        let query_layer = query_layer.transpose(1, 2)?.reshape((
            batch_size * self.num_heads,
            q_length,
            self.head_dim,
        ))?;
        let mut key_layer = key_layer.permute((0, 2, 3, 1))?.reshape((
            batch_size * self.num_heads,
            self.head_dim,
            q_length,
        ))?;
        let mut value_layer = value_layer.transpose(1, 2)?.reshape((
            batch_size * self.num_heads,
            q_length,
            self.head_dim,
        ))?;
        if let Some((past_key, past_value)) = layer_past {
            key_layer = Tensor::cat(&[past_key, &key_layer], 2)?;
            value_layer = Tensor::cat(&[past_value, &value_layer], 1)?;
        }

        let (_, _, kv_length) = key_layer.dims3()?;

        // copied `torch.Tensor.baddbmm` impl from https://github.com/pytorch/xla/pull/2471/files
        let matmul_result =
            ((query_layer.matmul(&key_layer)? * self.inv_norm_factor)? + (self.beta * alibi)?)?;

        let mut attention_scores =
            matmul_result.i((batch_size, self.num_heads, q_length, kv_length))?;

        let input_dtype = attention_scores.dtype();
        if input_dtype == DType::F16 {
            attention_scores = attention_scores.to_dtype(DType::F32)?;
        }
        let attn_weights = masked_fill(&attention_scores, attention_mask, f32::MIN)?;

        let attention_probs =
            candle_nn::ops::softmax_last_dim(&attn_weights)?.to_dtype(input_dtype)?;

        let mut attention_probs = self.attention_dropout.forward(&attention_probs, false)?;

        if let Some(head_mask) = head_mask {
            attention_probs = (attention_probs * head_mask)?
        }

        let attention_probs_reshaped =
            attention_probs.i((batch_size * self.num_heads, q_length, kv_length))?;

        let context_layer = attention_probs_reshaped.matmul(&value_layer)?;
        let context_layer = context_layer
            .i((batch_size, self.num_heads, seq_length, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((batch_size, seq_length, self.num_heads * self.head_dim))?;

        let output_tensor = if self.pretraining_tp > 1 && self.slow_but_exact {
            let slices = self.hidden_size as f64 / self.pretraining_tp as f64;
            let mut output_tensor = Tensor::zeros_like(&context_layer)?;
            for i in 0..self.pretraining_tp {
                let start = (i as f64 * slices) as usize;
                let end = ((i + 1) as f64 * slices) as usize;
                output_tensor = (output_tensor
                    + Linear::new(self.dense.weight().i((.., start..end))?, None)
                        .forward(&context_layer.i((.., .., start..end))?)?)?
            }
            output_tensor
        } else {
            self.dense.forward(&context_layer)?
        };
        // dropout add?
        let output_tensor = (residual + output_tensor)?;

        Ok(output_tensor)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

pub struct BloomMlp {
    hidden_size: usize,

    pretraining_tp: usize,
    slow_but_exact: bool,
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    hidden_dropout: f64,
}

impl BloomMlp {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let hidden_size = config.hidden_size;

        let pretraining_tp = config.pretraining_tp;
        let slow_but_exact = config.slow_but_exact;

        let dense_h_to_4h_weight = vb
            .pp("dense_h_to_4h")
            .get((hidden_size * 4, hidden_size), "weight")?;
        let dense_h_to_4h_bias = vb.pp("dense_h_to_4h").get(hidden_size * 4, "bias")?;
        let dense_h_to_4h = Linear::new(dense_h_to_4h_weight, Some(dense_h_to_4h_bias));

        let dense_4h_to_h_weight = vb
            .pp("dense_4h_to_h")
            .get((hidden_size, hidden_size * 4), "weight")?;
        let dense_4h_to_h_bias = vb.pp("dense_4h_to_h").get(hidden_size, "bias")?;
        let dense_4h_to_h = Linear::new(dense_4h_to_h_weight, Some(dense_4h_to_h_bias));

        let hidden_dropout = config.hidden_dropout;

        Ok(Self {
            hidden_size,
            pretraining_tp,
            slow_but_exact,
            dense_h_to_4h,
            dense_4h_to_h,
            hidden_dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense_h_to_4h.forward(hidden_states)?.gelu()?;

        let intermediate_output = if self.pretraining_tp > 1 && self.slow_but_exact {
            let mut intermediate_output = Tensor::zeros_like(residual)?;
            let slices = *self.dense_4h_to_h.weight().shape().dims().last().unwrap() as f64
                / self.pretraining_tp as f64;
            for i in 0..self.pretraining_tp {
                let start = (i as f64 * slices) as usize;
                let end = ((i + 1) as f64 * slices) as usize;
                intermediate_output = (intermediate_output
                    + Linear::new(self.dense_4h_to_h.weight().i((.., start..end))?, None)
                        .forward(&hidden_states.i((.., .., start..end))?)?)?;
            }
            intermediate_output
        } else {
            self.dense_4h_to_h.forward(&hidden_states)?
        };
        // dropout add?
        residual + intermediate_output
    }
}

pub struct BloomBlock {
    hidden_size: usize,
    input_layernorm: LayerNorm,
    num_heads: usize,
    self_attention: BloomAttention,
    post_attention_layernorm: LayerNorm,
    mlp: BloomMlp,
    apply_residual_connection_post_layernorm: bool,
    hidden_dropout: f64,
}

impl BloomBlock {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let hidden_size = config.hidden_size;

        let input_layernorm = LayerNorm::new(
            vb.pp("input_layernorm").get(hidden_size, "weight")?,
            vb.pp("input_layernorm").get(hidden_size, "bias")?,
            config.layer_norm_epsilon,
        );
        let num_heads = config.n_head;
        let self_attention = BloomAttention::load(vb.pp("self_attention"), config)?;
        let post_attention_layernorm = LayerNorm::new(
            vb.pp("post_attention_layernorm")
                .get(hidden_size, "weight")?,
            vb.pp("post_attention_layernorm").get(hidden_size, "bias")?,
            config.layer_norm_epsilon,
        );

        let mlp = BloomMlp::load(vb.pp("mlp"), config)?;

        let apply_residual_connection_post_layernorm =
            config.apply_residual_connection_post_layernorm;
        let hidden_dropout = config.hidden_dropout;

        Ok(Self {
            hidden_size,
            input_layernorm,
            num_heads,
            self_attention,
            post_attention_layernorm,
            mlp,
            apply_residual_connection_post_layernorm,
            hidden_dropout,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        alibi: &Tensor,
        attention_mask: &Tensor,
        layer_past: Option<&(Tensor, Tensor)>,
        head_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let layernorm_output = self.input_layernorm.forward(hidden_states)?;

        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            hidden_states
        };

        let attn_output = self.self_attention.forward(
            &layernorm_output,
            residual,
            alibi,
            attention_mask,
            layer_past,
            head_mask,
        )?;

        let layernorm_output = self.post_attention_layernorm.forward(&attn_output)?;

        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            &attn_output
        };

        let output = self.mlp.forward(&layernorm_output, residual)?;

        Ok(output)
    }
}

pub struct BloomModel {
    embed_dim: usize,
    num_heads: usize,
    word_embeddings: Embedding,
    word_embeddings_layernorm: LayerNorm,
    h: Vec<BloomBlock>,
    ln_f: LayerNorm,

    device: Device,
    dtype: DType,

    span: tracing::Span,
}

impl BloomModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.n_head;

        let word_embeddings_weights = vb.pp("word_embeddings").get(config.vocab_size, "weight")?;
        let word_embeddings = Embedding::new(word_embeddings_weights, embed_dim);

        let word_embeddings_layernorm = LayerNorm::new(
            vb.pp("word_embeddings_layernorm")
                .get(embed_dim, "weight")?,
            vb.pp("word_embeddings_layernorm").get(embed_dim, "bias")?,
            config.layer_norm_epsilon,
        );

        let h = (0..config.n_layer)
            .map(|index| BloomBlock::load(vb.pp(format!("h.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let ln_f = LayerNorm::new(
            vb.pp("ln_f").get(config.hidden_size, "weight")?,
            vb.pp("ln_f").get(config.hidden_size, "bias")?,
            config.layer_norm_epsilon,
        );

        Ok(Self {
            embed_dim,
            num_heads,
            word_embeddings,
            word_embeddings_layernorm,
            h,
            ln_f,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<Tensor> {
        let batch_size = batch.cumulative_seq_lengths.len() - 1;
        let max_length = batch.max_length as usize;

        let shape = (batch_size, max_length);

        let (input_ids, type_ids, position_ids, input_lengths) = if batch_size > 1 {
            // Prepare padded batch
            let elems = batch_size * max_length;

            let mut input_ids = Vec::with_capacity(elems);
            let mut type_ids = Vec::with_capacity(elems);
            let mut position_ids = Vec::with_capacity(elems);
            let mut attention_mask = Vec::with_capacity(elems);
            let mut attention_bias = Vec::with_capacity(elems);
            let mut input_lengths = Vec::with_capacity(batch_size);
            // Bool to know if we need to use the attention mask
            let mut masking = false;

            for i in 0..batch_size {
                let start = batch.cumulative_seq_lengths[i] as usize;
                let end = batch.cumulative_seq_lengths[i + 1] as usize;
                let seq_length = (end - start) as u32;
                input_lengths.push(seq_length as f32);

                // Copy values
                for j in start..end {
                    input_ids.push(batch.input_ids[j]);
                    type_ids.push(batch.token_type_ids[j]);
                    position_ids.push(batch.position_ids[j]);
                    attention_mask.push(1.0);
                    attention_bias.push(0.0);
                }

                // Add padding if needed
                let padding = batch.max_length - seq_length;
                if padding > 0 {
                    // Set bool to use attention mask
                    masking = true;
                    for _ in 0..padding {
                        input_ids.push(0);
                        type_ids.push(0);
                        position_ids.push(0);
                        attention_mask.push(0.0);
                        attention_bias.push(f32::NEG_INFINITY);
                    }
                }
            }

            let (attention_bias, attention_mask) = match masking {
                true => {
                    // We only need the mask if we use mean pooling
                    // For CLS pooling, the bias is enough
                    let attention_mask = if self.pool == Pool::Mean {
                        let attention_mask = Tensor::from_vec(
                            attention_mask,
                            (batch_size, max_length, 1),
                            &self.device,
                        )?
                        .to_dtype(self.dtype)?;

                        Some(attention_mask)
                    } else {
                        None
                    };

                    let attention_bias = Tensor::from_vec(
                        attention_bias,
                        (batch_size, 1, 1, max_length),
                        &self.device,
                    )?
                    .to_dtype(self.dtype)?;
                    // Broadcast once instead of at every layer
                    let attention_bias = attention_bias
                        .broadcast_as((
                            batch_size,
                            self.num_attention_heads,
                            max_length,
                            max_length,
                        ))?
                        .contiguous()?;
                    (Some(attention_bias), attention_mask)
                }
                false => (None, None),
            };

            (input_ids, type_ids, position_ids, input_lengths)
        } else {
            (
                batch.input_ids,
                batch.token_type_ids,
                batch.position_ids,
                vec![batch.max_length as f32],
            )
        };

        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let input_embeds = self.word_embeddings.forward(&input_ids)?;
        let mut hidden_states = self.word_embeddings_layernorm.forward(&input_embeds)?;
        let alibi = build_alibi_tensor(self.embed_dim, self.num_heads, &self.device, self.dtype)?;
        // causal mask?
        for (i, block) in self.h.iter().enumerate() {
            hidden_states = block.forward(&hidden_states, &alibi, None, None, None)?;
        }

        self.ln_f.forward(&hidden_states)
    }
}
