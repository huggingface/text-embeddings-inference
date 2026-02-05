use std::collections::HashMap;

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{conv1d, Conv1d, Conv1dConfig, Embedding, VarBuilder};
use serde::{Deserialize, Deserializer};

use text_embeddings_backend_core::{Batch, ModelType, Pool};

use crate::layers::{HiddenAct, LayerNorm, Linear};
use crate::models::Model;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

pub type Id2Label = HashMap<String, String>;
pub type Label2Id = HashMap<String, u32>;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DebertaV2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub relative_attention: bool,
    pub max_relative_positions: isize,
    pub pad_token_id: Option<usize>,
    pub position_biased_input: bool,
    #[serde(deserialize_with = "deserialize_pos_att_type")]
    pub pos_att_type: Vec<String>,
    pub position_buckets: Option<isize>,
    pub share_att_key: Option<bool>,
    pub attention_head_size: Option<usize>,
    pub embedding_size: Option<usize>,
    pub norm_rel_ebd: Option<String>,
    pub conv_kernel_size: Option<usize>,
    pub conv_groups: Option<usize>,
    pub conv_act: Option<String>,
    pub id2label: Option<Id2Label>,
    pub label2id: Option<Label2Id>,
    pub pooler_hidden_act: Option<HiddenAct>,
    pub pooler_hidden_size: Option<usize>,
}

// NOTE: https://huggingface.co/microsoft/deberta-v3-base/blob/main/config.json#L14
fn deserialize_pos_att_type<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize, Debug)]
    #[serde(untagged)]
    enum StringOrVec {
        String(String),
        Vec(Vec<String>),
    }

    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::String(s) => Ok(s.split('|').map(String::from).collect()),
        StringOrVec::Vec(v) => Ok(v),
    }
}

pub struct DebertaV2Embeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    position_biased_input: bool,
    type_vocab_size: usize,
    hidden_size: usize,
    embedding_size: usize,
    embed_proj: Option<Linear>,
    span: tracing::Span,
}

impl DebertaV2Embeddings {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let embedding_size = config.embedding_size.unwrap_or(config.hidden_size);

        let word_embeddings = Embedding::new(
            vb.pp("word_embeddings")
                .get((config.vocab_size, embedding_size), "weight")?,
            embedding_size,
        );

        let position_embeddings = if config.position_biased_input {
            Some(Embedding::new(
                vb.pp("position_embeddings")
                    .get((config.max_position_embeddings, embedding_size), "weight")?,
                embedding_size,
            ))
        } else {
            None
        };

        let token_type_embeddings = if config.type_vocab_size > 0 {
            Some(Embedding::new(
                vb.pp("token_type_embeddings")
                    .get((config.type_vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ))
        } else {
            None
        };

        let embed_proj = if embedding_size != config.hidden_size {
            let weight = vb
                .pp("embed_proj")
                .get((config.hidden_size, embedding_size), "weight")?;
            Some(Linear::new(weight, None, None))
        } else {
            None
        };

        let layer_norm = LayerNorm::load(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            position_biased_input: config.position_biased_input,
            type_vocab_size: config.type_vocab_size,
            hidden_size: config.hidden_size,
            embedding_size,
            embed_proj,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_embeds = self.word_embeddings.forward(input_ids)?;

        let position_embeddings = match &self.position_embeddings {
            Some(emb) => emb.forward(position_ids)?,
            None => Tensor::zeros_like(&input_embeds)?,
        };

        let mut embeddings = input_embeds;

        if self.position_biased_input {
            embeddings = embeddings.add(&position_embeddings)?;
        }

        if self.type_vocab_size > 0 {
            embeddings = self.token_type_embeddings.as_ref().map_or_else(
                || candle::bail!("token_type_embeddings must be set when type_vocab_size > 0"),
                |token_type_embeddings| {
                    embeddings.add(&token_type_embeddings.forward(token_type_ids)?)
                },
            )?;
        }

        if self.embedding_size != self.hidden_size {
            embeddings = if let Some(embed_proj) = &self.embed_proj {
                embed_proj.forward(&embeddings)?
            } else {
                candle::bail!("embed_proj must exist if embedding_size != hidden_size");
            }
        }

        embeddings = self.layer_norm.forward(&embeddings, None)?;

        if let Some(mask) = mask {
            let mut mask = mask.clone();
            if mask.dims() != embeddings.dims() {
                if mask.dims().len() == 4 {
                    mask = mask.squeeze(1)?.squeeze(1)?;
                }
                mask = mask.unsqueeze(2)?;
            }

            mask = mask.to_dtype(embeddings.dtype())?;
            embeddings = embeddings.broadcast_mul(&mask)?;
        }

        Ok(embeddings)
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L605
pub struct DebertaV2DisentangledSelfAttention {
    num_attention_heads: usize,
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    device: Device,
    relative_attention: bool,
    position_buckets: isize,
    max_relative_positions: isize,
    pos_ebd_size: isize,
    share_att_key: bool,
    pos_key_proj: Option<Linear>,
    pos_query_proj: Option<Linear>,
    is_c2p_attn: bool,
    is_p2c_attn: bool,
    base_scale: Tensor,
    span: tracing::Span,
}

impl DebertaV2DisentangledSelfAttention {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(candle::Error::Msg(format!(
                "The hidden size {} is not a multiple of the number of attention heads {}",
                config.hidden_size, config.num_attention_heads
            )));
        }

        let num_attention_heads = config.num_attention_heads;

        let attention_head_size = config
            .attention_head_size
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        let all_head_size = num_attention_heads * attention_head_size;

        let query_proj = Linear::new(
            vb.pp("query_proj")
                .get((all_head_size, config.hidden_size), "weight")?,
            Some(vb.pp("query_proj").get(all_head_size, "bias")?),
            None,
        );
        let key_proj = Linear::new(
            vb.pp("key_proj")
                .get((all_head_size, config.hidden_size), "weight")?,
            Some(vb.pp("key_proj").get(all_head_size, "bias")?),
            None,
        );
        let value_proj = Linear::new(
            vb.pp("value_proj")
                .get((all_head_size, config.hidden_size), "weight")?,
            Some(vb.pp("value_proj").get(all_head_size, "bias")?),
            None,
        );

        let share_att_key = config.share_att_key.unwrap_or(false);
        let relative_attention = config.relative_attention;
        let mut max_relative_positions = config.max_relative_positions;

        // Precompute attention type checks
        let is_c2p_attn = config.pos_att_type.iter().any(|s| s == "c2p");
        let is_p2c_attn = config.pos_att_type.iter().any(|s| s == "p2c");

        let mut pos_ebd_size: isize = 0;
        let position_buckets = config.position_buckets.unwrap_or(-1);
        let mut pos_key_proj: Option<Linear> = None;
        let mut pos_query_proj: Option<Linear> = None;

        if relative_attention {
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings as isize;
            }
            pos_ebd_size = max_relative_positions;
            if position_buckets > 0 {
                pos_ebd_size = position_buckets
            }

            if !share_att_key {
                if is_c2p_attn {
                    pos_key_proj = Some(Linear::new(
                        vb.pp("pos_key_proj")
                            .get((all_head_size, config.hidden_size), "weight")?,
                        Some(vb.pp("pos_key_proj").get(all_head_size, "bias")?),
                        None,
                    ));
                }
                if is_p2c_attn {
                    pos_query_proj = Some(Linear::new(
                        vb.pp("pos_query_proj")
                            .get((all_head_size, config.hidden_size), "weight")?,
                        Some(vb.pp("pos_query_proj").get(all_head_size, "bias")?),
                        None,
                    ));
                }
            }
        }

        let device = vb.device().clone();

        // Pre-compute base scale tensor to avoid recreating it on every forward pass
        // q_size equals attention_head_size, so we compute sqrt(attention_head_size) once
        let base_scale = Tensor::new(&[attention_head_size as f32], &device)?.sqrt()?;

        Ok(Self {
            num_attention_heads,
            query_proj,
            key_proj,
            value_proj,
            device,
            relative_attention,
            position_buckets,
            max_relative_positions,
            pos_ebd_size,
            share_att_key,
            pos_key_proj,
            pos_query_proj,
            is_c2p_attn,
            is_p2c_attn,
            base_scale,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query_states = match query_states {
            Some(qs) => qs,
            None => hidden_states,
        };

        let query_layer = self.transpose_for_scores(&self.query_proj.forward(query_states)?)?;
        let key_layer = self.transpose_for_scores(&self.key_proj.forward(query_states)?)?;
        let value_layer = self.transpose_for_scores(&self.value_proj.forward(query_states)?)?;

        let mut scale_factor: usize = 1;

        if self.is_c2p_attn {
            scale_factor += 1;
        }

        if self.is_p2c_attn {
            scale_factor += 1;
        }

        // Use pre-computed base_scale and multiply by sqrt(scale_factor)
        // This is mathematically equivalent to sqrt(q_size * scale_factor)
        let scale = self
            .base_scale
            .broadcast_mul(&Tensor::new(&[(scale_factor as f32).sqrt()], &self.device)?)?;

        let mut attention_scores: Tensor = {
            let key_layer_transposed = key_layer.t()?;
            let div = key_layer_transposed
                .broadcast_div(scale.to_dtype(query_layer.dtype())?.as_ref())?;
            query_layer.matmul(&div)?
        };

        let rel_att = if self.relative_attention {
            if let Some(rel_embeddings) = rel_embeddings {
                Some(self.disentangled_attention_bias(
                    query_layer,
                    key_layer,
                    relative_pos,
                    rel_embeddings.clone(),
                    scale_factor,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(rel_att) = rel_att {
            attention_scores = attention_scores.broadcast_add(&rel_att)?;
        }

        attention_scores = attention_scores.reshape((
            (),
            self.num_attention_heads,
            attention_scores.dim(D::Minus2)?,
            attention_scores.dim(D::Minus1)?,
        ))?;

        // Add attention mask bias and apply softmax (ModernBERT approach)
        let attention_mask = attention_mask.to_dtype(attention_scores.dtype())?;
        let attention_probs = attention_scores.broadcast_add(&attention_mask)?;
        let attention_probs = candle_nn::ops::softmax(&attention_probs, D::Minus1)?;

        let mut context_layer = attention_probs
            .reshape((
                (),
                attention_probs.dim(D::Minus2)?,
                attention_probs.dim(D::Minus1)?,
            ))?
            .matmul(&value_layer)?;

        context_layer = context_layer
            .reshape((
                (),
                self.num_attention_heads,
                context_layer.dim(D::Minus2)?,
                context_layer.dim(D::Minus1)?,
            ))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let dims = context_layer.dims();

        context_layer = match dims.len() {
            2 => context_layer.reshape(())?,
            3 => context_layer.reshape((dims[0], ()))?,
            4 => context_layer.reshape((dims[0], dims[1], ()))?,
            5 => context_layer.reshape((dims[0], dims[1], dims[2], ()))?,
            _ => {
                candle::bail!(
                    "Invalid shape for DisentangledSelfAttention context layer: {:?}",
                    dims
                )
            }
        };

        Ok(context_layer)
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims().to_vec();
        match dims.len() {
            3 => {
                let reshaped = xs.reshape((dims[0], dims[1], self.num_attention_heads, ()))?;

                reshaped.transpose(1, 2)?.contiguous()?.reshape((
                    (),
                    reshaped.dim(1)?,
                    reshaped.dim(D::Minus1)?,
                ))
            }
            shape => {
                candle::bail!(
                    "Invalid shape for transpose_for_scores. Expected 3 dimensions, got {shape}"
                )
            }
        }
    }

    fn disentangled_attention_bias(
        &self,
        query_layer: Tensor,
        key_layer: Tensor,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Tensor,
        scale_factor: usize,
    ) -> Result<Tensor> {
        let mut relative_pos = relative_pos.map_or(
            build_relative_position(
                query_layer.dim(D::Minus2)?,
                key_layer.dim(D::Minus2)?,
                &self.device,
                Some(self.position_buckets),
                Some(self.max_relative_positions),
            )?,
            |pos| pos.clone(),
        );

        relative_pos = match relative_pos.dims().len() {
            2 => relative_pos.unsqueeze(0)?.unsqueeze(0)?,
            3 => relative_pos.unsqueeze(1)?,
            other => {
                candle::bail!(
                    "Relative position ids must be of dim 2 or 3 or 4. Got dim of size {other}"
                )
            }
        };

        let att_span = self.pos_ebd_size;

        let rel_embeddings = rel_embeddings
            .narrow(0, 0, (att_span * 2) as usize)?
            .unsqueeze(0)?;

        let mut pos_query_layer: Option<Tensor> = None;
        let mut pos_key_layer: Option<Tensor> = None;

        let repeat_with = query_layer.dim(0)? / self.num_attention_heads;
        if self.share_att_key {
            pos_query_layer = Some(
                self.transpose_for_scores(&self.query_proj.forward(&rel_embeddings)?)?
                    .repeat(repeat_with)?,
            );

            pos_key_layer = Some(
                self.transpose_for_scores(&self.key_proj.forward(&rel_embeddings)?)?
                    .repeat(repeat_with)?,
            )
        } else {
            if self.is_c2p_attn {
                pos_key_layer = Some(
                    self.transpose_for_scores(
                        &self
                            .pos_key_proj
                            .as_ref()
                            .context(
                                "Need pos_key_proj when share_att_key is false or not specified",
                            )?
                            .forward(&rel_embeddings)?,
                    )?
                    .repeat(repeat_with)?,
                )
            }
            if self.is_p2c_attn {
                pos_query_layer = Some(self.transpose_for_scores(&self
                    .pos_query_proj
                    .as_ref()
                    .context("Need a pos_query_proj when share_att_key is false or not specified")?
                    .forward(&rel_embeddings)?)?.repeat(repeat_with)?)
            }
        }

        // Initialize score tensor with the same dtype as query_layer to avoid dtype mismatches
        let mut score = Tensor::new(&[0 as f32], &self.device)?.to_dtype(query_layer.dtype())?;

        if self.is_c2p_attn {
            let pos_key_layer = pos_key_layer.context("c2p without pos_key_layer")?;

            let scale = Tensor::new(
                &[(pos_key_layer.dim(D::Minus1)? * scale_factor) as f32],
                &self.device,
            )?
            .sqrt()?;

            let mut c2p_att = query_layer.matmul(&pos_key_layer.t()?)?;

            let c2p_pos = relative_pos
                .broadcast_add(
                    &Tensor::new(&[att_span as i64], &self.device)?
                        .to_dtype(relative_pos.dtype())?,
                )?
                .clamp(0 as f32, (att_span * 2 - 1) as f32)?;

            c2p_att = c2p_att.gather(
                &c2p_pos
                    .squeeze(0)?
                    .expand(&[
                        query_layer.dim(0)?,
                        query_layer.dim(1)?,
                        relative_pos.dim(D::Minus1)?,
                    ])?
                    .contiguous()?,
                D::Minus1,
            )?;

            score = score.broadcast_add(
                &c2p_att.broadcast_div(scale.to_dtype(c2p_att.dtype())?.as_ref())?,
            )?;
        }

        if self.is_p2c_attn {
            let pos_query_layer = pos_query_layer.context("p2c without pos_key_layer")?;

            let scale = Tensor::new(
                &[(pos_query_layer.dim(D::Minus1)? * scale_factor) as f32],
                &self.device,
            )?
            .sqrt()?;

            let r_pos = {
                if key_layer.dim(D::Minus2)? != query_layer.dim(D::Minus2)? {
                    build_relative_position(
                        key_layer.dim(D::Minus2)?,
                        key_layer.dim(D::Minus2)?,
                        &self.device,
                        Some(self.position_buckets),
                        Some(self.max_relative_positions),
                    )?
                    .unsqueeze(0)?
                } else {
                    relative_pos
                }
            };

            let p2c_pos = r_pos
                .to_dtype(DType::F32)?
                .neg()?
                .broadcast_add(&Tensor::new(&[att_span as f32], &self.device)?)?
                .clamp(0f32, (att_span * 2 - 1) as f32)?;

            let p2c_att = key_layer
                .matmul(&pos_query_layer.t()?)?
                .gather(
                    &p2c_pos
                        .squeeze(0)?
                        .expand(&[
                            query_layer.dim(0)?,
                            key_layer.dim(D::Minus2)?,
                            key_layer.dim(D::Minus2)?,
                        ])?
                        .contiguous()?
                        .to_dtype(DType::U32)?,
                    D::Minus1,
                )?
                .t()?;

            score =
                score.broadcast_add(&p2c_att.broadcast_div(&scale.to_dtype(p2c_att.dtype())?)?)?;
        }

        Ok(score)
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L270
pub struct DebertaV2Attention {
    dsa: DebertaV2DisentangledSelfAttention,
    output: DebertaV2SelfOutput,
    span: tracing::Span,
}

impl DebertaV2Attention {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let dsa = DebertaV2DisentangledSelfAttention::load(vb.pp("attention.self"), config)?;
        let output = DebertaV2SelfOutput::load(vb.pp("attention.output"), config)?;
        Ok(Self {
            dsa,
            output,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_output = self.dsa.forward(
            hidden_states,
            attention_mask,
            query_states,
            relative_pos,
            rel_embeddings,
        )?;

        self.output
            .forward(&self_output, query_states.unwrap_or(hidden_states))
    }
}

// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L255
pub struct DebertaV2SelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DebertaV2SelfOutput {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let dense = Linear::new(
            vb.pp("dense")
                .get((config.hidden_size, config.hidden_size), "weight")?,
            Some(vb.pp("dense").get(config.hidden_size, "bias")?),
            None,
        );
        let layer_norm = LayerNorm::load(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;
        Ok(Self {
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "self-output"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layer_norm.forward(&hidden_states, Some(input_tensor))
    }
}

pub struct DebertaV2Intermediate {
    dense: Linear,
    span: tracing::Span,
}

impl DebertaV2Intermediate {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let dense = Linear::new(
            vb.pp("intermediate.dense")
                .get((config.intermediate_size, config.hidden_size), "weight")?,
            Some(
                vb.pp("intermediate.dense")
                    .get(config.intermediate_size, "bias")?,
            ),
            Some(config.hidden_act.clone()),
        );
        Ok(Self {
            dense,
            span: tracing::span!(tracing::Level::TRACE, "intermediate"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.dense.forward(hidden_states)
    }
}

pub struct DebertaV2Output {
    dense: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl DebertaV2Output {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let dense = Linear::new(
            vb.pp("output.dense")
                .get((config.hidden_size, config.intermediate_size), "weight")?,
            Some(vb.pp("output.dense").get(config.hidden_size, "bias")?),
            None,
        );
        let layer_norm = LayerNorm::load(
            vb.pp("output.LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;
        Ok(Self {
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layer_norm.forward(&hidden_states, Some(input_tensor))
    }
}

pub struct DebertaV2Layer {
    attention: DebertaV2Attention,
    intermediate: DebertaV2Intermediate,
    output: DebertaV2Output,
    span: tracing::Span,
}

impl DebertaV2Layer {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let attention = DebertaV2Attention::load(vb.clone(), config)?;
        let intermediate = DebertaV2Intermediate::load(vb.clone(), config)?;
        let output = DebertaV2Output::load(vb.clone(), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attention_output = self.attention.forward(
            hidden_states,
            attention_mask,
            query_states,
            relative_pos,
            rel_embeddings,
        )?;

        let intermediate_output = self.intermediate.forward(&attention_output)?;

        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;

        Ok(layer_output)
    }
}

// TODO: In order to fully test ConvLayer a model needs to be found has a configuration where
// `conv_kernel_size` exists and is > 0
// https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L373
pub struct ConvLayer {
    _conv_act: String,
    _conv: Conv1d,
    _layer_norm: LayerNorm,
}

impl ConvLayer {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let groups = config.conv_groups.unwrap_or(1);
        let conv_act = config
            .conv_act
            .clone()
            .unwrap_or_else(|| "tanh".to_string());

        let conv_conf = Conv1dConfig {
            padding: (kernel_size - 1) / 2,
            groups,
            ..Default::default()
        };

        let conv = conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            conv_conf,
            vb.pp("conv"),
        )?;

        let layer_norm = LayerNorm::load(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        Ok(Self {
            _conv_act: conv_act,
            _conv: conv,
            _layer_norm: layer_norm,
        })
    }

    pub fn forward(
        &self,
        _hidden_states: &Tensor,
        _residual_states: &Tensor,
        _input_mask: &Tensor,
    ) -> Result<Tensor> {
        todo!("Need a model that contains a conv layer to test against.")
    }
}

pub struct DebertaV2Encoder {
    layer: Vec<DebertaV2Layer>,
    relative_attention: bool,
    max_relative_positions: isize,
    position_buckets: isize,
    rel_embeddings: Option<Embedding>,
    norm_rel_ebd: String,
    layer_norm: Option<LayerNorm>,
    conv: Option<ConvLayer>,
    device: Device,
    span: tracing::Span,
}

impl DebertaV2Encoder {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let layer = (0..config.num_hidden_layers)
            .map(|index| DebertaV2Layer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let relative_attention = config.relative_attention;
        let mut max_relative_positions = config.max_relative_positions;

        let position_buckets = config.position_buckets.unwrap_or(-1);

        let mut rel_embeddings: Option<Embedding> = None;

        if relative_attention {
            if max_relative_positions < 1 {
                max_relative_positions = config.max_position_embeddings as isize;
            }

            let mut pos_ebd_size = max_relative_positions * 2;

            if position_buckets > 0 {
                pos_ebd_size = position_buckets * 2;
            }

            rel_embeddings = Some(Embedding::new(
                vb.pp("rel_embeddings")
                    .get((pos_ebd_size as usize, config.hidden_size), "weight")?,
                config.hidden_size,
            ));
        }

        // NOTE: The Python counterpart in Transformers assumes that the config attribute
        // `norm_rel_ebd` is an array, but most examples have it as a string.
        let norm_rel_ebd = match config.norm_rel_ebd.as_ref() {
            Some(nre) => nre.trim().to_string(),
            None => "none".to_string(),
        };

        let layer_norm = if norm_rel_ebd == "layer_norm" {
            Some(LayerNorm::load(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps as f32,
            )?)
        } else {
            None
        };

        let conv: Option<ConvLayer> = if config.conv_kernel_size.unwrap_or(0) > 0 {
            Some(ConvLayer::load(vb.pp("conv"), config)?)
        } else {
            None
        };

        Ok(Self {
            layer,
            relative_attention,
            max_relative_positions,
            position_buckets,
            rel_embeddings,
            norm_rel_ebd,
            layer_norm,
            conv,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "encoder"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_mask = if attention_mask.dims().len() <= 2 {
            attention_mask.clone()
        } else {
            attention_mask
                .sum_keepdim(attention_mask.rank() - 2)?
                .gt(0.)?
        };

        let attention_mask = self.get_attention_mask(attention_mask.clone())?;

        let relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)?;

        let mut next_kv: Tensor = hidden_states.clone();
        let rel_embeddings = self.get_rel_embedding()?;
        let mut output_states = next_kv.to_owned();
        let mut query_states: Option<Tensor> = query_states.cloned();

        for (i, layer_module) in self.layer.iter().enumerate() {
            output_states = layer_module.forward(
                next_kv.as_ref(),
                &attention_mask,
                query_states.as_ref(),
                relative_pos.as_ref(),
                rel_embeddings.as_ref(),
            )?;

            if i == 0 {
                if let Some(conv) = &self.conv {
                    output_states = conv.forward(hidden_states, &output_states, &input_mask)?;
                }
            }

            if query_states.is_some() {
                query_states = Some(output_states.clone());
            } else {
                next_kv = output_states.clone();
            }
        }

        Ok(output_states)
    }

    fn get_attention_mask(&self, mut attention_mask: Tensor) -> Result<Tensor> {
        match attention_mask.dims().len() {
            0..=2 => {
                let extended_attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
                attention_mask = extended_attention_mask.broadcast_mul(
                    &extended_attention_mask
                        .squeeze(D::Minus2)?
                        .unsqueeze(D::Minus1)?,
                )?;
            }
            3 => attention_mask = attention_mask.unsqueeze(1)?,
            len => candle::bail!("Unsupported attentiom mask size length: {len}"),
        }

        // Convert binary mask to additive bias: 0 for valid positions, large negative for masked
        let one = Tensor::ones_like(&attention_mask)?;
        let bias = attention_mask.broadcast_sub(&one)?.broadcast_mul(
            &Tensor::new(&[10000.0_f32], &self.device)?.to_dtype(attention_mask.dtype())?,
        )?;

        Ok(bias)
    }

    fn get_rel_pos(
        &self,
        hidden_states: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        if self.relative_attention && relative_pos.is_none() {
            let q = if let Some(query_states) = query_states {
                query_states.dim(D::Minus2)?
            } else {
                hidden_states.dim(D::Minus2)?
            };

            return Ok(Some(build_relative_position(
                q,
                hidden_states.dim(D::Minus2)?,
                &self.device,
                Some(self.position_buckets),
                Some(self.max_relative_positions),
            )?));
        }

        if relative_pos.is_some() {
            Ok(relative_pos.cloned())
        } else {
            Ok(None)
        }
    }
    fn get_rel_embedding(&self) -> Result<Option<Tensor>> {
        if !self.relative_attention {
            return Ok(None);
        }

        let rel_embeddings = self
            .rel_embeddings
            .as_ref()
            .context("self.rel_embeddings not present when using relative_attention")?
            .embeddings()
            .clone();

        if !self.norm_rel_ebd.contains("layer_norm") {
            return Ok(Some(rel_embeddings));
        }

        let layer_normed_embeddings = self
            .layer_norm
            .as_ref()
            .context("DebertaV2Encoder layer_norm is None when norm_rel_ebd contains layer_norm")?
            .forward(&rel_embeddings, None)?;

        Ok(Some(layer_normed_embeddings))
    }
}

pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
    classifier: Option<DebertaV2SeqClassificationHead>,
    pool: Option<Pool>,
    pub device: Device,
    pub dtype: DType,
    span: tracing::Span,
}

impl DebertaV2Model {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config, model_type: ModelType) -> Result<Self> {
        let (classifier, pool) = match model_type {
            ModelType::Classifier => {
                let classifier = DebertaV2SeqClassificationHead::load(vb.clone(), config)?;
                (Some(classifier), None)
            }
            ModelType::Embedding(pool) => (None, Some(pool)),
        };

        let embeddings = match DebertaV2Embeddings::load(vb.pp("embeddings"), config) {
            Ok(embeddings) => embeddings,
            Err(_) => DebertaV2Embeddings::load(vb.pp("deberta.embeddings"), config)?,
        };

        let encoder = match DebertaV2Encoder::load(vb.pp("encoder"), config) {
            Ok(encoder) => encoder,
            Err(_) => DebertaV2Encoder::load(vb.pp("deberta.encoder"), config)?,
        };

        Ok(Self {
            embeddings,
            encoder,
            classifier,
            pool,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let shape = (batch_size, max_length);

        let (input_ids, token_type_ids, position_ids, input_lengths, attention_mask) =
            if batch_size > 1 {
                // Prepare padded batch
                let elems = batch_size * max_length;

                let mut input_ids = Vec::with_capacity(elems);
                let mut token_type_ids = Vec::with_capacity(elems);
                let mut position_ids = Vec::with_capacity(elems);
                let mut attention_mask = Vec::with_capacity(elems);
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
                        token_type_ids.push(batch.token_type_ids[j]);
                        position_ids.push(batch.position_ids[j]);
                        attention_mask.push(1.0_f32);
                    }

                    // Add padding if needed
                    let padding = batch.max_length - seq_length;
                    if padding > 0 {
                        // Set bool to use attention mask
                        masking = true;
                        for _ in 0..padding {
                            input_ids.push(0);
                            token_type_ids.push(0);
                            position_ids.push(0);
                            attention_mask.push(0.0_f32);
                        }
                    }
                }

                let attention_mask = match masking {
                    true => {
                        let attention_mask = Tensor::from_vec(
                            attention_mask,
                            (batch_size, max_length, 1),
                            &self.device,
                        )?
                        .to_dtype(self.dtype)?;

                        Some(attention_mask)
                    }
                    false => None,
                };

                (
                    input_ids,
                    token_type_ids,
                    position_ids,
                    input_lengths,
                    attention_mask,
                )
            } else {
                (
                    batch.input_ids,
                    batch.token_type_ids,
                    batch.position_ids,
                    vec![batch.max_length as f32],
                    None,
                )
            };

        let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
        let token_type_ids = Tensor::from_vec(token_type_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;
        let mut input_lengths =
            Tensor::from_vec(input_lengths, (batch_size, 1), &self.device)?.to_dtype(self.dtype)?;

        let embedding_output = self.embeddings.forward(
            &input_ids,
            &token_type_ids,
            &position_ids,
            attention_mask.as_ref(),
        )?;

        let encoder_attention_mask = attention_mask
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Tensor::ones(shape, self.dtype, &self.device).unwrap());
        let encoder_output =
            self.encoder
                .forward(&embedding_output, &encoder_attention_mask, None, None)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            let pooled_indices_length = batch.pooled_indices.len();
            let mut outputs = encoder_output.clone();

            // Only use pooled_indices if at least one member of the batch ask for raw embeddings
            let pooled_indices = if has_raw_requests {
                let pooled_indices =
                    Tensor::from_vec(batch.pooled_indices, pooled_indices_length, &self.device)?;

                // Select values in the batch
                outputs = outputs.index_select(&pooled_indices, 0)?;
                Some(pooled_indices)
            } else {
                None
            };

            let pooled_embeddings = match self.pool {
                // CLS pooling
                Some(Pool::Cls) => outputs.i((.., 0))?,
                // Last token pooling is not supported for this model
                Some(Pool::LastToken) => unreachable!(),
                // Mean pooling
                Some(Pool::Mean) => {
                    if let Some(ref attention_mask) = attention_mask {
                        let mut attention_mask = attention_mask.clone();

                        if let Some(pooled_indices) = pooled_indices {
                            // Select values in the batch
                            attention_mask = attention_mask.index_select(&pooled_indices, 0)?;
                            input_lengths = input_lengths.index_select(&pooled_indices, 0)?;
                        };

                        // Mask padded values
                        outputs = outputs.broadcast_mul(&attention_mask)?;
                    }

                    (outputs.sum(1)?.broadcast_div(&input_lengths))?
                }
                Some(Pool::Splade) => unreachable!(),
                None => outputs,
            };
            Some(pooled_embeddings)
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            // Reshape outputs
            let (b, l, h) = encoder_output.shape().dims3()?;
            let outputs = encoder_output.reshape((b * l, h))?;

            // We need to remove the padding tokens only if batch_size > 1 and there are some
            // member of the batch that require pooling
            // or if batch_size > 1 and the members of the batch have different lengths
            if (attention_mask.is_some() || has_pooling_requests) && batch_size > 1 {
                let mut final_indices: Vec<u32> = Vec::with_capacity(batch_size * max_length);

                for i in batch.raw_indices.into_iter() {
                    let start = i * batch.max_length;
                    let i = i as usize;
                    let length =
                        batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i];

                    for j in start..start + length {
                        // Add indices for the tokens of this specific member of the batch
                        final_indices.push(j);
                    }
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

                // Select the tokens with final indices
                Some(outputs.index_select(&final_indices, 0)?)
            } else {
                Some(outputs)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

pub struct DebertaV2SeqClassificationHead {
    pooler: DebertaV2ContextPooler,
    classifier: Linear,
    span: tracing::Span,
}

impl DebertaV2SeqClassificationHead {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let id2label_len = match &config.id2label {
            None => candle::bail!("`id2label` must be set for classifier models"),
            Some(id2label) => id2label.len(),
        };
        let pooler = DebertaV2ContextPooler::load(vb.clone(), config)?;
        let output_dim = pooler.output_dim();

        // Try loading classifier from "classifier" first, then "deberta.classifier"
        let classifier_vb = vb.root().pp("classifier");
        let classifier = match classifier_vb.get((id2label_len, output_dim), "weight") {
            Ok(weight) => Linear::new(weight, Some(classifier_vb.get(id2label_len, "bias")?), None),
            Err(_) => {
                let classifier_vb = vb.root().pp("deberta.classifier");
                Linear::new(
                    classifier_vb.get((id2label_len, output_dim), "weight")?,
                    Some(classifier_vb.get(id2label_len, "bias")?),
                    None,
                )
            }
        };

        Ok(Self {
            pooler,
            classifier,
            span: tracing::span!(tracing::Level::TRACE, "classifier"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let pooled_output = self.pooler.forward(hidden_states)?;
        self.classifier.forward(&pooled_output)
    }
}

pub struct DebertaV2ContextPooler {
    dense: Linear,
    pooler_hidden_size: usize,
    span: tracing::Span,
}

impl DebertaV2ContextPooler {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let pooler_hidden_size = config
            .pooler_hidden_size
            .context("config.pooler_hidden_size is required for DebertaV2ContextPooler")?;

        let pooler_hidden_act = config
            .pooler_hidden_act
            .clone()
            .context("config.pooler_hidden_act is required for DebertaV2ContextPooler")?;

        let dense = Linear::new(
            vb.root()
                .pp("pooler.dense")
                .get((pooler_hidden_size, pooler_hidden_size), "weight")?,
            Some(
                vb.root()
                    .pp("pooler.dense")
                    .get(pooler_hidden_size, "bias")?,
            ),
            Some(pooler_hidden_act),
        );

        Ok(Self {
            dense,
            pooler_hidden_size,
            span: tracing::span!(tracing::Level::TRACE, "pooler"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let context_token = hidden_states.narrow(1, 0, 1)?.squeeze(1)?;

        self.dense.forward(&context_token.contiguous()?)
    }

    pub fn output_dim(&self) -> usize {
        self.pooler_hidden_size
    }
}

pub(crate) fn build_relative_position(
    query_size: usize,
    key_size: usize,
    device: &Device,
    bucket_size: Option<isize>,
    max_position: Option<isize>,
) -> Result<Tensor> {
    let q_ids = Tensor::arange(0, query_size as i64, device)?.unsqueeze(0)?;
    let k_ids: Tensor = Tensor::arange(0, key_size as i64, device)?.unsqueeze(D::Minus1)?;
    let mut rel_pos_ids = k_ids.broadcast_sub(&q_ids)?;
    let bucket_size = bucket_size.unwrap_or(-1);
    let max_position = max_position.unwrap_or(-1);

    if bucket_size > 0 && max_position > 0 {
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position, device)?;
    }

    rel_pos_ids = rel_pos_ids.to_dtype(DType::I64)?;
    rel_pos_ids = rel_pos_ids.narrow(0, 0, query_size)?;
    rel_pos_ids.unsqueeze(0)
}

pub(crate) fn make_log_bucket_position(
    relative_pos: Tensor,
    bucket_size: isize,
    max_position: isize,
    device: &Device,
) -> Result<Tensor> {
    let sign = relative_pos.to_dtype(DType::F32)?.sign()?;

    let mid = bucket_size / 2;

    let lt_mid = relative_pos.lt(mid as i64)?;
    let gt_neg_mid = relative_pos.gt(-mid as i64)?;

    let condition = lt_mid
        .to_dtype(candle::DType::F32)?
        .mul(&gt_neg_mid.to_dtype(candle::DType::F32)?)?
        .to_dtype(DType::U8)?;

    let on_true = Tensor::new(&[(mid - 1) as u32], device)?
        .broadcast_as(relative_pos.shape())?
        .to_dtype(relative_pos.dtype())?;

    let on_false = relative_pos
        .to_dtype(DType::F32)?
        .abs()?
        .to_dtype(DType::I64)?;

    let abs_pos = condition.where_cond(&on_true, &on_false)?;

    let mid_as_tensor = Tensor::from_slice(&[mid as f32], (1,), device)?;

    let log_pos = {
        let first_log = abs_pos
            .to_dtype(DType::F32)?
            .broadcast_div(&mid_as_tensor)?
            .log()?;

        let second_log =
            Tensor::from_slice(&[((max_position as f32 - 1.0) / mid as f32)], (1,), device)?
                .log()?;

        let first_div_second = first_log.broadcast_div(&second_log)?;

        let to_ceil = first_div_second
            .broadcast_mul(Tensor::from_slice(&[(mid - 1) as f32], (1,), device)?.as_ref())?;

        let ceil = to_ceil.ceil()?;

        ceil.broadcast_add(&mid_as_tensor)?
    };

    Ok({
        let abs_pos_lte_mid = abs_pos.to_dtype(DType::F32)?.broadcast_le(&mid_as_tensor)?;
        let relative_pos = relative_pos.to_dtype(relative_pos.dtype())?;
        let log_pos_mul_sign = log_pos.broadcast_mul(&sign.to_dtype(DType::F32)?)?;
        abs_pos_lte_mid.where_cond(&relative_pos.to_dtype(DType::F32)?, &log_pos_mul_sign)?
    })
}

impl Model for DebertaV2Model {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        match &self.classifier {
            None => candle::bail!("`predict` is not implemented for this model"),
            Some(classifier) => {
                let (pooled_embeddings, _raw_embeddings) = self.forward(batch)?;
                let pooled_embeddings =
                    pooled_embeddings.expect("pooled_embeddings is empty. This is a bug.");
                classifier.forward(&pooled_embeddings)
            }
        }
    }
}
