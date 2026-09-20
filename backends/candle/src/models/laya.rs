use crate::layers::{HiddenAct, LayerNorm, Linear};
use crate::models::{Model, ModernBertConfig, ModernBertModel};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use text_embeddings_backend_core::{DecisionInput, DecisionResult, ModelType};

#[derive(Debug, Default, Deserialize)]
pub struct LayaConfig {
    pub temperature: Option<Vec<f32>>,
    pub temperature_by_options: Option<HashMap<String, f32>>,
    pub head_layers: Option<usize>,
    pub n_act: Option<usize>,
    pub act_costs: Option<HashMap<String, f32>>,
}

impl LayaConfig {
    fn head_layers(&self) -> usize {
        self.head_layers.unwrap_or(2)
    }

    fn action_count(&self) -> usize {
        self.n_act
            .or_else(|| self.act_costs.as_ref().map(|costs| costs.len() + 1))
            .unwrap_or(2)
    }

    fn temperature(&self, qtype: u32, option_count: usize) -> f32 {
        let qtype_name = match qtype {
            0 => "choice",
            1 => "score",
            _ => "noul",
        };
        let bucket = if option_count <= 2 {
            "2"
        } else if option_count <= 5 {
            "3-5"
        } else if option_count <= 10 {
            "6-10"
        } else {
            "11+"
        };
        self.temperature_by_options
            .as_ref()
            .and_then(|temperatures| temperatures.get(&format!("{qtype_name}:{bucket}")))
            .copied()
            .or_else(|| self.temperature.as_ref().and_then(|values| values.get(qtype as usize).copied()))
            .unwrap_or(1.0)
            .max(f32::EPSILON)
    }
}

#[cfg(test)]
mod tests {
    use super::LayaConfig;
    use std::collections::HashMap;

    #[test]
    fn temperature_uses_question_bucket_before_type_default() {
        let config = LayaConfig {
            temperature: Some(vec![1.0, 1.1, 1.2]),
            temperature_by_options: Some(HashMap::from([
                ("choice:3-5".to_string(), 0.7),
                ("noul:2".to_string(), 0.9),
            ])),
            ..Default::default()
        };

        assert_eq!(config.temperature(0, 4), 0.7);
        assert_eq!(config.temperature(1, 4), 1.1);
        assert_eq!(config.temperature(2, 2), 0.9);
    }

    #[test]
    fn temperature_clamps_non_positive_values() {
        let config = LayaConfig {
            temperature: Some(vec![0.0]),
            temperature_by_options: None,
            ..Default::default()
        };

        assert!(config.temperature(0, 2) > 0.0);
    }
}

struct DecisionTransformerLayer {
    qkv: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    hidden_size: usize,
    num_heads: usize,
}

impl DecisionTransformerLayer {
    fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
        let self_attn = vb.pp("self_attn");
        Ok(Self {
            qkv: Linear::new(
                self_attn.get((hidden_size * 3, hidden_size), "in_proj_weight")?,
                Some(self_attn.get(hidden_size * 3, "in_proj_bias")?),
                None,
            ),
            out_proj: Linear::new(
                self_attn.pp("out_proj").get((hidden_size, hidden_size), "weight")?,
                Some(self_attn.pp("out_proj").get(hidden_size, "bias")?),
                None,
            ),
            linear1: Linear::new(
                vb.pp("linear1").get((hidden_size * 4, hidden_size), "weight")?,
                Some(vb.pp("linear1").get(hidden_size * 4, "bias")?),
                Some(HiddenAct::Gelu),
            ),
            linear2: Linear::new(
                vb.pp("linear2").get((hidden_size, hidden_size * 4), "weight")?,
                Some(vb.pp("linear2").get(hidden_size, "bias")?),
                None,
            ),
            norm1: LayerNorm::load(vb.pp("norm1"), hidden_size, epsilon)?,
            norm2: LayerNorm::load(vb.pp("norm2"), hidden_size, epsilon)?,
            hidden_size,
            num_heads: 16,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let normalized = self.norm1.forward(hidden_states, None)?;
        let qkv = self.qkv.forward(&normalized)?.chunk(3, candle::D::Minus1)?;
        let (batch_size, sequence_length, _) = hidden_states.dims3()?;
        let head_size = self.hidden_size / self.num_heads;
        let query = qkv[0]
            .reshape((batch_size, sequence_length, self.num_heads, head_size))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = qkv[1]
            .reshape((batch_size, sequence_length, self.num_heads, head_size))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = qkv[2]
            .reshape((batch_size, sequence_length, self.num_heads, head_size))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_transposed = key.transpose(2, 3)?.contiguous()?;
        let scores = (query.matmul(&key_transposed)?
            / (head_size as f64).sqrt())?
            .broadcast_add(attention_mask)?;
        let attention = candle_nn::ops::softmax_last_dim(&scores)?;
        let attention = attention
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, sequence_length, self.hidden_size))?;
        let attention = self.out_proj.forward(&attention)?;
        let hidden_states = hidden_states.add(&attention)?;
        let feed_forward = self
            .linear2
            .forward(&self.linear1.forward(&self.norm2.forward(&hidden_states, None)?)?)?;
        hidden_states.add(&feed_forward)
    }
}

pub struct LayaModel {
    encoder: ModernBertModel,
    head: Vec<DecisionTransformerLayer>,
    type_emb: Embedding,
    scorer_norm: LayerNorm,
    scorer_dense: Linear,
    scorer: Linear,
    act_dense: Linear,
    act: Linear,
    device: Device,
    dtype: DType,
    config: LayaConfig,
}

impl LayaModel {
    pub fn load(
        vb: VarBuilder,
        config: &ModernBertConfig,
        laya_config: LayaConfig,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let encoder = ModernBertModel::load(vb.clone(), config, ModelType::Decision)?;
        let head = (0..laya_config.head_layers())
            .map(|index| {
                DecisionTransformerLayer::load(
                    vb.pp(format!("head.layers.{index}")),
                    hidden_size,
                    config.norm_eps as f32,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let type_emb = Embedding::new(
            vb.pp("type_emb").get((3, hidden_size), "weight")?,
            hidden_size,
        );
        let scorer_norm = LayerNorm::load(vb.pp("scorer.0"), hidden_size, config.norm_eps as f32)?;
        let scorer_dense = Linear::new(
            vb.pp("scorer.1").get((hidden_size, hidden_size), "weight")?,
            Some(vb.pp("scorer.1").get(hidden_size, "bias")?),
            Some(HiddenAct::Gelu),
        );
        let scorer = Linear::new(
            vb.pp("scorer.3").get((1, hidden_size), "weight")?,
            Some(vb.pp("scorer.3").get(1, "bias")?),
            None,
        );
        let act_dense = Linear::new(
            vb.pp("act_head.0").get((256, hidden_size + 4), "weight")?,
            Some(vb.pp("act_head.0").get(256, "bias")?),
            Some(HiddenAct::Gelu),
        );
        let act = Linear::new(
            vb.pp("act_head.2").get((laya_config.action_count(), 256), "weight")?,
            Some(vb.pp("act_head.2").get(laya_config.action_count(), "bias")?),
            None,
        );
        Ok(Self {
            encoder,
            head,
            type_emb,
            scorer_norm,
            scorer_dense,
            scorer,
            act_dense,
            act,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            config: laya_config,
        })
    }

    fn forward(&self, inputs: Vec<DecisionInput>) -> Result<Vec<DecisionResult>> {
        let batch_size = inputs.len();
        let max_length = inputs.iter().map(|input| input.input_ids.len()).max().unwrap_or(0);
        if max_length == 0 {
            candle::bail!("decision inputs cannot be empty");
        }
        let mut input_ids = Vec::with_capacity(batch_size * max_length);
        let mut masks = Vec::with_capacity(batch_size * max_length);
        let mut positions = Vec::with_capacity(batch_size * max_length);
        for input in &inputs {
            let length = input.input_ids.len();
            input_ids.extend_from_slice(&input.input_ids);
            input_ids.extend(std::iter::repeat_n(self.encoder.pad_token_id, max_length - length));
            if let Some(mask) = &input.attention_mask {
                if mask.len() != length {
                    candle::bail!("decision attention mask length does not match input length");
                }
                masks.extend_from_slice(mask);
                masks.extend(std::iter::repeat_n(0, max_length - mask.len()));
            } else {
                masks.extend(std::iter::repeat_n(1, length));
                masks.extend(std::iter::repeat_n(0, max_length - length));
            }
            positions.extend(0..length as u32);
            positions.extend(std::iter::repeat_n(0, max_length - length));
        }
        let input_ids = Tensor::from_vec(input_ids, (batch_size, max_length), &self.device)?;
        let positions = Tensor::from_vec(positions, batch_size * max_length, &self.device)?;
        let mask = Tensor::from_vec(masks, (batch_size, max_length, 1), &self.device)?
            .to_dtype(self.dtype)?;
        let hidden = self.encoder.forward_hidden(&input_ids, &positions, Some(&mask))?;
        let type_ids = Tensor::from_vec(
            inputs.iter().map(|input| input.qtype.min(2)).collect(),
            batch_size,
            &self.device,
        )?;
        let mut hidden = hidden.broadcast_add(&self.type_emb.forward(&type_ids)?.unsqueeze(1)?)?;
        let head_mask =
            ((1.0 - mask.squeeze(2)?.unsqueeze(1)?.unsqueeze(1)?)? * -65504.0)?;
        for layer in &self.head {
            hidden = layer.forward(&hidden, &head_mask)?;
        }
        let scored = self
            .scorer
            .forward(&self.scorer_dense.forward(&self.scorer_norm.forward(&hidden, None)?)?)?
            .squeeze(2)?;
        let mut decisions = Vec::with_capacity(batch_size);
        for (index, input) in inputs.into_iter().enumerate() {
            if input.marker_positions.is_empty()
                || input
                    .marker_positions
                    .iter()
                    .any(|position| *position as usize >= input.input_ids.len())
            {
                candle::bail!("decision marker position is outside the input");
            }
            let markers = input.marker_positions;
            let marker_tensor = Tensor::from_vec(markers.clone(), markers.len(), &self.device)?;
            let temperature = self.config.temperature(input.qtype, markers.len());
            let logits = scored
                .i(index)?
                .index_select(&marker_tensor, 0)?
                .broadcast_div(&Tensor::new(&[temperature], &self.device)?.to_dtype(self.dtype)?)?;
            let probabilities = candle_nn::ops::softmax_last_dim(
                &logits.unsqueeze(0)?,
            )?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1()?;
            let max_probability = probabilities.iter().copied().fold(0.0, f32::max);
            let mut sorted = probabilities.clone();
            sorted.sort_by(|left, right| right.total_cmp(left));
            let top_two = sorted.get(1).copied().unwrap_or(0.0);
            let entropy = probabilities
                .iter()
                .filter(|probability| **probability > 0.0)
                .map(|probability| -probability * probability.ln())
                .sum::<f32>();
            let normalized_entropy = if probabilities.len() > 1 {
                entropy / (probabilities.len() as f32).ln()
            } else {
                0.0
            };
            let confidence = 1.0 - normalized_entropy;
            let features = Tensor::from_vec(
                vec![max_probability, max_probability - top_two, normalized_entropy, probabilities.len() as f32 / 255.0],
                (1, 4),
                &self.device,
            )?
            .to_dtype(self.dtype)?;
            let cls = hidden.i((index, 0usize))?.unsqueeze(0)?;
            let action_probabilities: Vec<f32> = candle_nn::ops::softmax_last_dim(
                &self.act.forward(&self.act_dense.forward(&Tensor::cat(&[&cls, &features], 1)?)?)?,
            )?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1()?;
            let action = action_probabilities
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(action, _)| action)
                .unwrap_or(0);
            let action_probability = action_probabilities.get(action).copied().unwrap_or(0.0);
            decisions.push(DecisionResult {
                question_index: input.question_index,
                probabilities,
                confidence,
                action,
                action_probability,
            });
        }
        Ok(decisions)
    }
}

impl Model for LayaModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn decide(&self, inputs: Vec<DecisionInput>) -> Result<Vec<DecisionResult>> {
        self.forward(inputs)
    }
}
