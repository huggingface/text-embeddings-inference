use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::{Div, Mul};
use std::path::Path;
use std::sync::Mutex;

use ndarray::{s, Axis};
use nohash_hasher::BuildNoHashHasher;
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use serde::Deserialize;

use text_embeddings_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Pool, Predictions,
};

#[cfg(test)]
mod tests;

const TOKEN_EMBEDDING_OUTPUTS: [&str; 2] = ["last_hidden_state", "token_embeddings"];

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub pad_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,

    // NOTE: `bidirectional_pplx_qwen3` produces `pooler_output_int8` embeddings besides
    // `last_hidden_state` (+ pooling), which should be used until there's native support for a
    // quantization or precision parameter
    pub model_type: Option<String>,

    // NOTE: The fields below are only required when the ONNX model expects the `past_key_values`
    // as input i.e., whenever the ONNX model has been exported with optimized MHA/MQA nodes
    // NOTE: The aliases from `n_embd`, `n_layer`, and `n_head` have been included for some edge
    // cases as e.g. `nomic-ai/nomic-embed-text-v1`, given that those ONNX exports use MQA
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,
    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_head")]
    pub num_key_value_heads: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum PaddingSide {
    Left,
    #[default]
    Right,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    #[serde(default)]
    padding_side: PaddingSide,
}

struct ModelInputs {
    pub input_ids: ndarray::Array2<i64>,
    pub attention_mask: ndarray::Array2<i64>,
    pub token_type_ids: Option<ndarray::Array2<i64>>,
    pub position_ids: Option<ndarray::Array2<i64>>,
    pub input_lengths: ndarray::Array1<f32>,
    pub past_key_values: Option<Vec<(ndarray::Array4<f32>, ndarray::Array4<f32>)>>,
}

pub struct OrtBackend {
    session: Mutex<Session>,
    config: Config,
    tokenizer_config: TokenizerConfig,

    token_type_ids: bool,
    // NOTE: required since the key can either be `token_type_ids` or `input_type`
    token_type_ids_key: String,
    position_ids: bool,
    past_key_values: bool,

    pool: Pool,
}

impl OrtBackend {
    pub fn new(
        model_path: &Path,
        dtype: String,
        model_type: ModelType,
    ) -> Result<Self, BackendError> {
        if dtype != "float32" {
            return Err(BackendError::Start(format!(
                "Dtype {dtype} is not supported for `ort`, only float32."
            )));
        };

        let pool = match model_type {
            ModelType::Classifier => Pool::Cls,
            ModelType::Embedding(pool) => match pool {
                Pool::Splade => {
                    return Err(BackendError::Start(format!(
                        "Pooling {pool} is not supported for `ort`, use `candle` instead."
                    )));
                }
                _ => pool,
            },
        };

        let onnx_path = {
            let default_path = model_path.join("model.onnx");
            match default_path.exists() {
                true => default_path,
                false => model_path.join("onnx/model.onnx"),
            }
        };

        let config: Config = {
            let content = std::fs::read_to_string(model_path.join("config.json"))
                .map_err(|e| BackendError::Start(format!("Failed to read `config.json`: {}", e)))?;
            serde_json::from_str(&content)
                .map_err(|e| BackendError::Start(format!("Failed to parse `config.json`: {}", e)))?
        };

        let tokenizer_config_path = model_path.join("tokenizer_config.json");
        let tokenizer_config: TokenizerConfig = if tokenizer_config_path.exists() {
            let content = std::fs::read_to_string(&tokenizer_config_path).map_err(|e| {
                BackendError::Start(format!("Failed to read `tokenizer_config.json`: {}", e))
            })?;
            serde_json::from_str(&content).map_err(|e| {
                BackendError::Start(format!("Failed to parse `tokenizer_config.json`: {}", e))
            })?
        } else {
            TokenizerConfig {
                padding_side: PaddingSide::default(),
            }
        };

        let session = Session::builder()
            .s()?
            .with_intra_threads(num_cpus::get())
            .s()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .s()?
            .commit_from_file(onnx_path)
            .s()?;

        let mut token_type_ids = false;
        let mut token_type_ids_key = String::from("token_type_ids");
        let mut position_ids = false;
        let mut past_key_values = false;

        for input in &session.inputs {
            match input.name.as_str() {
                "token_type_ids" | "input_type" => {
                    token_type_ids = true;
                    token_type_ids_key = String::from("token_type_ids");
                }
                "position_ids" => {
                    position_ids = true;
                }
                name if name.starts_with("past_key_values.") => {
                    past_key_values = true;
                }
                // NOTE: no need to handle `inputs_ids` and `attention_mask` since those are always
                // required
                _ => {}
            }
        }

        Ok(Self {
            session: Mutex::new(session),
            config,
            tokenizer_config,
            token_type_ids,
            token_type_ids_key,
            position_ids,
            past_key_values,
            pool,
        })
    }
}

impl OrtBackend {
    fn prepare_inputs(
        &self,
        batch: &Batch,
        padding_side: &PaddingSide,
    ) -> Result<(ModelInputs, bool), BackendError> {
        let batch_size = batch.len();
        let max_length = batch.max_length as usize;
        let elems = batch_size * max_length;

        let pad_token_id = self
            .config
            .pad_token_id
            .unwrap_or(self.config.eos_token_id.unwrap_or(0)) as i64;

        let (input_ids, attention_mask, token_type_ids, position_ids, input_lengths, masking) =
            if batch_size > 1 {
                let mut input_ids = Vec::with_capacity(elems);
                let mut attention_mask = Vec::with_capacity(elems);
                let mut token_type_ids = Vec::with_capacity(elems);
                let mut position_ids = Vec::with_capacity(elems);
                let mut input_lengths = Vec::with_capacity(batch_size);

                // Whether at least one of the request in the batch is padded
                let mut masking = false;

                match padding_side {
                    PaddingSide::Right => {
                        for i in 0..batch_size {
                            let start = batch.cumulative_seq_lengths[i] as usize;
                            let end = batch.cumulative_seq_lengths[i + 1] as usize;
                            let seq_length = (end - start) as u32;
                            input_lengths.push(seq_length as f32);

                            for (pos, j) in (start..end).enumerate() {
                                input_ids.push(batch.input_ids[j] as i64);
                                attention_mask.push(1_i64);
                                token_type_ids.push(batch.token_type_ids[j] as i64);
                                position_ids.push(pos as i64);
                            }

                            let padding = batch.max_length - seq_length;
                            if padding > 0 {
                                // NOTE: Set `masking=true` to use attention mask as not all the
                                // sequences in the batch have the same length
                                masking = true;
                                for pad_pos in 0..padding {
                                    input_ids.push(pad_token_id);
                                    attention_mask.push(0_i64);
                                    token_type_ids.push(0);
                                    position_ids.push((seq_length + pad_pos) as i64);
                                }
                            }
                        }

                        (
                            input_ids,
                            attention_mask,
                            token_type_ids,
                            position_ids,
                            input_lengths,
                            masking,
                        )
                    }
                    PaddingSide::Left => {
                        for i in 0..batch_size {
                            let start = batch.cumulative_seq_lengths[i] as usize;
                            let end = batch.cumulative_seq_lengths[i + 1] as usize;
                            let seq_length = (end - start) as u32;
                            input_lengths.push(seq_length as f32);

                            let padding = batch.max_length - seq_length;
                            if padding > 0 {
                                // NOTE: Set `masking=true` to use attention mask as not all the
                                // sequences in the batch have the same length
                                masking = true;
                                for _ in 0..padding {
                                    input_ids.push(pad_token_id);
                                    attention_mask.push(0_i64);
                                    token_type_ids.push(0);
                                    position_ids.push(0);
                                }
                            }

                            for (pos, j) in (start..end).enumerate() {
                                input_ids.push(batch.input_ids[j] as i64);
                                attention_mask.push(1_i64);
                                token_type_ids.push(batch.token_type_ids[j] as i64);
                                position_ids.push((padding + pos as u32) as i64);
                            }
                        }

                        (
                            input_ids,
                            attention_mask,
                            token_type_ids,
                            position_ids,
                            input_lengths,
                            masking,
                        )
                    }
                }
            } else {
                let attention_mask = vec![1_i64; elems];
                let position_ids: Vec<i64> = (0..max_length as i64).collect();

                (
                    batch.input_ids.iter().map(|v| *v as i64).collect(),
                    attention_mask,
                    batch.token_type_ids.iter().map(|v| *v as i64).collect(),
                    position_ids,
                    vec![batch.max_length as f32],
                    // NOTE: no need to mask the inputs when the batch only contains one element
                    false,
                )
            };

        let input_ids = ndarray::Array2::from_shape_vec((batch_size, max_length), input_ids).e()?;
        let attention_mask =
            ndarray::Array2::from_shape_vec((batch_size, max_length), attention_mask).e()?;

        let token_type_ids = if self.token_type_ids {
            Some(ndarray::Array2::from_shape_vec((batch_size, max_length), token_type_ids).e()?)
        } else {
            None
        };

        let position_ids = if self.position_ids {
            Some(ndarray::Array2::from_shape_vec((batch_size, max_length), position_ids).e()?)
        } else {
            None
        };

        let input_lengths = ndarray::Array1::from_vec(input_lengths);

        let past_key_values = if self.past_key_values {
            let hidden_size = self.config.hidden_size;
            let num_hidden_layers = self.config.num_hidden_layers;
            let num_key_value_heads = self
                .config
                .num_key_value_heads
                .unwrap_or(self.config.num_hidden_layers);
            let head_size = hidden_size / num_key_value_heads;
            let mut arrays = Vec::new();

            for _ in 0..num_hidden_layers {
                let shape = (batch_size, num_key_value_heads, 0, head_size);
                let key_array = ndarray::Array4::<f32>::zeros(shape);
                let value_array = ndarray::Array4::<f32>::zeros(shape);
                arrays.push((key_array, value_array));
            }

            Some(arrays)
        } else {
            None
        };

        Ok((
            ModelInputs {
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                input_lengths,
                past_key_values,
            },
            masking,
        ))
    }

    fn prepare_ort_inputs(
        &self,
        input_ids: ndarray::Array2<i64>,
        attention_mask: ndarray::Array2<i64>,
        token_type_ids: Option<ndarray::Array2<i64>>,
        position_ids: Option<ndarray::Array2<i64>>,
        past_key_values: Option<Vec<(ndarray::Array4<f32>, ndarray::Array4<f32>)>>,
    ) -> Result<Vec<(Cow<'_, str>, SessionInputValue<'_>)>, BackendError> {
        let mut inputs = ort::inputs![
            "input_ids" => ort::value::Tensor::from_array(input_ids).e()?,
            "attention_mask" => ort::value::Tensor::from_array(attention_mask).e()?,
        ];

        if let Some(token_type_ids) = token_type_ids {
            let token_type_ids = ort::value::Tensor::from_array(token_type_ids).e()?;
            inputs.push((
                self.token_type_ids_key.clone().into(),
                token_type_ids.into(),
            ));
        }

        if let Some(position_ids) = position_ids {
            let position_ids = ort::value::Tensor::from_array(position_ids).e()?;
            inputs.push(("position_ids".into(), position_ids.into()));
        }

        if let Some(past_key_values) = past_key_values {
            for (layer_idx, (key, value)) in past_key_values.into_iter().enumerate() {
                let key = ort::value::Tensor::from_array(key).e()?;
                let value = ort::value::Tensor::from_array(value).e()?;

                inputs.push((
                    format!("past_key_values.{}.key", layer_idx).into(),
                    key.into(),
                ));
                inputs.push((
                    format!("past_key_values.{}.value", layer_idx).into(),
                    value.into(),
                ));
            }
        }

        Ok(inputs)
    }
}

impl Backend for OrtBackend {
    fn max_batch_size(&self) -> Option<usize> {
        Some(8)
    }

    fn health(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn is_padded(&self) -> bool {
        true
    }

    fn supports_token_embeddings(&self) -> bool {
        self.session
            .lock()
            .unwrap()
            .outputs
            .iter()
            .any(|output| TOKEN_EMBEDDING_OUTPUTS.contains(&output.name.as_str()))
    }

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let (model_inputs, masking) =
            self.prepare_inputs(&batch, &self.tokenizer_config.padding_side)?;

        let inputs = self.prepare_ort_inputs(
            model_inputs.input_ids,
            model_inputs.attention_mask.clone(),
            model_inputs.token_type_ids,
            model_inputs.position_ids,
            model_inputs.past_key_values,
        )?;

        // Run model
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs).e()?;

        let mut embeddings =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();
        let token_embeddings = || {
            TOKEN_EMBEDDING_OUTPUTS
                .iter()
                .find_map(|name| outputs.get(name))
                .ok_or_else(|| {
                    BackendError::Inference(
                        "The ONNX model does not provide token embeddings (last_hidden_state or token_embeddings)"
                            .to_string(),
                    )
                })?
                .try_extract_array::<f32>()
                .e()
        };

        if has_raw_requests {
            let values = token_embeddings()?;
            if !matches!(values.shape(), [b, s, _] if *b == batch_size && *s >= max_length) {
                return Err(BackendError::Inference(format!(
                    "Unexpected ONNX token embedding shape: {:?}",
                    values.shape()
                )));
            }
            for &index in &batch.raw_indices {
                let i = index as usize;
                let length = (batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i])
                    as usize;
                let start = match self.tokenizer_config.padding_side {
                    PaddingSide::Left => max_length - length,
                    PaddingSide::Right => 0,
                };
                let rows = values.slice(s![i, start..start + length, ..]);
                let tokens = rows.rows().into_iter().map(|row| row.to_vec()).collect();
                embeddings.insert(i, Embedding::All(tokens));
            }
        }

        let int8_model = self.config.model_type.as_deref() == Some("bidirectional_pplx_qwen3");
        let int8_pooled = outputs.get("pooler_output_int8").filter(|_| int8_model);
        let sentence_pooled = outputs.get("sentence_embedding");
        if int8_model && int8_pooled.is_none() && sentence_pooled.is_none() {
            tracing::warn!("`model_type` in `config.json` is set to `bidirectional_pplx_qwen3` but the output key `pooler_output_int8` is missing from the ONNX export, which might lead to degraded performance given that tanh should be applied to the pooled embeddings and then scaled to INT8.");
        }

        if has_pooling_requests {
            if let Some(output) = int8_pooled {
                // Preserve the quantized model output, converted to the response type.
                insert_pooled_embeddings(
                    output.try_extract_array::<i8>().e()?,
                    &batch,
                    &mut embeddings,
                )?;
            } else if let Some(output) = sentence_pooled {
                insert_pooled_embeddings(
                    output.try_extract_array::<f32>().e()?,
                    &batch,
                    &mut embeddings,
                )?;
            } else {
                let mut outputs = token_embeddings()?.to_owned();

                // Only use pooled_indices if at least one member of the batch ask for raw embeddings
                let indices = if has_raw_requests {
                    let indices: Vec<usize> =
                        batch.pooled_indices.iter().map(|v| *v as usize).collect();

                    // Select values in the batch
                    outputs = outputs.select(Axis(0), &indices);
                    Some(indices)
                } else {
                    None
                };

                let input_lengths = match &indices {
                    Some(indices) => model_inputs.input_lengths.select(Axis(0), indices),
                    None => model_inputs.input_lengths,
                };
                let pooled_embeddings = match self.pool {
                    Pool::Cls => match self.tokenizer_config.padding_side {
                        PaddingSide::Left => {
                            if masking {
                                let mut cls_embeddings = Vec::new();
                                for (batch_idx, &seq_length) in input_lengths.iter().enumerate() {
                                    let padding = max_length as f32 - seq_length;
                                    let cls_pos = padding as usize;
                                    cls_embeddings
                                        .push(outputs.slice(s![batch_idx, cls_pos, ..]).to_owned());
                                }
                                ndarray::stack(
                                    Axis(0),
                                    &cls_embeddings.iter().map(|x| x.view()).collect::<Vec<_>>(),
                                )
                                .unwrap()
                                .into_dyn()
                            } else {
                                outputs.slice(s![.., 0, ..]).into_owned().into_dyn()
                            }
                        }
                        PaddingSide::Right => outputs.slice(s![.., 0, ..]).into_owned().into_dyn(),
                    },
                    Pool::LastToken => match self.tokenizer_config.padding_side {
                        // NOTE: when using left-padding, the last-token is always in the last position
                        // as the padding tokens are on the left (note that given that the last token
                        // in the sequence is the EOS token we need to use the last - 1.
                        PaddingSide::Left => {
                            let axis_len = outputs.len_of(Axis(1));
                            outputs
                                .slice(s![.., axis_len - 1, ..])
                                .into_owned()
                                .into_dyn()
                        }
                        PaddingSide::Right => {
                            if masking {
                                let mut last_token_embeddings = Vec::new();
                                for (batch_idx, &seq_length) in input_lengths.iter().enumerate() {
                                    let last_pos = seq_length as usize - 1;
                                    last_token_embeddings.push(
                                        outputs.slice(s![batch_idx, last_pos, ..]).to_owned(),
                                    );
                                }
                                ndarray::stack(
                                    Axis(0),
                                    &last_token_embeddings
                                        .iter()
                                        .map(|x| x.view())
                                        .collect::<Vec<_>>(),
                                )
                                .unwrap()
                                .into_dyn()
                            } else {
                                let axis_len = outputs.len_of(Axis(1));
                                outputs
                                    .slice(s![.., axis_len - 1, ..])
                                    .into_owned()
                                    .into_dyn()
                            }
                        }
                    },
                    Pool::Mean => {
                        if masking {
                            let mut attention_mask = model_inputs.attention_mask;

                            if let Some(indices) = indices {
                                // Select values in the batch
                                attention_mask = attention_mask.select(Axis(0), &indices);
                            };

                            match self.tokenizer_config.padding_side {
                                PaddingSide::Left => {
                                    let mut mean_embeddings = Vec::new();
                                    for (batch_idx, &seq_length) in input_lengths.iter().enumerate()
                                    {
                                        let padding = max_length as f32 - seq_length;
                                        let start_pos = padding as usize;
                                        let valid_embeddings =
                                            outputs.slice(s![batch_idx, start_pos.., ..]);
                                        mean_embeddings
                                            .push(valid_embeddings.mean_axis(Axis(0)).unwrap());
                                    }
                                    ndarray::stack(
                                        Axis(0),
                                        &mean_embeddings
                                            .iter()
                                            .map(|x| x.view())
                                            .collect::<Vec<_>>(),
                                    )
                                    .unwrap()
                                    .into_dyn()
                                }
                                PaddingSide::Right => {
                                    let attention_mask =
                                        attention_mask.mapv(|x| x as f32).insert_axis(Axis(2));
                                    outputs = outputs.mul(attention_mask);
                                    outputs
                                        .sum_axis(Axis(1))
                                        .div(input_lengths.insert_axis(Axis(1)))
                                }
                            }
                        } else {
                            outputs.mean_axis(Axis(1)).unwrap()
                        }
                    }
                    Pool::Splade => unreachable!(),
                };

                for (i, e) in batch
                    .pooled_indices
                    .into_iter()
                    .zip(pooled_embeddings.rows())
                {
                    embeddings.insert(i as usize, Embedding::Pooled(e.to_vec()));
                }
            }
        }

        Ok(embeddings)
    }

    fn predict(&self, batch: Batch) -> Result<Predictions, BackendError> {
        let batch_size = batch.len();

        let (model_inputs, _) = self.prepare_inputs(&batch, &self.tokenizer_config.padding_side)?;

        let inputs = self.prepare_ort_inputs(
            model_inputs.input_ids,
            model_inputs.attention_mask.clone(),
            model_inputs.token_type_ids,
            model_inputs.position_ids,
            model_inputs.past_key_values,
        )?;

        // Run model
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs).e()?;

        // Get last_hidden_state ndarray
        let outputs = outputs["logits"].try_extract_array::<f32>().e()?.to_owned();

        let mut predictions =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());
        for (i, r) in outputs.rows().into_iter().enumerate() {
            predictions.insert(i, r.to_vec());
        }

        Ok(predictions)
    }
}

fn insert_pooled_embeddings<T: Copy + Into<f32>>(
    values: ndarray::ArrayViewD<'_, T>,
    batch: &Batch,
    embeddings: &mut Embeddings,
) -> Result<(), BackendError> {
    // The last axis holds each vector, including exports with singleton dimensions.
    let rows = values.rows().into_iter();
    if rows.len() != batch.len() {
        return Err(BackendError::Inference(format!(
            "Unexpected ONNX sentence embedding shape: {:?}",
            values.shape()
        )));
    }
    for (i, values) in rows.enumerate() {
        if batch.pooled_indices.contains(&(i as u32)) {
            let row = values.iter().copied().map(Into::into).collect();
            embeddings.insert(i, Embedding::Pooled(row));
        }
    }
    Ok(())
}

pub trait WrapErr<O> {
    fn s(self) -> Result<O, BackendError>;
    fn e(self) -> Result<O, BackendError>;
}

impl<O> WrapErr<O> for Result<O, ort::Error> {
    fn s(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Start(e.to_string()))
    }
    fn e(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Inference(e.to_string()))
    }
}

impl<O> WrapErr<O> for Result<O, ndarray::ShapeError> {
    fn s(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Start(e.to_string()))
    }
    fn e(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Inference(e.to_string()))
    }
}
