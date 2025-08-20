use ndarray::{s, Axis};
use nohash_hasher::BuildNoHashHasher;
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::Deserialize;
use std::collections::HashMap;
use std::ops::{Div, Mul};
use std::path::Path;
use std::sync::Mutex;
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Pool, Predictions,
};

#[derive(Debug, Clone, Deserialize)]
pub struct PastKeyValuesConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
}

pub struct OrtBackend {
    session: Mutex<Session>,

    token_type_ids: bool,
    // NOTE: required since the key can either be `token_type_ids` or `input_type`
    token_type_ids_key: String,
    position_ids: bool,
    past_key_values: bool,
    past_key_values_config: Option<PastKeyValuesConfig>,

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

        let past_key_values_config = match past_key_values {
            true => {
                let path = model_path.join("config.json");
                if !path.exists() {
                    return Err(BackendError::Start(format!(
                        "config.json not found at {:?}",
                        path
                    )));
                }
                let content = std::fs::read_to_string(path).map_err(|e| {
                    BackendError::Start(format!("Failed to read config.json: {}", e))
                })?;
                Some(
                    serde_json::from_str::<PastKeyValuesConfig>(&content).map_err(|e| {
                        BackendError::Start(format!("Failed to parse config.json: {}", e))
                    })?,
                )
            }
            false => None,
        };

        Ok(Self {
            session: Mutex::new(session),
            token_type_ids,
            token_type_ids_key,
            position_ids,
            past_key_values,
            past_key_values_config,
            pool,
        })
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

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        // Whether a least one of the request in the batch is padded
        let mut masking = false;

        let (input_ids, token_type_ids, input_lengths, attention_mask, position_ids) = {
            let elems = batch_size * max_length;

            if batch_size > 1 {
                // Prepare padded batch
                let mut input_ids = Vec::with_capacity(elems);
                let mut token_type_ids = Vec::with_capacity(elems);
                let mut attention_mask = Vec::with_capacity(elems);
                let mut position_ids = Vec::with_capacity(elems);
                let mut input_lengths = Vec::with_capacity(batch_size);

                for i in 0..batch_size {
                    let start = batch.cumulative_seq_lengths[i] as usize;
                    let end = batch.cumulative_seq_lengths[i + 1] as usize;
                    let seq_length = (end - start) as u32;
                    input_lengths.push(seq_length as f32);

                    // Copy values
                    for (pos, j) in (start..end).enumerate() {
                        input_ids.push(batch.input_ids[j] as i64);
                        token_type_ids.push(batch.token_type_ids[j] as i64);
                        attention_mask.push(1_i64);
                        position_ids.push(pos as i64);
                    }

                    // Add padding if needed
                    let padding = batch.max_length - seq_length;
                    if padding > 0 {
                        // Set bool to use attention mask
                        masking = true;
                        for pad_pos in 0..padding {
                            input_ids.push(0);
                            token_type_ids.push(0);
                            attention_mask.push(0_i64);
                            position_ids.push((seq_length + pad_pos) as i64);
                        }
                    }
                }
                (
                    input_ids,
                    token_type_ids,
                    input_lengths,
                    attention_mask,
                    position_ids,
                )
            } else {
                let attention_mask = vec![1_i64; elems];
                let position_ids: Vec<i64> = (0..max_length as i64).collect();

                (
                    batch.input_ids.into_iter().map(|v| v as i64).collect(),
                    batch.token_type_ids.into_iter().map(|v| v as i64).collect(),
                    vec![batch.max_length as f32],
                    attention_mask,
                    position_ids,
                )
            }
        };

        // Create ndarrays
        let input_ids = ndarray::Array2::from_shape_vec((batch_size, max_length), input_ids).e()?;
        let attention_mask =
            ndarray::Array2::from_shape_vec((batch_size, max_length), attention_mask).e()?;
        let position_ids =
            ndarray::Array2::from_shape_vec((batch_size, max_length), position_ids).e()?;
        let input_lengths = ndarray::Array1::from_vec(input_lengths);

        let inputs = {
            let mut inputs = ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids).e()?,
                "attention_mask" => ort::value::Tensor::from_array(attention_mask.clone()).e()?,
            ];

            if self.token_type_ids {
                let token_type_ids_tensor =
                    ndarray::Array2::from_shape_vec((batch_size, max_length), token_type_ids)
                        .e()?;
                let token_type_ids_value =
                    ort::value::Tensor::from_array(token_type_ids_tensor).e()?;
                inputs.push((
                    self.token_type_ids_key.clone().into(),
                    token_type_ids_value.into(),
                ));
            }

            if self.position_ids {
                let position_ids_value = ort::value::Tensor::from_array(position_ids).e()?;
                inputs.push(("position_ids".into(), position_ids_value.into()));
            }

            if self.past_key_values {
                let config = self.past_key_values_config.as_ref().unwrap();
                let head_size = config.hidden_size / config.num_key_value_heads;

                for i in 0..config.num_hidden_layers {
                    let key_shape = (batch_size, config.num_key_value_heads, 0, head_size);
                    let value_shape = (batch_size, config.num_key_value_heads, 0, head_size);

                    let empty_key = ndarray::Array4::<f32>::zeros(key_shape);
                    let empty_value = ndarray::Array4::<f32>::zeros(value_shape);

                    let key_value = ort::value::Tensor::from_array(empty_key).e()?;
                    let value_value = ort::value::Tensor::from_array(empty_value).e()?;
                    inputs.push((
                        format!("past_key_values.{}.key", i).into(),
                        key_value.into(),
                    ));
                    inputs.push((
                        format!("past_key_values.{}.value", i).into(),
                        value_value.into(),
                    ));
                }
            }

            inputs
        };

        // Run model
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs).e()?;

        // Get last_hidden_state ndarray
        let outputs = outputs
            .get("last_hidden_state")
            .or(outputs.get("token_embeddings"))
            .ok_or(BackendError::Inference(format!(
                "Unknown output keys: {:?}",
                outputs
            )))?
            .try_extract_array::<f32>()
            .e()?
            .to_owned();

        // Final embeddings struct
        let mut embeddings =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        if has_pooling_requests {
            let mut outputs = outputs.clone();

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

            let pooled_embeddings = match self.pool {
                Pool::Cls => outputs.slice(s![.., 0, ..]).into_owned().into_dyn(),
                Pool::LastToken => {
                    let axis_len = outputs.len_of(Axis(1));
                    outputs
                        .slice(s![.., axis_len - 1, ..])
                        .into_owned()
                        .into_dyn()
                }
                Pool::Mean => {
                    if masking {
                        let mut attention_mask = attention_mask;
                        let mut input_lengths = input_lengths;

                        if let Some(indices) = indices {
                            // Select values in the batch
                            attention_mask = attention_mask.select(Axis(0), &indices);
                            input_lengths = input_lengths.select(Axis(0), &indices);
                        };

                        // Cast and reshape
                        let attention_mask = attention_mask.mapv(|x| x as f32).insert_axis(Axis(2));

                        // Mask padded values
                        outputs = outputs.mul(attention_mask);
                        outputs
                            .sum_axis(Axis(1))
                            .div(input_lengths.insert_axis(Axis(1)))
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
        };

        if has_raw_requests {
            // Reshape outputs
            let s = outputs.shape().to_vec();
            #[allow(deprecated)]
            let outputs = outputs.into_shape((s[0] * s[1], s[2])).e()?;

            // We need to remove the padding tokens only if batch_size > 1 and there are some
            // member of the batch that require pooling
            // or if batch_size > 1 and the members of the batch have different lengths
            let raw_embeddings = if (masking || has_pooling_requests) && batch_size > 1 {
                let mut final_indices: Vec<usize> = Vec::with_capacity(batch_size * max_length);

                for i in batch.raw_indices.iter() {
                    let start = i * batch.max_length;
                    let i = *i as usize;
                    let length =
                        batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i];

                    for j in start..start + length {
                        // Add indices for the tokens of this specific member of the batch
                        final_indices.push(j as usize);
                    }
                }

                // Select the tokens with final indices
                outputs.select(Axis(0), &final_indices)
            } else {
                outputs
            };

            // Used for indexing in the raw_embeddings tensor
            let input_lengths: Vec<usize> = (0..batch_size)
                .map(|i| {
                    (batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i]) as usize
                })
                .collect();

            let mut cumulative_length = 0;
            for i in batch.raw_indices.into_iter() {
                let length = input_lengths[i as usize];
                let e = raw_embeddings.slice(s![cumulative_length..cumulative_length + length, ..]);
                let e = e.rows().into_iter().map(|v| v.to_vec()).collect();

                embeddings.insert(i as usize, Embedding::All(e));
                cumulative_length += length;
            }
        }

        Ok(embeddings)
    }

    fn predict(&self, batch: Batch) -> Result<Predictions, BackendError> {
        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let (input_ids, token_type_ids, attention_mask, position_ids) = {
            let elems = batch_size * max_length;

            if batch_size > 1 {
                // Prepare padded batch
                let mut input_ids = Vec::with_capacity(elems);
                let mut token_type_ids = Vec::with_capacity(elems);
                let mut attention_mask = Vec::with_capacity(elems);
                let mut position_ids = Vec::with_capacity(elems);

                for i in 0..batch_size {
                    let start = batch.cumulative_seq_lengths[i] as usize;
                    let end = batch.cumulative_seq_lengths[i + 1] as usize;
                    let seq_length = (end - start) as u32;

                    // Copy values
                    for (pos, j) in (start..end).enumerate() {
                        input_ids.push(batch.input_ids[j] as i64);
                        token_type_ids.push(batch.token_type_ids[j] as i64);
                        attention_mask.push(1_i64);
                        position_ids.push(pos as i64);
                    }

                    // Add padding if needed
                    let padding = batch.max_length - seq_length;
                    if padding > 0 {
                        for pad_pos in 0..padding {
                            input_ids.push(0);
                            token_type_ids.push(0);
                            attention_mask.push(0_i64);
                            position_ids.push((seq_length + pad_pos) as i64);
                        }
                    }
                }
                (input_ids, token_type_ids, attention_mask, position_ids)
            } else {
                let attention_mask = vec![1_i64; elems];
                let position_ids: Vec<i64> = (0..max_length as i64).collect();

                (
                    batch.input_ids.into_iter().map(|v| v as i64).collect(),
                    batch.token_type_ids.into_iter().map(|v| v as i64).collect(),
                    attention_mask,
                    position_ids,
                )
            }
        };

        // Create ndarrays
        let input_ids = ndarray::Array2::from_shape_vec((batch_size, max_length), input_ids).e()?;
        let attention_mask =
            ndarray::Array2::from_shape_vec((batch_size, max_length), attention_mask).e()?;
        let position_ids =
            ndarray::Array2::from_shape_vec((batch_size, max_length), position_ids).e()?;

        let inputs = {
            let mut inputs = ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids).e()?,
                "attention_mask" => ort::value::Tensor::from_array(attention_mask.clone()).e()?,
            ];

            if self.token_type_ids {
                let token_type_ids_tensor =
                    ndarray::Array2::from_shape_vec((batch_size, max_length), token_type_ids)
                        .e()?;
                let token_type_ids_value =
                    ort::value::Tensor::from_array(token_type_ids_tensor).e()?;
                inputs.push((
                    self.token_type_ids_key.clone().into(),
                    token_type_ids_value.into(),
                ));
            }

            if self.position_ids {
                let position_ids_value = ort::value::Tensor::from_array(position_ids).e()?;
                inputs.push(("position_ids".into(), position_ids_value.into()));
            }

            if self.past_key_values {
                let config = self.past_key_values_config.as_ref().unwrap();
                let head_size = config.hidden_size / config.num_key_value_heads;

                for i in 0..config.num_hidden_layers {
                    let key_shape = (batch_size, config.num_key_value_heads, 0, head_size);
                    let value_shape = (batch_size, config.num_key_value_heads, 0, head_size);

                    let empty_key = ndarray::Array4::<f32>::zeros(key_shape);
                    let empty_value = ndarray::Array4::<f32>::zeros(value_shape);

                    let key_value = ort::value::Tensor::from_array(empty_key).e()?;
                    let value_value = ort::value::Tensor::from_array(empty_value).e()?;
                    inputs.push((
                        format!("past_key_values.{}.key", i).into(),
                        key_value.into(),
                    ));
                    inputs.push((
                        format!("past_key_values.{}.value", i).into(),
                        value_value.into(),
                    ));
                }
            }

            inputs
        };

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
