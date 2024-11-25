use ndarray::{s, Axis};
use nohash_hasher::BuildNoHashHasher;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::collections::HashMap;
use std::ops::{Div, Mul};
use std::path::Path;
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Pool, Predictions,
};

pub struct OrtBackend {
    session: Session,
    pool: Pool,
    type_id_name: Option<String>,
}

impl OrtBackend {
    pub fn new(
        model_path: &Path,
        dtype: String,
        model_type: ModelType,
    ) -> Result<Self, BackendError> {
        // Check dtype
        if dtype == "float32" {
        } else {
            return Err(BackendError::Start(format!(
                "DType {dtype} is not supported"
            )));
        };

        // Check model type
        let pool = match model_type {
            ModelType::Classifier => Pool::Cls,
            ModelType::Embedding(pool) => match pool {
                Pool::Splade | Pool::LastToken => {
                    return Err(BackendError::Start(format!(
                        "Pooling {pool} is not supported for this backend. Use `candle` backend instead."
                    )));
                }
                pool => pool,
            },
        };

        // Get model path
        let onnx_path = {
            let default_path = model_path.join("model.onnx");
            match default_path.exists() {
                true => default_path,
                false => model_path.join("onnx/model.onnx"),
            }
        };

        // Start onnx session
        let session = Session::builder()
            .s()?
            .with_intra_threads(num_cpus::get())
            .s()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .s()?
            .commit_from_file(onnx_path)
            .s()?;

        // Check if the model requires type tokens
        let mut type_id_name = None;
        for input in &session.inputs {
            if &input.name == "token_type_ids" || &input.name == "input_type" {
                type_id_name = Some(input.name.clone());
                break;
            }
        }

        Ok(Self {
            session,
            pool,
            type_id_name,
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

        let (input_ids, type_ids, input_lengths, attention_mask) = {
            let elems = batch_size * max_length;

            if batch_size > 1 {
                // Prepare padded batch
                let mut input_ids = Vec::with_capacity(elems);
                let mut type_ids = Vec::with_capacity(elems);
                let mut attention_mask = Vec::with_capacity(elems);
                let mut input_lengths = Vec::with_capacity(batch_size);

                for i in 0..batch_size {
                    let start = batch.cumulative_seq_lengths[i] as usize;
                    let end = batch.cumulative_seq_lengths[i + 1] as usize;
                    let seq_length = (end - start) as u32;
                    input_lengths.push(seq_length as f32);

                    // Copy values
                    for j in start..end {
                        input_ids.push(batch.input_ids[j] as i64);
                        type_ids.push(batch.token_type_ids[j] as i64);
                        attention_mask.push(1_i64);
                    }

                    // Add padding if needed
                    let padding = batch.max_length - seq_length;
                    if padding > 0 {
                        // Set bool to use attention mask
                        masking = true;
                        for _ in 0..padding {
                            input_ids.push(0);
                            type_ids.push(0);
                            attention_mask.push(0_i64);
                        }
                    }
                }
                (input_ids, type_ids, input_lengths, attention_mask)
            } else {
                let attention_mask = vec![1_i64; elems];

                (
                    batch.input_ids.into_iter().map(|v| v as i64).collect(),
                    batch.token_type_ids.into_iter().map(|v| v as i64).collect(),
                    vec![batch.max_length as f32],
                    attention_mask,
                )
            }
        };

        // Create ndarrays
        let input_ids = ndarray::Array2::from_shape_vec((batch_size, max_length), input_ids).e()?;
        let attention_mask =
            ndarray::Array2::from_shape_vec((batch_size, max_length), attention_mask).e()?;
        let input_lengths = ndarray::Array1::from_vec(input_lengths);

        // Create onnx inputs
        let inputs = match self.type_id_name.as_ref() {
            Some(type_id_name) => {
                // Add type ids to inputs
                let type_ids =
                    ndarray::Array2::from_shape_vec((batch_size, max_length), type_ids).e()?;
                ort::inputs!["input_ids" => input_ids, "attention_mask" => attention_mask.clone(), type_id_name => type_ids].e()?
            }
            None => {
                ort::inputs!["input_ids" => input_ids, "attention_mask" => attention_mask.clone()]
                    .e()?
            }
        };

        // Run model
        let outputs = self.session.run(inputs).e()?;
        // Get last_hidden_state ndarray

        let outputs = outputs
            .get("last_hidden_state")
            .or(outputs.get("token_embeddings"))
            .ok_or(BackendError::Inference(format!(
                "Unknown output keys: {:?}",
                self.session.outputs
            )))?
            .try_extract_tensor::<f32>()
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
                // CLS pooling
                Pool::Cls => outputs.slice(s![.., 0, ..]).into_owned().into_dyn(),
                // Last token pooling is not supported for this model
                Pool::LastToken => unreachable!(),
                // Mean pooling
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

        let (input_ids, type_ids, attention_mask) = {
            let elems = batch_size * max_length;

            if batch_size > 1 {
                // Prepare padded batch
                let mut input_ids = Vec::with_capacity(elems);
                let mut type_ids = Vec::with_capacity(elems);
                let mut attention_mask = Vec::with_capacity(elems);

                for i in 0..batch_size {
                    let start = batch.cumulative_seq_lengths[i] as usize;
                    let end = batch.cumulative_seq_lengths[i + 1] as usize;
                    let seq_length = (end - start) as u32;

                    // Copy values
                    for j in start..end {
                        input_ids.push(batch.input_ids[j] as i64);
                        type_ids.push(batch.token_type_ids[j] as i64);
                        attention_mask.push(1_i64);
                    }

                    // Add padding if needed
                    let padding = batch.max_length - seq_length;
                    if padding > 0 {
                        for _ in 0..padding {
                            input_ids.push(0);
                            type_ids.push(0);
                            attention_mask.push(0_i64);
                        }
                    }
                }
                (input_ids, type_ids, attention_mask)
            } else {
                let attention_mask = vec![1_i64; elems];

                (
                    batch.input_ids.into_iter().map(|v| v as i64).collect(),
                    batch.token_type_ids.into_iter().map(|v| v as i64).collect(),
                    attention_mask,
                )
            }
        };

        // Create ndarrays
        let input_ids = ndarray::Array2::from_shape_vec((batch_size, max_length), input_ids).e()?;
        let attention_mask =
            ndarray::Array2::from_shape_vec((batch_size, max_length), attention_mask).e()?;

        // Create onnx inputs
        let inputs = match self.type_id_name.as_ref() {
            Some(type_id_name) => {
                // Add type ids to inputs
                let type_ids =
                    ndarray::Array2::from_shape_vec((batch_size, max_length), type_ids).e()?;
                ort::inputs!["input_ids" => input_ids, "attention_mask" => attention_mask.clone(), type_id_name => type_ids].e()?
            }
            None => {
                ort::inputs!["input_ids" => input_ids, "attention_mask" => attention_mask.clone()]
                    .e()?
            }
        };

        // Run model
        let outputs = self.session.run(inputs).e()?;
        // Get last_hidden_state ndarray
        let outputs = outputs["logits"]
            .try_extract_tensor::<f32>()
            .e()?
            .to_owned();

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
