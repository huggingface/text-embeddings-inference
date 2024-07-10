use ndarray::s;
use nohash_hasher::BuildNoHashHasher;
use ort::{GraphOptimizationLevel, Session};
use std::collections::HashMap;
use std::path::PathBuf;
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, Bm42Params, Embedding, Embeddings, ModelType, Pool, Predictions,
};

pub struct Bm42Backend {
    session: Session,
    pool: Pool,
    type_id_name: Option<String>,
    invert_vocab: HashMap<u32, String>,
    punctuation: Vec<String>,
    alpha: f32,
    stemmer: rust_stemmers::Stemmer,
    stopwords: Vec<String>,
    special_tokens: Vec<String>,
}

impl Bm42Backend {
    pub fn new(
        model_path: PathBuf,
        dtype: String,
        model_type: ModelType,
        model_params: Bm42Params,
    ) -> Result<Self, BackendError> {
        // Check dtype
        if &dtype == "float32" {
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

        let punctuation = &[
            "!", "\"", "#", "$", "%", "&", "\'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";",
            "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
        ];
        let punctuation: Vec<String> = punctuation.iter().map(|x| x.to_string()).collect();
        let punctuation = punctuation.to_vec();

        Ok(Self {
            session,
            pool,
            type_id_name,
            invert_vocab: model_params.invert_vocab,
            punctuation,
            stemmer: rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English),
            alpha: 0.5,
            stopwords: model_params.stopwords,
            special_tokens: model_params.special_tokens,
        })
    }
}

impl Backend for Bm42Backend {
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
        println!("himoadf");
        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        // Whether a least one of the request in the batch is padded
        let mut masking = true;

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
        let inputs = match (self.type_id_name.as_ref(), &self.pool) {
            (_, Pool::BM42) => {
                println!("input_ids {:?}", input_ids);
                (ort::inputs![
                    "input_ids" => ort::Value::from_array(input_ids.clone()).unwrap()
                ])
                .e()?
            }
            (Some(type_id_name), _) => {
                // Add type ids to inputs
                let type_ids =
                    ndarray::Array2::from_shape_vec((batch_size, max_length), type_ids).e()?;
                ort::inputs!["input_ids" => input_ids.clone(), "attention_mask" => attention_mask.clone(), type_id_name => type_ids].e()?
            }
            (None, _) => {
                ort::inputs!["input_ids" => input_ids.clone(), "attention_mask" => attention_mask.clone()]
                    .e()?
            }
        };
        // Run model
        let session_outputs = self.session.run(inputs).e()?;

        // Final embeddings struct
        let mut embeddings =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());

        let outputs = {
            let mut output_final: Vec<Vec<Vec<f32>>> = Vec::with_capacity(batch_size);

            let outputs = session_outputs
                .get("attention_6")
                .ok_or(BackendError::Inference(format!(
                    "Unknown output keys: {:?}",
                    self.session.outputs
                )))?
                .try_extract_tensor::<f32>()
                .e()?
                .to_owned();

            for i in 0..batch_size {
                let output: Vec<Vec<f32>> = outputs
                    .view()
                    .slice(s![i, .., 0, ..])
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect();

                output_final.push(output);
            }

            output_final
        };

        let has_pooling_requests = !batch.pooled_indices.is_empty();

        if has_pooling_requests {
            let outputs = outputs.clone();

            let pooled_embeddings: Vec<Vec<f32>> = match self.pool {
                Pool::BM42 => outputs
                    .iter()
                    .map(|output| {
                        let mean =
                            output
                                .iter()
                                .fold(vec![0.0; output[0].len()], |acc, inner_vec| {
                                    acc.iter().zip(inner_vec).map(|(&a, &b)| a + b).collect()
                                });
                        let mean = mean
                            .iter()
                            .map(|&sum| sum / output.len() as f32)
                            .collect::<Vec<f32>>();

                        mean.iter()
                            .zip(attention_mask.clone())
                            .map(|(m, a)| m * a as f32)
                            .collect()
                    })
                    .collect(),
                _ => unreachable!(),
            };

            println!("fda");
            let mut rescored_vectors = vec![];

            for i in 0..input_ids.slice(s![.., 0]).len() {
                let document_token_ids = input_ids.slice(s![i, ..]);
                let attention_value = &pooled_embeddings[i];

                let doc_tokens_with_ids: Vec<(usize, String)> = document_token_ids
                    .iter()
                    .enumerate()
                    .map(|(idx, &id)| (idx, self.invert_vocab[&(id as u32)].clone()))
                    .collect();

                println!("spe {:?}", self.special_tokens);
                let reconstructed =
                    reconstruct_bpe(doc_tokens_with_ids, &self.special_tokens.clone());
                println!("rec {:?}", reconstructed);

                let filtered =
                    filter_pair_tokens(reconstructed, &self.stopwords, &self.punctuation);
                println!("fffc {:?}", filtered);

                let stemmed = stem_pair_tokens(&self.stemmer, filtered);
                println!("stemme {:?}", stemmed);

                let weighted = aggregate_weights(&stemmed, attention_value);
                println!("weights {:?}", weighted);

                let mut max_token_weight: HashMap<String, f32> = HashMap::new();

                weighted.into_iter().for_each(|(token, weight)| {
                    let weight = max_token_weight.get(&token).unwrap_or(&0.0).max(weight);
                    max_token_weight.insert(token, weight);
                });

                let rescored = rescore_vector(&max_token_weight, self.alpha);

                let max_value = *rescored.keys().max().unwrap_or(&0) as usize;

                // Convert HashMap<k, v> into vec![]
                let mut embedding = ndarray::Array::zeros(max_value + 1);

                for (k, v) in rescored.iter() {
                    embedding[*k as usize] = *v
                }

                rescored_vectors.push(embedding);
            }

            for (i, e) in batch.pooled_indices.into_iter().zip(rescored_vectors) {
                embeddings.insert(i as usize, Embedding::Pooled(e.to_vec()));
            }
        };

        println!("returning out ");
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

pub fn stem_pair_tokens(
    stemmer: &rust_stemmers::Stemmer,
    tokens: Vec<(String, Vec<usize>)>,
) -> Vec<(String, Vec<usize>)> {
    let mut result: Vec<(String, Vec<usize>)> = Vec::new();

    for (token, value) in tokens.into_iter() {
        let processed_token = stemmer.stem(&token).to_string();
        result.push((processed_token, value));
    }

    result
}

pub fn rescore_vector(vector: &HashMap<String, f32>, alpha: f32) -> HashMap<i32, f32> {
    let mut new_vector: HashMap<i32, f32> = HashMap::new();

    for (token, &value) in vector.iter() {
        let token_id =
            (murmur3::murmur3_32(&mut std::io::Cursor::new(token), 0).unwrap() as i32).abs();

        let new_score = (1.0 + value).ln().powf(alpha);

        new_vector.insert(token_id, new_score);
    }

    new_vector
}

pub fn aggregate_weights(tokens: &[(String, Vec<usize>)], weights: &[f32]) -> Vec<(String, f32)> {
    let mut result: Vec<(String, f32)> = Vec::new();

    for (token, idxs) in tokens.iter() {
        let sum_weight: f32 = idxs.iter().map(|&idx| weights[idx]).sum();
        result.push((token.clone(), sum_weight));
    }

    result
}

pub fn filter_pair_tokens(
    tokens: Vec<(String, Vec<usize>)>,
    stopwords: &[String],
    punctuation: &[String],
) -> Vec<(String, Vec<usize>)> {
    let mut result: Vec<(String, Vec<usize>)> = Vec::new();

    for (token, value) in tokens.into_iter() {
        if stopwords.contains(&token) || punctuation.contains(&token) {
            continue;
        }
        result.push((token.clone(), value));
    }

    result
}

pub fn reconstruct_bpe(
    bpe_tokens: impl IntoIterator<Item = (usize, String)>,
    special_tokens: &[String],
) -> Vec<(String, Vec<usize>)> {
    let mut result = Vec::new();
    let mut acc = String::new();
    let mut acc_idx = Vec::new();

    let continuing_subword_prefix = "##";
    let continuing_subword_prefix_len = continuing_subword_prefix.len();

    for (idx, token) in bpe_tokens {
        if special_tokens.contains(&token) {
            continue;
        }

        if token.starts_with(continuing_subword_prefix) {
            acc.push_str(&token[continuing_subword_prefix_len..]);
            acc_idx.push(idx);
        } else {
            if !acc.is_empty() {
                result.push((acc.clone(), acc_idx.clone()));
                acc_idx = vec![];
            }
            acc = token;
            acc_idx.push(idx);
        }
    }

    if !acc.is_empty() {
        result.push((acc, acc_idx));
    }

    result
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
