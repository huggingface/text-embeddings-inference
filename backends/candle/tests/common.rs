use anyhow::Result;
use hf_hub::api::sync::{ApiBuilder, ApiError, ApiRepo};
use hf_hub::{Repo, RepoType};
use insta::internals::YamlMatcher;
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use text_embeddings_backend_core::{Batch, Embedding, Embeddings};
use tokenizers::pre_tokenizers::metaspace::PrependScheme;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::{Encoding, PreTokenizerWrapper, Tokenizer};

#[derive(Serialize, Deserialize, Debug)]
pub struct Score(f32);

impl Score {
    fn is_close(&self, other: &Self, abs_tol: f32) -> bool {
        is_close::default()
            .abs_tol(abs_tol)
            .is_close(self.0, other.0)
    }
}

impl PartialEq for Score {
    fn eq(&self, other: &Self) -> bool {
        // Default tolerance for equality
        self.is_close(other, 5e-3)
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct SnapshotScores(Vec<Vec<Score>>);

impl Deref for SnapshotScores {
    type Target = Vec<Vec<Score>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec<Vec<f32>>> for SnapshotScores {
    fn from(value: Vec<Vec<f32>>) -> Self {
        Self(
            value
                .into_iter()
                .map(|v| v.into_iter().map(Score).collect())
                .collect(),
        )
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SnapEmbedding(Vec<f32>);

impl PartialEq for SnapEmbedding {
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.0.len(), other.0.len());

        let mut sumxx = 0.0;
        let mut sumyy = 0.0;
        let mut sumxy = 0.0;

        for (x, y) in self.0.iter().zip(other.0.iter()) {
            sumxx += x * x;
            sumyy += y * y;
            sumxy += x * y;
        }

        (sumxy / (sumxx * sumyy).sqrt()) > 0.999
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct SnapshotEmbeddings(Vec<SnapEmbedding>);

impl Deref for SnapshotEmbeddings {
    type Target = Vec<SnapEmbedding>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec<Vec<f32>>> for SnapshotEmbeddings {
    fn from(value: Vec<Vec<f32>>) -> Self {
        Self(value.into_iter().map(SnapEmbedding).collect())
    }
}

pub fn sort_embeddings(embeddings: Embeddings) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut pooled_embeddings = Vec::new();
    let mut raw_embeddings = Vec::new();

    for (_, embedding) in embeddings {
        match embedding {
            Embedding::Pooled(e) => pooled_embeddings.push(e),
            Embedding::All(e) => raw_embeddings.extend(e),
        }
    }

    (pooled_embeddings, raw_embeddings)
}

#[derive(Deserialize, PartialEq)]
enum ModuleType {
    #[serde(rename = "sentence_transformers.models.Dense")]
    Dense,
    #[serde(rename = "sentence_transformers.models.Normalize")]
    Normalize,
    #[serde(rename = "sentence_transformers.models.Pooling")]
    Pooling,
    #[serde(rename = "sentence_transformers.models.Transformer")]
    Transformer,
}

#[derive(Deserialize)]
struct ModuleConfig {
    #[allow(dead_code)]
    idx: usize,
    #[allow(dead_code)]
    name: String,
    path: String,
    #[serde(rename = "type")]
    module_type: ModuleType,
}

pub fn download_artifacts(
    model_id: &'static str,
    revision: Option<&'static str>,
    dense_path: Option<&'static str>,
) -> Result<(PathBuf, Option<Vec<String>>)> {
    let mut builder = ApiBuilder::from_env().with_progress(false);

    if let Ok(token) = std::env::var("HF_TOKEN") {
        builder = builder.with_token(Some(token));
    }

    if let Some(cache_dir) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        builder = builder.with_cache_dir(cache_dir.into());
    }

    let api = builder.build().unwrap();
    let api_repo = if let Some(revision) = revision {
        api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ))
    } else {
        api.repo(Repo::new(model_id.to_string(), RepoType::Model))
    };

    api_repo.get("config.json")?;
    api_repo.get("tokenizer.json")?;

    let model_files = match download_safetensors(&api_repo) {
        Ok(p) => p,
        Err(_) => {
            tracing::warn!("safetensors weights not found. Using `pytorch_model.bin` instead. Model loading will be significantly slower.");
            tracing::info!("Downloading `pytorch_model.bin`");
            let p = api_repo.get("pytorch_model.bin")?;
            vec![p]
        }
    };

    let dense_paths = if let Ok(modules_path) = api_repo.get("modules.json") {
        match parse_dense_paths_from_modules(&modules_path) {
            Ok(paths) => match paths.len() {
                0 => None,
                1 => {
                    let path = if let Some(path) = dense_path {
                        path.to_string()
                    } else {
                        paths[0].clone()
                    };

                    download_dense_module(&api_repo, &path)?;
                    Some(vec![path])
                }
                _ => {
                    for path in &paths {
                        download_dense_module(&api_repo, path)?;
                    }
                    Some(paths)
                }
            },
            _ => None,
        }
    } else {
        None
    };

    let model_root = model_files[0].parent().unwrap().to_path_buf();
    Ok((model_root, dense_paths))
}

fn download_safetensors(api: &ApiRepo) -> Result<Vec<PathBuf>, ApiError> {
    // Single file
    tracing::info!("Downloading `model.safetensors`");
    match api.get("model.safetensors") {
        Ok(p) => return Ok(vec![p]),
        Err(err) => tracing::warn!("Could not download `model.safetensors`: {}", err),
    };

    // Sharded weights
    // Download and parse index file
    tracing::info!("Downloading `model.safetensors.index.json`");
    let index_file = api.get("model.safetensors.index.json")?;
    let index_file_string: String =
        std::fs::read_to_string(index_file).expect("model.safetensors.index.json is corrupted");
    let json: serde_json::Value = serde_json::from_str(&index_file_string)
        .expect("model.safetensors.index.json is corrupted");

    let weight_map = match json.get("weight_map") {
        Some(serde_json::Value::Object(map)) => map,
        _ => panic!("model.safetensors.index.json is corrupted"),
    };

    let mut safetensors_filenames = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_filenames.insert(file.to_string());
        }
    }

    // Download weight files
    let mut safetensors_files = Vec::new();
    for n in safetensors_filenames {
        tracing::info!("Downloading `{}`", n);
        safetensors_files.push(api.get(&n)?);
    }

    Ok(safetensors_files)
}

fn parse_dense_paths_from_modules(modules_path: &PathBuf) -> Result<Vec<String>, std::io::Error> {
    let content = std::fs::read_to_string(modules_path)?;
    let modules: Vec<ModuleConfig> = serde_json::from_str(&content)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;

    Ok(modules
        .into_iter()
        .filter(|module| module.module_type == ModuleType::Dense)
        .map(|module| module.path)
        .collect::<Vec<String>>())
}

fn download_dense_module(api: &ApiRepo, dense_path: &str) -> Result<PathBuf, ApiError> {
    let config_file = format!("{}/config.json", dense_path);
    tracing::info!("Downloading `{}`", config_file);
    let config_path = api.get(&config_file)?;

    let safetensors_file = format!("{}/model.safetensors", dense_path);
    tracing::info!("Downloading `{}`", safetensors_file);
    match api.get(&safetensors_file) {
        Ok(_) => {}
        Err(err) => {
            tracing::warn!("Could not download `{}`: {}", safetensors_file, err);
            let pytorch_file = format!("{}/pytorch_model.bin", dense_path);
            tracing::info!("Downloading `{}`", pytorch_file);
            api.get(&pytorch_file)?;
        }
    }

    Ok(config_path.parent().unwrap().to_path_buf())
}

#[allow(unused)]
pub(crate) fn relative_matcher() -> YamlMatcher<SnapshotScores> {
    YamlMatcher::new()
}

pub fn cosine_matcher() -> YamlMatcher<SnapshotEmbeddings> {
    YamlMatcher::new()
}

pub fn load_tokenizer(model_root: &Path) -> Result<Tokenizer> {
    // Load tokenizer
    let tokenizer_path = model_root.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).expect("tokenizer.json not found");
    // See https://github.com/huggingface/tokenizers/pull/1357
    if let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer() {
        if let PreTokenizerWrapper::Metaspace(m) = pre_tokenizer {
            // We are forced to clone since `Tokenizer` does not have a `get_mut` for `pre_tokenizer`
            let mut m = m.clone();
            m.set_prepend_scheme(PrependScheme::First);
            tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Metaspace(m)));
        } else if let PreTokenizerWrapper::Sequence(s) = pre_tokenizer {
            let pre_tokenizers = s.get_pre_tokenizers();
            // Check if we have a Metaspace pre tokenizer in the sequence
            let has_metaspace = pre_tokenizers
                .iter()
                .any(|t| matches!(t, PreTokenizerWrapper::Metaspace(_)));

            if has_metaspace {
                let mut new_pre_tokenizers = Vec::with_capacity(s.get_pre_tokenizers().len());

                for pre_tokenizer in pre_tokenizers {
                    if let PreTokenizerWrapper::WhitespaceSplit(_) = pre_tokenizer {
                        // Remove WhitespaceSplit
                        // This will be done by the Metaspace pre tokenizer
                        continue;
                    }

                    let mut pre_tokenizer = pre_tokenizer.clone();

                    if let PreTokenizerWrapper::Metaspace(ref mut m) = pre_tokenizer {
                        m.set_prepend_scheme(PrependScheme::First);
                    }
                    new_pre_tokenizers.push(pre_tokenizer);
                }
                tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(
                    new_pre_tokenizers,
                ))));
            }
        }
    }

    tokenizer.with_padding(None);
    Ok(tokenizer)
}

pub fn batch(encodings: Vec<Encoding>, pooled_indices: Vec<u32>, raw_indices: Vec<u32>) -> Batch {
    let mut input_ids = Vec::new();
    let mut token_type_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut cumulative_seq_lengths = Vec::with_capacity(encodings.len() + 1);
    cumulative_seq_lengths.push(0);

    let mut max_length = 0;
    let mut cumulative_length = 0;

    for encoding in encodings.iter() {
        let encoding_length = encoding.len() as u32;
        input_ids.extend(encoding.get_ids().to_vec());
        token_type_ids.extend(encoding.get_type_ids().to_vec());
        position_ids.extend(0..encoding_length);
        cumulative_length += encoding_length;
        cumulative_seq_lengths.push(cumulative_length);
        max_length = max(max_length, encoding_length);
    }

    Batch {
        input_ids,
        token_type_ids,
        position_ids,
        cumulative_seq_lengths,
        max_length,
        pooled_indices,
        raw_indices,
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
    }
}
