use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;
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

pub fn download_artifacts(
    model_id: &'static str,
    revision: Option<&'static str>,
) -> Result<PathBuf> {
    let builder = ApiBuilder::new().with_progress(false);

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

    let model_root = match api_repo.get("model.safetensors") {
        Ok(p) => p,
        Err(_) => {
            let p = api_repo.get("pytorch_model.bin")?;
            tracing::warn!("`model.safetensors` not found. Using `pytorch_model.bin` instead. Model loading will be significantly slower.");
            p
        }
    }
        .parent().unwrap()
        .to_path_buf();
    Ok(model_root)
}

pub fn relative_matcher() -> YamlMatcher<SnapshotScores> {
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
            tokenizer.with_pre_tokenizer(PreTokenizerWrapper::Metaspace(m));
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
                tokenizer.with_pre_tokenizer(PreTokenizerWrapper::Sequence(Sequence::new(
                    new_pre_tokenizers,
                )));
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
    }
}
