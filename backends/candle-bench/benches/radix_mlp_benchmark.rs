use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};
use text_embeddings_core::radix_mlp::compute_fold_and_scatter;

use hf_hub::api::sync::{ApiBuilder, ApiError, ApiRepo};
use hf_hub::{Repo, RepoType};
use std::path::PathBuf;

/// huggingface hub downloader
pub fn download_artifacts(
    model_id: &'static str,
    revision: Option<&'static str>,
) -> Result<PathBuf> {
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

    let model_root = model_files[0].parent().unwrap().to_path_buf();
    Ok(model_root)
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

#[derive(Debug, Clone)]
struct Batch {
    input_ids: Vec<u32>,
    token_type_ids: Vec<u32>,
    position_ids: Vec<u32>,
    cumulative_seq_lengths: Vec<u32>,
    max_length: u32,
    pooled_indices: Vec<u32>,
    raw_indices: Vec<u32>,
    compact_input_ids: Option<Vec<u32>>,
    compact_position_ids: Option<Vec<u32>>,
    scatter_unfold: Option<Vec<u32>>,
    fold_gather: Option<Vec<u32>>,
}

impl From<Batch> for text_embeddings_backend_core::Batch {
    fn from(b: Batch) -> Self {
        text_embeddings_backend_core::Batch {
            input_ids: b.input_ids,
            token_type_ids: b.token_type_ids,
            position_ids: b.position_ids,
            cumulative_seq_lengths: b.cumulative_seq_lengths,
            max_length: b.max_length,
            pooled_indices: b.pooled_indices,
            raw_indices: b.raw_indices,
            compact_input_ids: b.compact_input_ids,
            compact_position_ids: b.compact_position_ids,
            scatter_unfold: b.scatter_unfold,
            fold_gather: b.fold_gather,
        }
    }
}

/// Sets up the backend and batch data needed for the benchmark.
fn setup(
    _backend: &CandleBackend,
    batch_size: usize,
    shared_prefix_len: usize,
    unique_suffix_len: usize,
) -> Result<(Batch, Batch, Batch)> {
    // 2. Create benchmark batch
    let shared_prefix_ids: Vec<u32> = vec![1; shared_prefix_len];

    let mut all_input_ids = Vec::new();
    let mut all_position_ids = Vec::new();
    let mut cumulative_seq_lengths: Vec<u32> = vec![0];
    let mut current_len: u32 = 0;

    for i in 0..batch_size {
        let unique_suffix_ids: Vec<u32> = vec![(i + 2) as u32; unique_suffix_len];
        let mut sequence_ids = shared_prefix_ids.clone();
        sequence_ids.extend(&unique_suffix_ids);

        let seq_len = sequence_ids.len();
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();

        current_len += seq_len as u32;
        all_input_ids.extend(sequence_ids);
        all_position_ids.extend(position_ids);
        cumulative_seq_lengths.push(current_len);
    }

    let max_length = (shared_prefix_len + unique_suffix_len) as u32;

    // Compute RadixMLP fold/scatter indices
    let (compact_input_ids, compact_position_ids, scatter_unfold, fold_gather) =
        compute_fold_and_scatter(
            &all_input_ids,
            &all_position_ids,
            &cumulative_seq_lengths,
            true,
        );

    let (compact_input_ids_un, compact_position_ids_un, scatter_unfold_un, fold_gather_un) =
        compute_fold_and_scatter(
            &all_input_ids,
            &all_position_ids,
            &cumulative_seq_lengths,
            false,
        );

    println!(
        "RadixMLP compression (prefix={}, suffix={}): {} original tokens -> {} compact tokens ({:.1}% reduction)",
        shared_prefix_len,
        unique_suffix_len,
        all_input_ids.len(),
        compact_input_ids.len(),
        (1.0 - compact_input_ids.len() as f64 / all_input_ids.len() as f64) * 100.0
    );

    let token_type_ids = vec![0u32; all_input_ids.len()];
    let pooled_indices: Vec<u32> = (0..batch_size as u32).collect();

    // Batch with RadixMLP enabled
    let enabled_batch = Batch {
        input_ids: all_input_ids.clone(),
        token_type_ids: token_type_ids.clone(),
        position_ids: all_position_ids.clone(),
        cumulative_seq_lengths: cumulative_seq_lengths.clone(),
        max_length,
        pooled_indices: pooled_indices.clone(),
        raw_indices: vec![],
        compact_input_ids: Some(compact_input_ids),
        compact_position_ids: Some(compact_position_ids),
        scatter_unfold: Some(scatter_unfold),
        fold_gather: Some(fold_gather),
    };

    let enabled_batch_unpadded = Batch {
        input_ids: all_input_ids.clone(),
        token_type_ids: token_type_ids.clone(),
        position_ids: all_position_ids.clone(),
        cumulative_seq_lengths: cumulative_seq_lengths.clone(),
        max_length,
        pooled_indices: pooled_indices.clone(),
        raw_indices: vec![],
        compact_input_ids: Some(compact_input_ids_un),
        compact_position_ids: Some(compact_position_ids_un),
        scatter_unfold: Some(scatter_unfold_un),
        fold_gather: Some(fold_gather_un),
    };

    // Batch with RadixMLP disabled (None for all compact fields)
    let disabled_batch = Batch {
        input_ids: all_input_ids,
        token_type_ids,
        position_ids: all_position_ids,
        cumulative_seq_lengths,
        max_length,
        pooled_indices,
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
    };

    Ok((enabled_batch, disabled_batch, enabled_batch_unpadded))
}

/// The main benchmark function.
fn bench_radix_mlp(c: &mut Criterion) {
    // 1. Setup backend
    let model_root = download_artifacts("Qwen/Qwen3-Embedding-0.6B", None)
        .expect("Failed to download artifacts");
    println!("Model downloaded to {:?}", model_root);
    let backend = CandleBackend::new(
        &model_root,
        "float16".to_string(),
        ModelType::Embedding(Pool::LastToken),
        None,
    )
    .expect("Could not start backend");
    println!("Backend initialized");

    let batch_size = 32;
    let size_configs = [(512, 256), (512, 512), (1024, 1024)];

    for (shared_prefix_len, unique_suffix_len) in size_configs {
        let (enabled_batch, disabled_batch, enabled_batch_unpadded) = setup(
            &backend,
            batch_size,
            shared_prefix_len,
            unique_suffix_len,
        )
        .expect("Failed to set up benchmark");

        // --- Correctness Check ---
        let radix_result = backend.embed(enabled_batch.clone().into()).unwrap();
        let regular_result = backend.embed(disabled_batch.clone().into()).unwrap();

        let radix_vecs: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| match radix_result.get(&i).unwrap() {
                text_embeddings_backend_core::Embedding::Pooled(v) => v.clone(),
                text_embeddings_backend_core::Embedding::All(vecs) => vecs.last().unwrap().clone(),
            })
            .collect();
        let regular_vecs: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| match regular_result.get(&i).unwrap() {
                text_embeddings_backend_core::Embedding::Pooled(v) => v.clone(),
                text_embeddings_backend_core::Embedding::All(vecs) => vecs.last().unwrap().clone(),
            })
            .collect();

        assert_eq!(radix_vecs.len(), regular_vecs.len());
        for i in 0..radix_vecs.len() {
            let diff: f32 = radix_vecs[i]
                .iter()
                .zip(regular_vecs[i].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                diff < 1e-2,
                "Correctness check failed for size ({}, {}): Embeddings for item {} differ by {}",
                shared_prefix_len,
                unique_suffix_len,
                i,
                diff
            );
        }
        println!(
            "Correctness check passed for size ({}, {}). Starting benchmark...",
            shared_prefix_len, unique_suffix_len
        );
        // --- End Correctness Check ---

        let mut group = c.benchmark_group(&format!(
            "RadixMLP Speedup (prefix: {}, suffix: {})",
            shared_prefix_len, unique_suffix_len
        ));
        group
            .sample_size(10)
            .warm_up_time(std::time::Duration::from_secs(3))
            .measurement_time(std::time::Duration::from_secs(15));

        // Benchmark WITH RadixMLP enabled (uses shared prefix computation)
        group.bench_function("radix_mlp_enabled", |b| {
            b.iter(|| backend.embed(enabled_batch.clone().into()).unwrap())
        });

        // Benchmark WITH RadixMLP enabled but without padding (uses shared prefix computation)
        group.bench_function("radix_mlp_enabled_unpadded", |b| {
            b.iter(|| {
                backend
                    .embed(enabled_batch_unpadded.clone().into())
                    .unwrap()
            })
        });

        // Benchmark WITHOUT RadixMLP (standard full computation)
        group.bench_function("radix_mlp_disabled", |b| {
            b.iter(|| backend.embed(disabled_batch.clone().into()).unwrap())
        });

        group.finish();
    }
}

criterion_group!(benches, bench_radix_mlp);
criterion_main!(benches);
