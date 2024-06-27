use hf_hub::api::tokio::{ApiError, ApiRepo};
use std::path::PathBuf;
use tracing::instrument;

// Old classes used other config names than 'sentence_bert_config.json'
pub const ST_CONFIG_NAMES: [&str; 7] = [
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "sentence_camembert_config.json",
    "sentence_albert_config.json",
    "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
];

#[instrument(skip_all)]
pub async fn download_artifacts(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    let start = std::time::Instant::now();

    tracing::info!("Starting download");

    tracing::info!("Downloading `config.json`");
    api.get("config.json").await?;

    tracing::info!("Downloading `tokenizer.json`");
    api.get("tokenizer.json").await?;

    let model_files = match download_safetensors(api).await {
        Ok(p) => p,
        Err(_) => {
            tracing::warn!("safetensors weights not found. Using `pytorch_model.bin` instead. Model loading will be significantly slower.");
            tracing::info!("Downloading `pytorch_model.bin`");
            let p = api.get("pytorch_model.bin").await?;
            vec![p]
        }
    };
    let model_root = model_files[0].parent().unwrap().to_path_buf();

    tracing::info!("Model artifacts downloaded in {:?}", start.elapsed());
    Ok(model_root)
}

#[instrument(skip_all)]
pub async fn download_pool_config(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    tracing::info!("Downloading `1_Pooling/config.json`");
    let pool_config_path = api.get("1_Pooling/config.json").await?;
    Ok(pool_config_path)
}

async fn download_safetensors(api: &ApiRepo) -> Result<Vec<PathBuf>, ApiError> {
    // Single file
    tracing::info!("Downloading `model.safetensors`");
    match api.get("model.safetensors").await {
        Ok(p) => return Ok(vec![p]),
        Err(err) => tracing::warn!("Could not download `model.safetensors`: {}", err),
    };

    // Sharded weights
    // Download and parse index file
    tracing::info!("Downloading `model.safetensors.index.json`");
    let index_file = api.get("model.safetensors.index.json").await?;
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
        safetensors_files.push(api.get(&n).await?);
    }

    Ok(safetensors_files)
}

#[instrument(skip_all)]
pub async fn download_st_config(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    // Try default path
    let err = match api.get(ST_CONFIG_NAMES[0]).await {
        Ok(st_config_path) => return Ok(st_config_path),
        Err(err) => err,
    };

    for name in &ST_CONFIG_NAMES[1..] {
        if let Ok(st_config_path) = api.get(name).await {
            return Ok(st_config_path);
        }
    }

    Err(err)
}

#[instrument(skip_all)]
pub async fn download_new_st_config(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    tracing::info!("Downloading `config_sentence_transformers.json`");
    let pool_config_path = api.get("config_sentence_transformers.json").await?;
    Ok(pool_config_path)
}
