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

    api.get("config.json").await?;
    api.get("tokenizer.json").await?;

    let model_root = match api.get("model.safetensors").await {
        Ok(p) => p,
        Err(_) => {
            let p = api.get("pytorch_model.bin").await?;
            tracing::warn!("`model.safetensors` not found. Using `pytorch_model.bin` instead. Model loading will be significantly slower.");
            p
        }
    }
        .parent()
        .unwrap()
        .to_path_buf();

    tracing::info!("Model artifacts downloaded in {:?}", start.elapsed());
    Ok(model_root)
}

#[instrument(skip_all)]
pub async fn download_pool_config(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    let pool_config_path = api.get("1_Pooling/config.json").await?;
    Ok(pool_config_path)
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
