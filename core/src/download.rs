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
pub async fn download_artifacts(api: &ApiRepo, pool_config: bool) -> Result<PathBuf, ApiError> {
    let start = std::time::Instant::now();

    tracing::info!("Starting download");

    // Optionally download the pooling config.
    if pool_config {
        // If a pooling config exist, download it
        let _ = download_pool_config(api).await.map_err(|err| {
            tracing::warn!("Download failed: {err}");
            err
        });
    }

    // Download legacy sentence transformers config
    // We don't warn on failure as it is a legacy file
    let _ = download_st_config(api).await;
    // Download new sentence transformers config
    let _ = download_new_st_config(api).await.map_err(|err| {
        tracing::warn!("Download failed: {err}");
        err
    });

    tracing::info!("Downloading `config.json`");
    api.get("config.json").await?;

    tracing::info!("Downloading `tokenizer.json`");
    let tokenizer_path = api.get("tokenizer.json").await?;

    let model_root = tokenizer_path.parent().unwrap().to_path_buf();
    tracing::info!("Model artifacts downloaded in {:?}", start.elapsed());
    Ok(model_root)
}

#[instrument(skip_all)]
pub async fn download_pool_config(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    tracing::info!("Downloading `1_Pooling/config.json`");
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

#[instrument(skip_all)]
pub async fn download_new_st_config(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    tracing::info!("Downloading `config_sentence_transformers.json`");
    let pool_config_path = api.get("config_sentence_transformers.json").await?;
    Ok(pool_config_path)
}
