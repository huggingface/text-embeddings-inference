use hf_hub::api::tokio::{ApiError, ApiRepo};
use std::path::PathBuf;
use tracing::instrument;

#[instrument(skip_all)]
pub async fn download_artifacts(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    let start = std::time::Instant::now();

    tracing::info!("Starting download");

    let model_root = match api.get("model.safetensors").await {
        Ok(p) => p,
        Err(_) => {
            tracing::warn!("`model.safetensors` not found. Using `pytorch_model.bin` instead. Model loading will be significantly slower.");
            api.get("pytorch_model.bin").await?
        },
    }
    .parent()
    .unwrap()
    .to_path_buf();

    api.get("config.json").await?;
    api.get("1_Pooling/config.json").await?;
    api.get("tokenizer.json").await?;

    tracing::info!("Model artifacts downloaded in {:?}", start.elapsed());
    Ok(model_root)
}
