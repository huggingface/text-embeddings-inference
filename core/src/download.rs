use hf_hub::api::tokio::{ApiError, ApiRepo};
use std::path::PathBuf;
use tracing::instrument;

#[instrument(skip_all)]
pub async fn download_artifacts(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    let start = std::time::Instant::now();

    tracing::info!("Starting download");

    let model_root = api
        .get("model.safetensors")
        .await?
        .parent()
        .unwrap()
        .to_path_buf();
    api.get("config.json").await?;
    api.get("1_Pooling/config.json").await?;
    api.get("tokenizer.json").await?;

    tracing::info!("Model artifacts downloaded in {:?}", start.elapsed());
    Ok(model_root)
}
