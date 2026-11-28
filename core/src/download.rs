use hf_hub::api::tokio::{ApiError, ApiRepo};
use std::path::PathBuf;
use tracing::instrument;

// `sentence_bert_config.json` default Sentence Transformers configuration file name, and other
// former / deprecated file names no longer used, but here for backwards compatibility
pub const ST_CONFIG_NAMES: [&str; 7] = [
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "sentence_camembert_config.json",
    "sentence_albert_config.json",
    "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
];

/// Helper to detect if an ApiError is caused by a missing Content-Range header
/// (common when downloading through caching proxies like JFrog Artifactory or certain mirrors
/// that respond to Range requests with HTTP 200 without Content-Range headers).
pub fn is_content_range_missing(err: &ApiError) -> bool {
    match err {
        ApiError::MissingHeader(h) => h.as_str().eq_ignore_ascii_case("content-range"),
        _ => {
            let msg = err.to_string().to_lowercase();
            msg.contains("content-range is missing") || msg.contains("header content-range")
        }
    }
}

/// Fallback full-file download for environments where range requests omit `Content-Range`
/// (e.g., JFrog Artifactory proxying the Hugging Face Hub).
async fn download_file_direct(api: &ApiRepo, file_path: &str) -> Result<PathBuf, ApiError> {
    let url = api.url(file_path);
    tracing::info!(
        "Falling back to direct GET download for `{}`: {}",
        file_path,
        url
    );

    let client = reqwest::Client::builder()
        .user_agent("text-embeddings-inference")
        .build()
        .map_err(|e| {
            ApiError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

    let mut request = client.get(&url);
    if let Ok(token) =
        std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
    {
        request = request.bearer_auth(token);
    }

    let response = request.send().await.map_err(|e| {
        ApiError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            e.to_string(),
        ))
    })?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(ApiError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("File `{file_path}` not found (404)"),
        )));
    }

    let response = response.error_for_status().map_err(|e| {
        ApiError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            e.to_string(),
        ))
    })?;

    // Determine cache destination root
    let cache_dir = if let Ok(dir) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        PathBuf::from(dir)
    } else if let Ok(dir) = std::env::var("HF_HUB_CACHE") {
        PathBuf::from(dir)
    } else if let Ok(home) = std::env::var("HF_HOME") {
        PathBuf::from(home).join("hub")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub")
    } else if let Ok(userprofile) = std::env::var("USERPROFILE") {
        PathBuf::from(userprofile)
            .join(".cache")
            .join("huggingface")
            .join("hub")
    } else {
        std::env::temp_dir().join("huggingface").join("hub")
    };

    // Extract repo folder & revision from URL format: .../{repo_id}/resolve/{revision}/{file_path}
    let (repo_part, revision_part) =
        if let Some((before_res, after_res)) = url.split_once("/resolve/") {
            let repo_id = before_res
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .split_once('/')
                .map(|(_, path)| path)
                .unwrap_or(before_res);
            let revision = after_res.split('/').next().unwrap_or("main");
            (repo_id.replace('/', "--"), revision.replace("%2F", "/"))
        } else {
            ("default".to_string(), "main".to_string())
        };

    let repo_folder = if repo_part.starts_with("models--") {
        repo_part
    } else {
        format!("models--{}", repo_part)
    };

    let target_dir = cache_dir
        .join(repo_folder)
        .join("snapshots")
        .join(revision_part);
    tokio::fs::create_dir_all(&target_dir)
        .await
        .map_err(ApiError::IoError)?;

    let file_dest = target_dir.join(file_path);
    if let Some(parent) = file_dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(ApiError::IoError)?;
    }

    let bytes = response.bytes().await.map_err(|e| {
        ApiError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            e.to_string(),
        ))
    })?;

    tokio::fs::write(&file_dest, &bytes)
        .await
        .map_err(ApiError::IoError)?;

    tracing::info!("Saved `{}` to {:?}", file_path, file_dest);
    Ok(file_dest)
}

async fn download_file(api: &ApiRepo, file_path: &str) -> Result<PathBuf, ApiError> {
    tracing::info!("Downloading `{}`", file_path);
    match api.get(file_path).await {
        Ok(path) => Ok(path),
        Err(err) if is_content_range_missing(&err) => {
            tracing::warn!(
                "Content-Range header missing when downloading `{file_path}` via Range request (common with caching proxies such as Artifactory); falling back to direct full GET download"
            );
            download_file_direct(api, file_path).await
        }
        Err(err) => Err(err),
    }
}

async fn download_st_config_legacy(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    // Try to download first the default path i.e., `sentence_bert_config.json`
    let err = match download_file(api, ST_CONFIG_NAMES[0]).await {
        Ok(st_config_path) => return Ok(st_config_path),
        Err(err) => err,
    };

    // Then try with the rest of the legacy configuration file names
    for name in &ST_CONFIG_NAMES[1..] {
        if let Ok(st_config_path) = download_file(api, name).await {
            return Ok(st_config_path);
        }
    }

    Err(err)
}

#[instrument(skip_all)]
pub async fn download_artifacts(api: &ApiRepo, pool_config: bool) -> Result<PathBuf, ApiError> {
    let start = std::time::Instant::now();
    tracing::info!("Starting download");

    // Try to download `1_Pooling`, only if `--pooling` hasn't been provided, otherwise, the
    // `--pooling` argument will be used instead.
    if pool_config {
        let _ = download_file(api, "1_Pooling/config.json")
            .await
            .map_err(|err| {
                tracing::warn!("Download failed: {err}");
                err
            });
    }

    // Download the legacy Sentence Transformers configuration files as defined in
    // `ST_CONFIG_NAMES` (no warn on failure as it's a legacy file)
    // NOTE: used to define the `max_seq_length`, otherwise it will be defined as
    // `max_position_embeddings - position_offset` from the `config.json` file
    let _ = download_st_config_legacy(api).await;

    // Download the actual Sentence Transformers configuration file
    let _ = download_file(api, "config_sentence_transformers.json")
        .await
        .map_err(|err| {
            tracing::warn!("Download failed: {err}");
            err
        });

    download_file(api, "config.json").await?;
    let path = download_file(api, "tokenizer.json").await?;

    tracing::info!("Model artifacts downloaded in {:?}", start.elapsed());

    Ok(path.parent().unwrap().to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_content_range_missing() {
        let err = ApiError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Header content-range is missing",
        ));
        assert!(is_content_range_missing(&err));

        let err_upper = ApiError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Header Content-Range is missing",
        ));
        assert!(is_content_range_missing(&err_upper));

        let err_other = ApiError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Header etag is missing",
        ));
        assert!(!is_content_range_missing(&err_other));
    }
}
