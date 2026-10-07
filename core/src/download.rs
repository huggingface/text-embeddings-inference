use hf_hub::{HFError as ApiError, HFRepository, RepoTypeModel};
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

/// Resolves `$HF_HOME`, defaulting to `~/.cache/huggingface`, as `hf-hub` doesn't read it
pub fn hf_home() -> PathBuf {
    std::env::var_os("HF_HOME")
        .filter(|hf_home| !hf_home.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface")
        })
}

async fn download_file(
    api: &HFRepository<RepoTypeModel>,
    revision: Option<&str>,
    file_path: &str,
) -> Result<PathBuf, ApiError> {
    tracing::info!("Downloading `{}`", file_path);
    api.download_file()
        .filename(file_path)
        .maybe_revision(revision)
        .send()
        .await
}

async fn download_st_config_legacy(
    api: &HFRepository<RepoTypeModel>,
    revision: Option<&str>,
) -> Result<PathBuf, ApiError> {
    // Try to download first the default path i.e., `sentence_bert_config.json`
    let err = match download_file(api, revision, ST_CONFIG_NAMES[0]).await {
        Ok(st_config_path) => return Ok(st_config_path),
        Err(err) => err,
    };

    // Then try with the rest of the legacy configuration file names
    for name in &ST_CONFIG_NAMES[1..] {
        if let Ok(st_config_path) = download_file(api, revision, name).await {
            return Ok(st_config_path);
        }
    }

    Err(err)
}

#[instrument(skip_all)]
pub async fn download_artifacts(
    api: &HFRepository<RepoTypeModel>,
    revision: Option<&str>,
    pool_config: bool,
) -> Result<PathBuf, ApiError> {
    let start = std::time::Instant::now();
    tracing::info!("Starting download");

    // Try to download `1_Pooling`, only if `--pooling` hasn't been provided, otherwise, the
    // `--pooling` argument will be used instead.
    if pool_config {
        let _ = download_file(api, revision, "1_Pooling/config.json")
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
    let _ = download_st_config_legacy(api, revision).await;

    // Download the actual Sentence Transformers configuration file
    let _ = download_file(api, revision, "config_sentence_transformers.json")
        .await
        .map_err(|err| {
            tracing::warn!("Download failed: {err}");
            err
        });

    download_file(api, revision, "config.json").await?;
    let path = download_file(api, revision, "tokenizer.json").await?;

    tracing::info!("Model artifacts downloaded in {:?}", start.elapsed());

    Ok(path.parent().unwrap().to_path_buf())
}
