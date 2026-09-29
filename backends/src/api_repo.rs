use std::path::PathBuf;

use hf_hub::repository::ModelInfo;
use hf_hub::{HFError, HFRepository, RepoTypeModel};

/// A model repository pinned to a revision
#[derive(Clone, Debug)]
pub struct ApiRepo {
    repo: HFRepository<RepoTypeModel>,
    revision: Option<String>,
}

impl ApiRepo {
    pub fn new(repo: HFRepository<RepoTypeModel>, revision: Option<String>) -> Self {
        Self { repo, revision }
    }

    pub async fn get(&self, filename: &str) -> Result<PathBuf, HFError> {
        self.repo
            .download_file()
            .filename(filename)
            .maybe_revision(self.revision.clone())
            .send()
            .await
    }

    pub async fn info(&self) -> Result<ModelInfo, HFError> {
        self.repo
            .info()
            .maybe_revision(self.revision.clone())
            .send()
            .await
    }
}
