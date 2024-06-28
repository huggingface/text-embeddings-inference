use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use text_embeddings_backend::DType;
use text_embeddings_router::run;
use tokio::time::Instant;

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
        self.is_close(other, 4e-3)
    }
}

async fn check_health(port: u16, timeout: Duration) -> Result<()> {
    let addr = format!("http://0.0.0.0:{port}/health");
    let client = reqwest::ClientBuilder::new()
        .timeout(timeout)
        .build()
        .unwrap();

    let start = Instant::now();
    loop {
        if client.get(&addr).send().await.is_ok() {
            return Ok(());
        }
        if start.elapsed() < timeout {
            tokio::time::sleep(Duration::from_secs(1)).await;
        } else {
            anyhow::bail!("Backend is not healthy");
        }
    }
}

pub async fn start_server(model_id: String, revision: Option<String>, dtype: DType) -> Result<()> {
    let server_task = tokio::spawn({
        run(
            model_id,
            revision,
            Some(1),
            Some(dtype),
            None,
            4,
            1024,
            None,
            32,
            false,
            None,
            None,
            None,
            None,
            8090,
            None,
            None,
            2_000_000,
            None,
            None,
            "text-embeddings-inference.server".to_owned(),
            None,
        )
    });

    tokio::select! {
        err = server_task => err?,
        _ = check_health(8090, Duration::from_secs(60)) => Ok(())
    }?;
    Ok(())
}
