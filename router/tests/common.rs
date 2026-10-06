use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use text_embeddings_backend::DType;
use text_embeddings_router::run;
use tokio::time::Instant;

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug)]
pub struct Score(f32);

#[allow(dead_code)]
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

#[allow(dead_code)]
pub async fn start_server(model_id: String, revision: Option<String>, dtype: DType) -> Result<()> {
    start_server_with_ports(model_id, revision, dtype, 8090, 9000).await
}

pub async fn start_server_with_ports(
    model_id: String,
    revision: Option<String>,
    dtype: DType,
    port: u16,
    prometheus_port: u16,
) -> Result<()> {
    let server_task = tokio::spawn({
        run(
            model_id.clone(),
            revision,
            Some(1),
            Some(dtype),
            model_id,
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
            None,
            port,
            None,
            None,
            2_000_000,
            None,
            None,
            "text-embeddings-inference.server".to_owned(),
            prometheus_port,
            None,
        )
    });

    tokio::select! {
        err = server_task => err?,
        _ = check_health(port, Duration::from_secs(60)) => Ok(())
    }?;
    Ok(())
}
