mod logging;
mod management;

use backend_grpc_client::Client;
use text_embeddings_backend_core::{Backend, BackendError, Batch, Embedding, ModelType, Pool};
use tokio::runtime::Runtime;

pub struct PythonBackend {
    _backend_process: management::BackendProcess,
    tokio_runtime: Runtime,
    backend_client: Client,
}

impl PythonBackend {
    pub fn new(
        model_path: String,
        dtype: String,
        model_type: ModelType,
        uds_path: String,
        otlp_endpoint: Option<String>,
    ) -> Result<Self, BackendError> {
        let pool = match model_type {
            ModelType::Classifier => {
                return Err(BackendError::Start(
                    "`classifier` model type is not supported".to_string(),
                ))
            }
            ModelType::Embedding(pool) => pool,
        };

        if pool != Pool::Cls {
            return Err(BackendError::Start(format!("{pool:?} is not supported")));
        }

        let backend_process =
            management::BackendProcess::new(model_path, dtype, &uds_path, otlp_endpoint)?;
        let tokio_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| BackendError::Start(format!("Could not start Tokio runtime: {err}")))?;

        let backend_client = tokio_runtime
            .block_on(Client::connect_uds(uds_path))
            .map_err(|err| {
                BackendError::Start(format!("Could not connect to backend process: {err}"))
            })?;

        Ok(Self {
            _backend_process: backend_process,
            tokio_runtime,
            backend_client,
        })
    }
}

impl Backend for PythonBackend {
    fn health(&self) -> Result<(), BackendError> {
        if self
            .tokio_runtime
            .block_on(self.backend_client.clone().health())
            .is_err()
        {
            return Err(BackendError::Unhealthy);
        }
        Ok(())
    }

    fn embed(&self, batch: Batch) -> Result<Vec<Embedding>, BackendError> {
        let results = self
            .tokio_runtime
            .block_on(self.backend_client.clone().embed(
                batch.input_ids,
                batch.token_type_ids,
                batch.position_ids,
                batch.cumulative_seq_lengths,
                batch.max_length,
            ))
            .map_err(|err| BackendError::Inference(err.to_string()))?;
        Ok(results.into_iter().map(|r| r.values).collect())
    }

    fn predict(&self, _batch: Batch) -> Result<Vec<Vec<f32>>, BackendError> {
        Err(BackendError::Inference(
            "`predict` is not implemented".to_string(),
        ))
    }
}
