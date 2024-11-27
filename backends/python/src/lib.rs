mod logging;
mod management;

use backend_grpc_client::Client;
use nohash_hasher::BuildNoHashHasher;
use std::collections::HashMap;
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Predictions,
};
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
        otlp_service_name: String,
    ) -> Result<Self, BackendError> {
        let pool = match model_type {
            ModelType::Classifier => {
                return Err(BackendError::Start(
                    "`classifier` model type is not supported".to_string(),
                ))
            }
            ModelType::Embedding(pool) => pool,
        };

        let backend_process = management::BackendProcess::new(
            model_path,
            dtype,
            &uds_path,
            otlp_endpoint,
            otlp_service_name,
            pool,
        )?;
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

    fn is_padded(&self) -> bool {
        false
    }

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        if !batch.raw_indices.is_empty() {
            return Err(BackendError::Inference(
                "raw embeddings are not supported for the Python backend.".to_string(),
            ));
        }
        let batch_size = batch.len();

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
        let pooled_embeddings: Vec<Vec<f32>> = results.into_iter().map(|r| r.values).collect();

        let mut embeddings =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());
        for (i, e) in pooled_embeddings.into_iter().enumerate() {
            embeddings.insert(i, Embedding::Pooled(e));
        }

        Ok(embeddings)
    }

    fn predict(&self, _batch: Batch) -> Result<Predictions, BackendError> {
        Err(BackendError::Inference(
            "`predict` is not implemented".to_string(),
        ))
    }
}
