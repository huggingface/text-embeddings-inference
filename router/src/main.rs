use anyhow::Result;
use clap::Parser;
use opentelemetry::global;
use text_embeddings_backend::DType;
use veil::Redact;

/// App Configuration
#[derive(Parser, Redact)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The name of the model to load.
    /// Can be a MODEL_ID as listed on <https://hf.co/models> like
    /// `thenlper/gte-base`.
    /// Or it can be a local directory containing the necessary files
    /// as saved by `save_pretrained(...)` methods of transformers
    #[clap(default_value = "thenlper/gte-base", long, env)]
    #[redact(partial)]
    model_id: String,

    /// The actual revision of the model if you're referring to a model
    /// on the hub. You can use a specific commit id or a branch like `refs/pr/2`.
    #[clap(long, env)]
    revision: Option<String>,

    /// Optionally control the number of tokenizer workers used for payload tokenization, validation
    /// and truncation.
    /// Default to the number of CPU cores on the machine.
    #[clap(long, env)]
    tokenization_workers: Option<usize>,

    /// The dtype to be forced upon the model.
    #[clap(long, env, value_enum)]
    dtype: Option<DType>,

    /// Optionally control the pooling method for embedding models.
    ///
    /// If `pooling` is not set, the pooling configuration will be parsed from the
    /// model `1_Pooling/config.json` configuration.
    ///
    /// If `pooling` is set, it will override the model pooling configuration
    #[clap(long, env, value_enum)]
    pooling: Option<text_embeddings_backend::Pool>,

    /// The maximum amount of concurrent requests for this particular deployment.
    /// Having a low limit will refuse clients requests instead of having them
    /// wait for too long and is usually good to handle backpressure correctly.
    #[clap(default_value = "512", long, env)]
    max_concurrent_requests: usize,

    /// **IMPORTANT** This is one critical control to allow maximum usage
    /// of the available hardware.
    ///
    /// This represents the total amount of potential tokens within a batch.
    ///
    /// For `max_batch_tokens=1000`, you could fit `10` queries of `total_tokens=100`
    /// or a single query of `1000` tokens.
    ///
    /// Overall this number should be the largest possible until the model is compute bound.
    /// Since the actual memory overhead depends on the model implementation,
    /// text-embeddings-inference cannot infer this number automatically.
    #[clap(default_value = "16384", long, env)]
    max_batch_tokens: usize,

    /// Optionally control the maximum number of individual requests in a batch
    #[clap(long, env)]
    max_batch_requests: Option<usize>,

    /// Control the maximum number of inputs that a client can send in a single request
    #[clap(default_value = "32", long, env)]
    max_client_batch_size: usize,

    /// Your HuggingFace hub token
    #[clap(long, env)]
    #[redact(partial)]
    hf_api_token: Option<String>,

    /// The IP address to listen on
    #[clap(default_value = "0.0.0.0", long, env)]
    hostname: String,

    /// The port to listen on.
    #[clap(default_value = "3000", long, short, env)]
    port: u16,

    /// The name of the unix socket some text-embeddings-inference backends will use as they
    /// communicate internally with gRPC.
    #[clap(default_value = "/tmp/text-embeddings-inference-server", long, env)]
    uds_path: String,

    /// The location of the huggingface hub cache.
    /// Used to override the location if you want to provide a mounted disk for instance
    #[clap(long, env)]
    huggingface_hub_cache: Option<String>,

    /// Outputs the logs in JSON format (useful for telemetry)
    #[clap(long, env)]
    json_output: bool,

    #[clap(long, env)]
    otlp_endpoint: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Pattern match configuration
    let args: Args = Args::parse();

    // Initialize logging and telemetry
    let global_tracer =
        text_embeddings_router::init_logging(args.otlp_endpoint.as_ref(), args.json_output);

    tracing::info!("{args:?}");

    text_embeddings_router::run(
        args.model_id,
        args.revision,
        args.tokenization_workers,
        args.dtype,
        args.pooling,
        args.max_concurrent_requests,
        args.max_batch_tokens,
        args.max_batch_requests,
        args.max_client_batch_size,
        args.hf_api_token,
        Some(args.hostname),
        args.port,
        Some(args.uds_path),
        args.huggingface_hub_cache,
        args.otlp_endpoint,
    )
    .await?;

    if global_tracer {
        // Shutdown tracer
        global::shutdown_tracer_provider();
    }
    Ok(())
}
