use anyhow::Result;
use clap::Parser;
use opentelemetry::global;
use text_embeddings_backend::DType;
use veil::Redact;

#[cfg(not(target_os = "linux"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// App Configuration
#[derive(Parser, Redact)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The name of the model to load.
    /// Can be a MODEL_ID as listed on <https://hf.co/models> like
    /// `BAAI/bge-large-en-v1.5`.
    /// Or it can be a local directory containing the necessary files
    /// as saved by `save_pretrained(...)` methods of transformers
    #[clap(default_value = "BAAI/bge-large-en-v1.5", long, env)]
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

    /// Automatically truncate inputs that are longer than the maximum supported size
    ///
    /// Unused for gRPC servers
    #[clap(long, env)]
    auto_truncate: bool,

    /// The name of the prompt that should be used by default for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``default_prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    ///
    /// The argument '--default-prompt-name <DEFAULT_PROMPT_NAME>' cannot be used with
    /// '--default-prompt <DEFAULT_PROMPT>`
    #[clap(long, env, conflicts_with = "default_prompt")]
    default_prompt_name: Option<String>,

    /// The prompt that should be used by default for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// For example if ``default_prompt`` is "query: " then the sentence "What is the capital of
    /// France?" will be encoded as "query: What is the capital of France?" because the prompt
    /// text will be prepended before any text to encode.
    ///
    /// The argument '--default-prompt <DEFAULT_PROMPT>' cannot be used with
    /// '--default-prompt-name <DEFAULT_PROMPT_NAME>`
    #[clap(long, env, conflicts_with = "default_prompt_name")]
    default_prompt: Option<String>,

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

    /// Payload size limit in bytes
    ///
    /// Default is 2MB
    #[clap(default_value = "2000000", long, env)]
    payload_limit: usize,

    /// Set an api key for request authorization.
    ///
    /// By default the server responds to every request. With an api key set, the requests must have the Authorization header set with the api key as Bearer token.
    #[clap(long, env)]
    api_key: Option<String>,

    /// Outputs the logs in JSON format (useful for telemetry)
    #[clap(long, env)]
    json_output: bool,

    /// The grpc endpoint for opentelemetry. Telemetry is sent to this endpoint as OTLP over gRPC.
    /// e.g. `http://localhost:4317`
    #[clap(long, env)]
    otlp_endpoint: Option<String>,

    /// The service name for opentelemetry.
    /// e.g. `text-embeddings-inference.server`
    #[clap(default_value = "text-embeddings-inference.server", long, env)]
    otlp_service_name: String,

    /// Unused for gRPC servers
    #[clap(long, env)]
    cors_allow_origin: Option<Vec<String>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Pattern match configuration
    let args: Args = Args::parse();

    // Initialize logging and telemetry
    let global_tracer = text_embeddings_router::init_logging(
        args.otlp_endpoint.as_ref(),
        args.otlp_service_name.clone(),
        args.json_output,
    );

    tracing::info!("{args:?}");

    // Hack to trim pages regularly
    // see: https://www.algolia.com/blog/engineering/when-allocators-are-hoarding-your-precious-memory/
    // and: https://github.com/huggingface/text-embeddings-inference/issues/156
    #[cfg(target_os = "linux")]
    tokio::spawn(async move {
        use tokio::time::Duration;
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            unsafe {
                libc::malloc_trim(0);
            }
        }
    });

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
        args.auto_truncate,
        args.default_prompt,
        args.default_prompt_name,
        args.hf_api_token,
        Some(args.hostname),
        args.port,
        Some(args.uds_path),
        args.huggingface_hub_cache,
        args.payload_limit,
        args.api_key,
        args.otlp_endpoint,
        args.otlp_service_name,
        args.cors_allow_origin,
    )
    .await?;

    if global_tracer {
        // Shutdown tracer
        global::shutdown_tracer_provider();
    }
    Ok(())
}
