use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Sampler;
use opentelemetry_sdk::{trace, Resource};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Configures logging using environment variables and options.
#[derive(Debug)]
struct LoggingConfig {
    /// Optional endpoint for OpenTelemetry collector.
    otlp_endpoint: Option<String>,
    /// Service name for OpenTelemetry.
    service_name: String,
    /// Whether to output logs in JSON format.
    json_output: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            otlp_endpoint: None,
            service_name: "my-service".to_string(),
            json_output: false,
        }
    }
}

/// Init logging using env variables LOG_LEVEL and LOG_FORMAT:
///     - otlp_endpoint is an optional URL to an Open Telemetry collector
///     - LOG_LEVEL may be TRACE, DEBUG, INFO, WARN or ERROR (default to INFO)
pub fn init_logging(config: &LoggingConfig) -> bool {
    let mut layers = Vec::new();

    // STDOUT/STDERR layer
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true);

    let fmt_layer = match config.json_output {
        true => fmt_layer.json().flatten_event(true).boxed(),
        false => fmt_layer.boxed(),
    };
    layers.push(fmt_layer);

    // OpenTelemetry tracing layer
    let mut global_tracer = false;
    if let Some(endpoint) = config.otlp_endpoint.as_ref() {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(endpoint.clone()),
            )
            .with_trace_config(
                trace::config()
                    .with_resource(Resource::new(vec![
                        KeyValue::new("service.name", config.service_name.clone()),
                        KeyValue::new("environment", std::env::var("RUST_ENV").unwrap_or_default()),
                    ]))
                    .with_sampler(Sampler::from_rate(1.0)),
            )
            .install_batch(opentelemetry_sdk::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            init_tracing_opentelemetry::init_propagator().unwrap();
            global_tracer = true;
        };
    }

    // Filter events with LOG_LEVEL
    let env_filter = EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();

    global_tracer
}

/// Sets the log level dynamically.
pub fn set_log_level(level: &str) {
    let mut builder = EnvFilter::builder();
    builder.add_env("RUST_LOG");
    builder.parse_default_env();
    builder.add_directive(format!("{}={}", "RUST_LOG", level));
    tracing_subscriber::EnvFilter::try_from(builder.build()).unwrap().try_init().unwrap();
}

/// Logs a message with the current log level.
pub fn log_message(message: &str) {
    tracing::error!(message);
}
