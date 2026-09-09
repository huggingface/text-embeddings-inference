use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::{WithExportConfig, OTEL_EXPORTER_OTLP_HEADERS};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Sampler;
use opentelemetry_sdk::{trace, Resource};
use std::collections::HashMap;
use std::str::FromStr;
use text_embeddings_backend::OtlpProtocol;
use tonic::metadata::{MetadataKey, MetadataMap};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Parse the standard `OTEL_EXPORTER_OTLP_HEADERS` environment variable
/// (comma-separated `key=value` pairs)
fn parse_otlp_headers() -> HashMap<String, String> {
    let mut headers = HashMap::new();
    if let Ok(raw) = std::env::var(OTEL_EXPORTER_OTLP_HEADERS) {
        for pair in raw.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            match pair.split_once('=') {
                Some((key, value)) => {
                    headers.insert(key.trim().to_string(), value.trim().to_string());
                }
                None => {
                    tracing::warn!(
                        "Ignoring malformed entry in {OTEL_EXPORTER_OTLP_HEADERS}: {pair:?}"
                    );
                }
            }
        }
    }
    headers
}

fn build_metadata_map(headers: &HashMap<String, String>) -> MetadataMap {
    let mut map = MetadataMap::new();
    for (key, value) in headers {
        match (MetadataKey::from_str(key), value.parse()) {
            (Ok(key), Ok(value)) => {
                map.insert(key, value);
            }
            _ => tracing::warn!("Skipping invalid OTLP header entry: {key}"),
        }
    }
    map
}

#[cfg(feature = "http")]
pub mod http {
    use axum::{extract::Request, middleware::Next, response::Response};
    use opentelemetry::trace::{SpanContext, TraceContextExt};
    use opentelemetry::trace::{SpanId, TraceFlags, TraceId};
    use opentelemetry::Context;
    struct TraceParent {
        #[allow(dead_code)]
        version: u8,
        trace_id: TraceId,
        parent_id: SpanId,
        trace_flags: TraceFlags,
    }

    fn parse_traceparent(header_value: &str) -> Option<TraceParent> {
        let parts: Vec<&str> = header_value.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        let version = u8::from_str_radix(parts[0], 16).ok()?;
        if version == 0xff {
            return None;
        }

        let trace_id = TraceId::from_hex(parts[1]).ok()?;
        let parent_id = SpanId::from_hex(parts[2]).ok()?;
        let trace_flags = u8::from_str_radix(parts[3], 16).ok()?;

        Some(TraceParent {
            version,
            trace_id,
            parent_id,
            trace_flags: TraceFlags::new(trace_flags),
        })
    }

    pub async fn trace_context_middleware(mut request: Request, next: Next) -> Response {
        let context = request
            .headers()
            .get("traceparent")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_traceparent)
            .map(|traceparent| {
                Context::new().with_remote_span_context(SpanContext::new(
                    traceparent.trace_id,
                    traceparent.parent_id,
                    traceparent.trace_flags,
                    true,
                    Default::default(),
                ))
            });

        request.extensions_mut().insert(context);

        next.run(request).await
    }
}

/// Init logging using env variables LOG_LEVEL and LOG_FORMAT:
///     - otlp_endpoint is an optional URL to an Open Telemetry collector
///     - LOG_LEVEL may be TRACE, DEBUG, INFO, WARN or ERROR (default to INFO)
pub fn init_logging(
    otlp_endpoint: Option<&String>,
    otlp_service_name: String,
    otlp_protocol: OtlpProtocol,
    json_output: bool,
    disable_spans: bool,
) -> bool {
    let mut layers = Vec::new();

    // STDOUT/STDERR layer
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true);

    let fmt_layer = match json_output {
        true => fmt_layer
            .json()
            .flatten_event(true)
            .with_current_span(!disable_spans)
            .with_span_list(!disable_spans)
            .boxed(),
        false => fmt_layer.boxed(),
    };
    layers.push(fmt_layer);

    // OpenTelemetry tracing layer
    let mut global_tracer = false;
    if let Some(otlp_endpoint) = otlp_endpoint {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let headers = parse_otlp_headers();

        let trace_config = trace::config()
            .with_resource(Resource::new(vec![KeyValue::new(
                "service.name",
                otlp_service_name,
            )]))
            .with_sampler(Sampler::AlwaysOn);

        let tracer = match otlp_protocol {
            OtlpProtocol::Grpc => {
                let mut exporter = opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp_endpoint);
                if !headers.is_empty() {
                    exporter = exporter.with_metadata(build_metadata_map(&headers));
                }
                opentelemetry_otlp::new_pipeline()
                    .tracing()
                    .with_exporter(exporter)
                    .with_trace_config(trace_config)
                    .install_batch(opentelemetry_sdk::runtime::Tokio)
            }
            OtlpProtocol::HttpProto => {
                let exporter = opentelemetry_otlp::new_exporter()
                    .http()
                    .with_endpoint(otlp_endpoint)
                    .with_headers(headers);
                opentelemetry_otlp::new_pipeline()
                    .tracing()
                    .with_exporter(exporter)
                    .with_trace_config(trace_config)
                    .install_batch(opentelemetry_sdk::runtime::Tokio)
            }
        };

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            init_tracing_opentelemetry::init_propagator().unwrap();
            global_tracer = true;
        };
    } else if std::env::var(OTEL_EXPORTER_OTLP_HEADERS).is_ok() {
        tracing::warn!("{OTEL_EXPORTER_OTLP_HEADERS} is set but --otlp-endpoint is not; export headers will be ignored");
    }

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();
    global_tracer
}
