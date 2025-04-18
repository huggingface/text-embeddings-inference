use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Sampler;
use opentelemetry_sdk::{trace, Resource};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

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

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp_endpoint),
            )
            .with_trace_config(
                trace::config()
                    .with_resource(Resource::new(vec![KeyValue::new(
                        "service.name",
                        otlp_service_name,
                    )]))
                    .with_sampler(Sampler::AlwaysOn),
            )
            .install_batch(opentelemetry_sdk::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            init_tracing_opentelemetry::init_propagator().unwrap();
            global_tracer = true;
        };
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
