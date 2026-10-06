use opentelemetry::trace::TraceContextExt;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Sampler;
use opentelemetry_sdk::{trace, Resource};
use std::fmt;
use tracing::{Event, Subscriber};
use tracing_opentelemetry::OtelData;
use tracing_subscriber::fmt::format::{Format, Json, Writer};
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// JSON event formatter that adds the OpenTelemetry `trace_id` and `span_id` of the
/// current span to each record, so log lines can be matched with their exported trace.
/// Records outside of a span, or without the OpenTelemetry layer, are left unchanged.
struct JsonWithTraceIds(Format<Json>);

impl<S, N> FormatEvent<S, N> for JsonWithTraceIds
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let ids = ctx.event_scope().and_then(|mut scope| {
            let span = scope.next()?;
            let extensions = span.extensions();
            let otel = extensions.get::<OtelData>()?;
            // Same rule the exporter uses: a parent context (e.g. from a `traceparent`
            // header) owns the trace id, otherwise the span started a new trace
            let trace_id = if otel.parent_cx.has_active_span() {
                otel.parent_cx.span().span_context().trace_id()
            } else {
                otel.builder.trace_id?
            };
            Some((trace_id, otel.builder.span_id?))
        });

        let Some((trace_id, span_id)) = ids else {
            return self.0.format_event(ctx, writer, event);
        };

        let mut line = String::new();
        self.0.format_event(ctx, Writer::new(&mut line), event)?;
        // The JSON formatter writes one object per line, so the ids go before the closing brace
        match line.trim_end().strip_suffix('}') {
            Some(record) => writeln!(
                writer,
                "{record},\"trace_id\":\"{trace_id}\",\"span_id\":\"{span_id}\"}}"
            ),
            None => writer.write_str(&line),
        }
    }
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
            .map_event_format(JsonWithTraceIds)
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

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::{SpanContext, SpanId, TraceFlags, TraceId, TracerProvider as _};
    use opentelemetry::Context;
    use std::io;
    use std::sync::{Arc, Mutex};
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl io::Write for Buffer {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// Runs `f` with the JSON and OpenTelemetry layers and returns the parsed log records
    fn capture_logs(f: impl FnOnce()) -> Vec<serde_json::Value> {
        let buffer = Buffer::default();
        let writer = buffer.clone();
        // The tracer only keeps a weak reference to its provider, so the provider must
        // live until the end of the test or every generated id is zero
        let provider = opentelemetry_sdk::trace::TracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .flatten_event(true)
                    .map_event_format(JsonWithTraceIds)
                    .with_writer(move || writer.clone()),
            )
            .with(tracing_opentelemetry::layer().with_tracer(tracer));

        tracing::subscriber::with_default(subscriber, f);

        let output = String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap();
        output
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn json_logs_have_ids_of_current_span() {
        let mut expected = None;
        let logs = capture_logs(|| {
            let span = tracing::info_span!("request");
            let _guard = span.enter();
            tracing::info!("inside span");
            let cx = span.context();
            let span_context = cx.span().span_context().clone();
            expected = Some((
                span_context.trace_id().to_string(),
                span_context.span_id().to_string(),
            ));
        });
        let (trace_id, span_id) = expected.unwrap();
        assert_ne!(trace_id, TraceId::INVALID.to_string());
        assert_ne!(span_id, SpanId::INVALID.to_string());

        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0]["message"], "inside span");
        assert_eq!(logs[0]["trace_id"], trace_id.as_str());
        assert_eq!(logs[0]["span_id"], span_id.as_str());
    }

    #[test]
    fn json_logs_use_trace_id_of_remote_parent() {
        let remote = SpanContext::new(
            TraceId::from_hex("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
            SpanId::from_hex("00f067aa0ba902b7").unwrap(),
            TraceFlags::SAMPLED,
            true,
            Default::default(),
        );
        let logs = capture_logs(|| {
            let span = tracing::info_span!("request");
            span.set_parent(Context::new().with_remote_span_context(remote));
            let _guard = span.enter();
            tracing::info!("inside span");
        });

        assert_eq!(logs[0]["trace_id"], "4bf92f3577b34da6a3ce929d0e0e4736");
        let span_id = logs[0]["span_id"].as_str().unwrap();
        assert_ne!(span_id, "00f067aa0ba902b7");
        assert_ne!(span_id, SpanId::INVALID.to_string());
    }

    #[test]
    fn json_logs_outside_span_have_no_ids() {
        let logs = capture_logs(|| tracing::info!("no span"));

        assert_eq!(logs[0]["message"], "no span");
        assert!(logs[0].get("trace_id").is_none());
        assert!(logs[0].get("span_id").is_none());
    }
}
