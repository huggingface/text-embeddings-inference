use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Sampler;
use opentelemetry_sdk::{trace, Resource};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

/// Rate-limited logger for request success messages
/// Prevents logging every single request at high load
#[derive(Clone)]
pub struct RateLimitedLogger {
    /// Maximum number of request logs per interval
    max_logs_per_interval: u64,
    /// Counter for current interval
    counter: Arc<AtomicU64>,
    /// When the current interval started
    interval_start: Arc<Mutex<Instant>>,
    /// Length of each interval
    interval_duration: Duration,
}

impl RateLimitedLogger {
    pub fn new(max_logs_per_second: u64) -> Self {
        Self {
            max_logs_per_interval: max_logs_per_second,
            counter: Arc::new(AtomicU64::new(0)),
            interval_start: Arc::new(Mutex::new(Instant::now())),
            interval_duration: Duration::from_secs(1),
        }
    }

    /// Try to log a success message. Returns true if the log was emitted,
    /// false if it was rate-limited
    pub fn try_log_success(&self) -> bool {
        let now = Instant::now();

        // Check if we need to reset the counter for a new interval
        {
            let mut start = self.interval_start.lock().unwrap();
            if now.duration_since(*start) >= self.interval_duration {
                // Reset for new interval
                self.counter.store(0, Ordering::Relaxed);
                *start = now;
            }
        }

        // Increment counter and check if we should log
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        if count < self.max_logs_per_interval {
            tracing::info!("Success");
            true
        } else {
            false
        }
    }

    /// Get the current count of logged requests in this interval
    pub fn get_count(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }
}

/// Aggregate statistics tracker for request metrics
#[derive(Clone)]
pub struct AggregateLogger {
    /// Total success count
    success_count: Arc<AtomicU64>,
    /// Total error count
    error_count: Arc<AtomicU64>,
    /// Total compute characters processed
    compute_chars: Arc<AtomicU64>,
    /// Total tokens processed
    total_tokens: Arc<AtomicU64>,
    /// When the logger was started
    start_time: Arc<Mutex<Instant>>,
    /// Whether the logger is active
    active: Arc<Mutex<bool>>,
}

impl AggregateLogger {
    pub fn new() -> Self {
        Self {
            success_count: Arc::new(AtomicU64::new(0)),
            error_count: Arc::new(AtomicU64::new(0)),
            compute_chars: Arc::new(AtomicU64::new(0)),
            total_tokens: Arc::new(AtomicU64::new(0)),
            start_time: Arc::new(Mutex::new(Instant::now())),
            active: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the aggregate logger
    pub fn start(&self) {
        let mut active = self.active.lock().unwrap();
        *active = true;
        let mut start_time = self.start_time.lock().unwrap();
        *start_time = Instant::now();
    }

    /// Stop the aggregate logger
    pub fn stop(&self) {
        let mut active = self.active.lock().unwrap();
        *active = false;
    }

    /// Record a successful request
    pub fn record_success(&self, chars: u64, tokens: u64) {
        if *self.active.lock().unwrap() {
            self.success_count.fetch_add(1, Ordering::Relaxed);
            self.compute_chars.fetch_add(chars, Ordering::Relaxed);
            self.total_tokens.fetch_add(tokens, Ordering::Relaxed);
        }
    }

    /// Record an error
    pub fn record_error(&self) {
        if *self.active.lock().unwrap() {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get and reset the current stats
    pub fn get_and_reset_stats(&self) -> AggregateStats {
        let success = self.success_count.swap(0, Ordering::Relaxed);
        let errors = self.error_count.swap(0, Ordering::Relaxed);
        let chars = self.compute_chars.swap(0, Ordering::Relaxed);
        let tokens = self.total_tokens.swap(0, Ordering::Relaxed);

        let elapsed = {
            let mut start_time = self.start_time.lock().unwrap();
            let elapsed = start_time.elapsed();
            *start_time = Instant::now();
            elapsed
        };

        AggregateStats {
            success_count: success,
            error_count: errors,
            compute_chars: chars,
            total_tokens: tokens,
            elapsed,
        }
    }

    pub fn is_active(&self) -> bool {
        *self.active.lock().unwrap()
    }
}

impl Default for AggregateLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate statistics snapshot
pub struct AggregateStats {
    pub success_count: u64,
    pub error_count: u64,
    pub compute_chars: u64,
    pub total_tokens: u64,
    pub elapsed: Duration,
}

impl std::fmt::Display for AggregateStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let secs = self.elapsed.as_secs_f64().max(0.001);
        let req_per_sec = self.success_count as f64 / secs;
        let chars_per_sec = self.compute_chars as f64 / secs;
        let tokens_per_sec = self.total_tokens as f64 / secs;

        write!(
            f,
            "Request aggregate: {} requests ({:.1}/s), {} errors | {} chars ({:.0}/s) | {} tokens ({:.0}/s)",
            self.success_count,
            req_per_sec,
            self.error_count,
            self.compute_chars,
            chars_per_sec,
            self.total_tokens,
            tokens_per_sec
        )
    }
}

/// Init logging using env variables LOG_LEVEL and LOG_FORMAT:
///     - otlp_endpoint is an optional URL to an Open Telemetry collector
///     - LOG_LEVEL may be TRACE, DEBUG, INFO, WARN or ERROR (default to INFO)
/// 
/// Returns: (global_tracer, rate_limited_logger, aggregate_logger)
pub fn init_logging(
    otlp_endpoint: Option<&String>,
    otlp_service_name: String,
    json_output: bool,
    disable_spans: bool,
    log_sample_rate: u64,
    log_aggregate_interval: u64,
) -> (bool, Option<RateLimitedLogger>, Option<AggregateLogger>) {
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

    // Initialize rate-limited logger if sample rate > 0
    let rate_limited_logger = if log_sample_rate > 0 {
        Some(RateLimitedLogger::new(log_sample_rate))
    } else {
        None
    };

    // Initialize aggregate logger if interval > 0
    let aggregate_logger = if log_aggregate_interval > 0 {
        let agg_logger = AggregateLogger::new();
        agg_logger.start();
        Some(agg_logger)
    } else {
        None
    };

    (global_tracer, rate_limited_logger, aggregate_logger)
}

/// Spawn a background task to log aggregate statistics at regular intervals
pub fn spawn_aggregate_logger_task(
    aggregate_logger: AggregateLogger,
    interval_secs: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let interval = Duration::from_secs(interval_secs);
        let mut interval_timer = tokio::time::interval(interval);
        
        // Skip the first tick (waits for the full interval)
        interval_timer.tick().await;
        
        loop {
            interval_timer.tick().await;
            if aggregate_logger.is_active() {
                let stats = aggregate_logger.get_and_reset_stats();
                if stats.success_count > 0 || stats.error_count > 0 {
                    tracing::info!("{}", stats);
                }
            }
        }
    })
}
