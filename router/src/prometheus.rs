use std::net::SocketAddr;

use metrics_exporter_prometheus::{BuildError, Matcher, PrometheusBuilder};

pub(crate) fn prometheus_builer(
    addr: SocketAddr,
    port: u16,
    max_input_length: usize,
) -> Result<PrometheusBuilder, BuildError> {
    let mut addr = addr;
    addr.set_port(port);

    // Duration buckets
    let duration_matcher = Matcher::Suffix(String::from("duration"));
    let n_duration_buckets = 35;
    let mut duration_buckets = Vec::with_capacity(n_duration_buckets);
    // Minimum duration in seconds
    let mut value = 0.00001;
    for _ in 0..n_duration_buckets {
        // geometric sequence
        value *= 1.5;
        duration_buckets.push(value);
    }

    // Input Length buckets
    let input_length_matcher = Matcher::Full(String::from("te_request_input_length"));
    let input_length_buckets: Vec<f64> = (0..20)
        .map(|x| 2.0_f64.powi(x))
        .filter(|x| (*x as usize) <= max_input_length)
        .collect();

    // Batch size buckets
    let batch_size_matcher = Matcher::Full(String::from("te_batch_next_size"));
    let batch_size_buckets: Vec<f64> = (0..13).map(|x| 2.0_f64.powi(x)).collect();

    // Batch tokens buckets
    let batch_tokens_matcher = Matcher::Full(String::from("te_batch_next_tokens"));
    let batch_tokens_buckets: Vec<f64> = (0..21).map(|x| 2.0_f64.powi(x)).collect();

    // Compression ratio buckets (for values between 0 and 1)
    let compression_ratio_matcher = Matcher::Full(String::from("te_radix_mlp_compression_ratio"));
    let compression_ratio_buckets: Vec<f64> = vec![
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0
    ];

    // Prometheus handler
    PrometheusBuilder::new()
        .with_http_listener(addr)
        .set_buckets_for_metric(duration_matcher, &duration_buckets)?
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)?
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)?
        .set_buckets_for_metric(batch_tokens_matcher, &batch_tokens_buckets)?
        .set_buckets_for_metric(compression_ratio_matcher, &compression_ratio_buckets)
}
