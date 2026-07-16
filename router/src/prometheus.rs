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

    // Input Length buckets: half-steps up to 32k, then powers of 2
    let input_length_matcher = Matcher::Full(String::from("te_request_input_length"));
    let input_length_buckets: Vec<f64> = [
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 768.0, 1024.0, 1536.0, 2048.0,
        3072.0, 4096.0, 6144.0, 8192.0, 12288.0, 16384.0, 24576.0, 32768.0, 65536.0, 131072.0,
        262144.0, 524288.0,
    ]
    .into_iter()
    .filter(|x| (*x as usize) <= max_input_length)
    .collect();

    // Batch size buckets: 1-32 (all integers), then powers of 2
    let batch_size_matcher = Matcher::Full(String::from("te_batch_next_size"));
    let batch_size_buckets: Vec<f64> = (1..=32)
        .map(|x| x as f64)
        .chain((7..13).map(|x| 2.0_f64.powi(x)))
        .collect();

    // Batch tokens buckets: half-steps up to 32k, then powers of 2
    let batch_tokens_matcher = Matcher::Full(String::from("te_batch_next_tokens"));
    let batch_tokens_buckets: Vec<f64> = [
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 768.0, 1024.0, 1536.0, 2048.0,
        3072.0, 4096.0, 6144.0, 8192.0, 12288.0, 16384.0, 24576.0, 32768.0, 65536.0, 131072.0,
        262144.0, 524288.0, 1048576.0,
    ]
    .into_iter()
    .collect();

    // Prometheus handler
    PrometheusBuilder::new()
        .with_http_listener(addr)
        .set_buckets_for_metric(duration_matcher, &duration_buckets)?
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)?
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)?
        .set_buckets_for_metric(batch_tokens_matcher, &batch_tokens_buckets)
}
