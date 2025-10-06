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

    // Listwise reranker metrics buckets
    // Duration buckets for listwise block processing (milliseconds)
    let lbnl_duration_matcher = Matcher::Full(String::from("tei_lbnl_ms_per_group"));
    let lbnl_duration_buckets: Vec<f64> = vec![
        1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
    ];

    // Sequence token buckets (typically 1k-32k range for Qwen3)
    let lbnl_tokens_matcher = Matcher::Full(String::from("tei_lbnl_seq_tokens"));
    let lbnl_tokens_buckets: Vec<f64> = vec![
        512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0,
    ];

    // Group size buckets (max 125 docs per block)
    let lbnl_group_matcher = Matcher::Full(String::from("tei_lbnl_group_size"));
    let lbnl_group_buckets: Vec<f64> = vec![1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 125.0];

    // Prometheus handler
    PrometheusBuilder::new()
        .with_http_listener(addr)
        .set_buckets_for_metric(duration_matcher, &duration_buckets)?
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)?
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)?
        .set_buckets_for_metric(batch_tokens_matcher, &batch_tokens_buckets)?
        .set_buckets_for_metric(lbnl_duration_matcher, &lbnl_duration_buckets)?
        .set_buckets_for_metric(lbnl_tokens_matcher, &lbnl_tokens_buckets)?
        .set_buckets_for_metric(lbnl_group_matcher, &lbnl_group_buckets)
}
