use metrics_exporter_prometheus::{BuildError, Matcher, PrometheusBuilder};

pub(crate) fn prometheus_builer(max_input_length: usize) -> Result<PrometheusBuilder, BuildError> {
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
    let input_length_buckets: Vec<f64> = (0..100)
        .map(|x| (max_input_length as f64 / 100.0) * (x + 1) as f64)
        .collect();

    // Batch size buckets
    let batch_size_matcher = Matcher::Full(String::from("te_batch_next_size"));
    let batch_size_buckets: Vec<f64> = (0..2048).map(|x| (x + 1) as f64).collect();

    // Batch tokens buckets
    let batch_tokens_matcher = Matcher::Full(String::from("te_batch_next_tokens"));
    let batch_tokens_buckets: Vec<f64> = (0..100_000).map(|x| (x + 1) as f64).collect();

    // Prometheus handler
    PrometheusBuilder::new()
        .set_buckets_for_metric(duration_matcher, &duration_buckets)?
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)?
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)?
        .set_buckets_for_metric(batch_tokens_matcher, &batch_tokens_buckets)
}
