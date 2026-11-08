<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Prometheus metrics

Text Embeddings Inference (TEI) exports [Prometheus](https://prometheus.io/) metrics that you can use to monitor
traffic, errors, latency and batching. This page lists every metric, where to scrape it and what it measures.

## Request metrics

These metrics are recorded for every HTTP and gRPC route that runs the model.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `te_request_count` | Counter | `method` | Number of requests received. |
| `te_request_success` | Counter | `method` | Number of requests that completed successfully. |
| `te_request_failure` | Counter | `err` | Number of failed requests, by failure reason. |
| `te_request_duration` | Histogram | | End-to-end time of a successful request, in seconds. |
| `te_request_tokenization_duration` | Histogram | | Time spent tokenizing a successful request, in seconds. |
| `te_request_queue_duration` | Histogram | | Time a successful request waited in the queue, in seconds. |
| `te_request_inference_duration` | Histogram | | Time spent running the model for a successful request, in seconds. |
| `te_request_input_length` | Histogram | | Number of tokens in each input, after truncation. |

The `method` label is `single` when the request contains one input and `batch` when it contains a list of inputs.
`/rerank` requests are always counted as `batch`.

For `batch` requests, `te_request_tokenization_duration`, `te_request_queue_duration` and
`te_request_inference_duration` record the average time per input, while `te_request_duration` records the total time
of the request.

The `err` label of `te_request_failure` takes the following values:

| `err` | Cause |
|-------|-------|
| `validation` | The request is invalid, for example the inputs are empty or `dimensions` is out of range. |
| `batch_size` | The request contains more inputs than `--max-client-batch-size`. |
| `tokenization` | An input could not be tokenized, including inputs longer than the model's maximum input length when truncation is disabled. |
| `model_type` | The route does not match the model, for example `/embed` on a classification model. |
| `overloaded` | The server is already handling `--max-concurrent-requests` requests. HTTP clients receive a `429` status code. |
| `inference` | The model failed to process the input. |
| `dropped` | The client disconnected while its request was waiting in the queue. |
| `missing_values` | gRPC only: a streaming rerank request did not return a score for every text. |

Don't compute an error ratio as `te_request_failure / te_request_count`: failures are counted per input and on routes
that don't increment `te_request_count`.

## Embedding metrics

These metrics are recorded by the embedding routes: `/embed`, `/embed_all`, `/embed_sparse`, `/embeddings`,
`/v1/embeddings` and `/similarity`, as well as their gRPC equivalents.

| Metric | Type | Description |
|--------|------|-------------|
| `te_embed_count` | Counter | Number of inputs sent for embedding. |
| `te_embed_success` | Counter | Number of inputs embedded successfully. |
| `te_embed_duration` | Histogram | End-to-end time to embed one input, in seconds. |
| `te_embed_tokenization_duration` | Histogram | Time spent tokenizing one input, in seconds. |
| `te_embed_queue_duration` | Histogram | Time one input waited in the queue, in seconds. |
| `te_embed_inference_duration` | Histogram | Time spent running the model for one input, in seconds. |

Unlike the request metrics, the embedding metrics are recorded **per input**: a request with 8 inputs increments
`te_request_count` by 1 and `te_embed_count` by 8.

## Prediction metrics

These metrics are recorded by the classification and reranking routes: `/predict` and `/rerank`, as well as their
gRPC equivalents.

| Metric | Type | Description |
|--------|------|-------------|
| `te_predict_count` | Counter | Number of inputs sent for prediction. |
| `te_predict_success` | Counter | Number of inputs predicted successfully. |
| `te_predict_duration` | Histogram | End-to-end time to predict one input, in seconds. |
| `te_predict_tokenization_duration` | Histogram | Time spent tokenizing one input, in seconds. |
| `te_predict_queue_duration` | Histogram | Time one input waited in the queue, in seconds. |
| `te_predict_inference_duration` | Histogram | Time spent running the model for one input, in seconds. |

Like the embedding metrics, they are recorded per input. For `/rerank`, every query-text pair counts as one input.

## Queue and batching metrics

| Metric | Type | Description |
|--------|------|-------------|
| `te_queue_size` | Gauge | Number of inputs waiting for inference. An input is counted from the moment it is queued until the batch it belongs to is sent to the model, and is removed if the client disconnects. |
| `te_batch_next_size` | Histogram | Number of inputs in each batch formed by the scheduler. |
| `te_batch_next_tokens` | Histogram | Number of tokens (without padding) in each batch formed by the scheduler, bounded by `--max-batch-tokens`. |

## Histogram buckets

| Metrics | Buckets |
|---------|---------|
| All `*_duration` histograms | 35 buckets growing by a factor of 1.5, from 15 µs to about 14.6 s. |
| `te_request_input_length` | Powers of 2, with extra steps in between from 512 to 32768 (768, 1536, 3072, …), capped at the model's maximum input length. |
| `te_batch_next_size` | Every integer from 1 to 32, then 64, 128, 256, 512, 1024, 2048 and 4096. |
| `te_batch_next_tokens` | Same steps as `te_request_input_length`, up to 1048576. |
