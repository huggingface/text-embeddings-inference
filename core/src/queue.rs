use crate::infer::InferResult;
use crate::tokenization::ValidEncoding;
use std::cmp::max;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use text_embeddings_backend::{BackendError, Batch};
use tokio::sync::{mpsc, oneshot};
use tracing::{instrument, Span};

/// Queue entry
#[derive(Debug)]
pub struct Entry {
    /// Payload
    pub encoding: ValidEncoding,
    /// Entry metadata
    pub metadata: Metadata,
}

/// Entry metadata
#[derive(Debug)]
pub struct Metadata {
    /// InferResponse sender to communicate between the Infer struct and the batching_task
    pub(crate) response_tx: oneshot::Sender<Result<InferResult, BackendError>>,
    /// Tokenization duration
    pub(crate) tokenization: Duration,
    /// Instant when this entry was queued
    pub(crate) queue_time: Instant,
    /// Number of tokens in the prompt
    pub(crate) prompt_tokens: usize,
    /// Pooled embedding
    pub(crate) pooling: bool,
}

/// Request Queue
#[derive(Debug, Clone)]
pub struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::Sender<QueueCommand>,
}

impl Queue {
    pub fn new(
        padded_model: bool,
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        max_concurrent_requests: usize,
    ) -> Self {
        // Create channels
        // The queue rarely fails to send the QueueCommand due to a lack of buffer size.
        // So, naively increasing the buffer size to twice than `max_concurrent_requests` to prevent the failure temporarily
        let (queue_sender, queue_receiver) = mpsc::channel(2 * max_concurrent_requests);

        // Launch background queue task
        std::thread::spawn(move || {
            queue_blocking_task(
                padded_model,
                max_batch_tokens,
                max_batch_requests,
                max_concurrent_requests,
                queue_receiver,
            )
        });

        Self { queue_sender }
    }

    /// Append an entry to the queue
    #[instrument(skip_all)]
    pub fn append(&self, entry: Entry) {
        // Send append command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .try_send(QueueCommand::Append(Box::new(entry), Span::current()))
            .expect("Queue background task dropped the receiver or the receiver is too behind. This is a bug.");
    }

    /// Get the next batch from the queue
    #[instrument(skip(self))]
    pub async fn next_batch(&self) -> Option<NextBatch> {
        let (response_sender, response_receiver) = oneshot::channel();

        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .try_send(QueueCommand::NextBatch {
                response_sender,
                span: Span::current(),
            })
            .expect("Queue background task dropped the receiver or the receiver is too behind. This is a bug.");
        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.expect(
            "Queue background task dropped the sender without sending a new batch. This is a bug.",
        )
    }
}

// Background task responsible of the queue state
fn queue_blocking_task(
    padded_model: bool,
    max_batch_tokens: usize,
    max_batch_requests: Option<usize>,
    max_concurrent_requests: usize,
    mut queue_receiver: mpsc::Receiver<QueueCommand>,
) {
    let capacity = max_batch_requests.unwrap_or(max_concurrent_requests);

    let mut entries: VecDeque<Entry> = VecDeque::with_capacity(max_concurrent_requests);

    while let Some(cmd) = queue_receiver.blocking_recv() {
        match cmd {
            QueueCommand::Append(entry, span) => {
                let _span = span.entered();
                entries.push_back(*entry);
                // The entry is counted as queued until it is either dropped by the client or
                // handed over to the backend as part of a batch (see `record_batch_dispatched`)
                let gauge = metrics::gauge!("te_queue_size");
                gauge.increment(1.0);
            }
            QueueCommand::NextBatch {
                response_sender,
                span,
            } => {
                let _span = span.entered();

                let mut input_ids = Vec::with_capacity(max_batch_tokens);
                let mut token_type_ids = Vec::with_capacity(max_batch_tokens);
                let mut position_ids = Vec::with_capacity(max_batch_tokens);

                let mut pooled_indices = Vec::with_capacity(capacity);
                let mut raw_indices = Vec::with_capacity(capacity);
                let mut metadata = Vec::with_capacity(capacity);
                let mut cu_seq_lengths = Vec::with_capacity(capacity);
                cu_seq_lengths.push(0);

                let mut current_tokens = 0;
                let mut max_length = 0;

                let mut entry_index = 0;

                while let Some(entry) = entries.pop_front() {
                    // Filter entries where the response receiver was dropped (== entries where the request
                    // was dropped by the client)
                    if entry.metadata.response_tx.is_closed() {
                        let counter = metrics::counter!("te_request_failure", "err" => "dropped");
                        counter.increment(1);
                        let gauge = metrics::gauge!("te_queue_size");
                        gauge.decrement(1.0);
                        continue;
                    }

                    let entry_tokens = entry.encoding.input_ids.len();

                    let total_tokens = if padded_model {
                        (max(max_length, entry_tokens as u32) * (metadata.len() + 1) as u32)
                            as usize
                    } else {
                        current_tokens + entry_tokens
                    };

                    if total_tokens > max_batch_tokens {
                        entries.push_front(entry);
                        break;
                    }

                    match entry.metadata.pooling {
                        true => pooled_indices.push(entry_index),
                        false => raw_indices.push(entry_index),
                    }

                    max_length = max(max_length, entry_tokens as u32);

                    input_ids.extend(entry.encoding.input_ids);
                    token_type_ids.extend(entry.encoding.token_type_ids);
                    position_ids.extend(entry.encoding.position_ids);

                    current_tokens += entry_tokens;
                    metadata.push(entry.metadata);
                    cu_seq_lengths.push(current_tokens as u32);

                    entry_index += 1;

                    if Some(metadata.len()) == max_batch_requests {
                        break;
                    }
                }

                let batch_size = metadata.len();
                let next_batch = if metadata.is_empty() {
                    None
                } else {
                    Some((
                        metadata,
                        Batch {
                            input_ids,
                            token_type_ids,
                            position_ids,
                            cumulative_seq_lengths: cu_seq_lengths,
                            max_length,
                            pooled_indices,
                            raw_indices,
                        },
                    ))
                };

                let _ = response_sender.send(next_batch);

                let histogram = metrics::histogram!("te_batch_next_size");
                histogram.record(batch_size as f64);
                let histogram = metrics::histogram!("te_batch_next_tokens");
                histogram.record(current_tokens as f64);
            }
        }
    }
}

/// Record that a batch pulled from the queue was handed over to the backend for inference.
///
/// `te_queue_size` tracks the number of requests waiting for inference: a request is counted from
/// the moment it is appended to the queue until the batch it belongs to is dispatched to the
/// backend. Batches are pulled from the queue as soon as one can be prefetched, so the requests of
/// a prefetched batch are still waiting and must not be removed from the gauge before this point.
pub(crate) fn record_batch_dispatched(batch_size: usize) {
    let gauge = metrics::gauge!("te_queue_size");
    gauge.decrement(batch_size as f64);
}

pub type NextBatch = (Vec<Metadata>, Batch);

#[derive(Debug)]
enum QueueCommand {
    Append(Box<Entry>, Span),
    NextBatch {
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use tokio::runtime::Runtime;
    use tokio::sync::oneshot::Receiver;

    /// Install a debugging recorder as the global metrics recorder (once per test binary) and
    /// return its snapshotter
    fn snapshotter() -> &'static Snapshotter {
        static SNAPSHOTTER: OnceLock<Snapshotter> = OnceLock::new();
        SNAPSHOTTER.get_or_init(|| {
            let recorder = DebuggingRecorder::new();
            let snapshotter = recorder.snapshotter();
            recorder
                .install()
                .expect("failed to install the debugging metrics recorder");
            snapshotter
        })
    }

    /// `te_queue_size` is global to the test binary: tests reading it must not run concurrently
    fn queue_size_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Current value of the `te_queue_size` gauge
    fn queue_size() -> f64 {
        snapshotter()
            .snapshot()
            .into_vec()
            .into_iter()
            .find(|(key, _, _, _)| key.key().name() == "te_queue_size")
            .map(|(_, _, _, value)| match value {
                DebugValue::Gauge(value) => value.0,
                other => panic!("te_queue_size is not a gauge: {other:?}"),
            })
            .unwrap_or(0.0)
    }

    /// Appends are processed by the background queue task: wait for the gauge to reach `expected`
    fn wait_for_queue_size(expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let value = queue_size();
            if value == expected as f64 {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "te_queue_size is {value}, expected {expected}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn append(queue: &Queue, n_tokens: usize) -> Receiver<Result<InferResult, BackendError>> {
        let (response_tx, response_rx) = oneshot::channel();
        queue.append(Entry {
            encoding: ValidEncoding {
                input_ids: vec![1; n_tokens],
                token_type_ids: vec![0; n_tokens],
                position_ids: (0..n_tokens as u32).collect(),
            },
            metadata: Metadata {
                response_tx,
                tokenization: Duration::ZERO,
                queue_time: Instant::now(),
                prompt_tokens: n_tokens,
                pooling: true,
            },
        });
        response_rx
    }

    #[test]
    fn test_queue_size_counts_batched_requests_until_dispatched() {
        let _lock = queue_size_lock();
        let runtime = Runtime::new().unwrap();
        let queue = Queue::new(false, 1024, None, 16);

        let _receivers: Vec<_> = (0..4).map(|_| append(&queue, 8)).collect();
        wait_for_queue_size(4);

        // The batching task pulls every queued request into a single batch as soon as it can
        // prefetch one, but the requests are still waiting for the backend
        let (metadata, batch) = runtime.block_on(queue.next_batch()).unwrap();
        assert_eq!(metadata.len(), 4);
        assert_eq!(batch.input_ids.len(), 32);
        // The queue task processes commands in order: once this second `next_batch` returns, the
        // metric updates of the first one have been applied
        assert!(runtime.block_on(queue.next_batch()).is_none());
        assert_eq!(queue_size(), 4.0);

        // The requests leave the queue once the batch is handed over to the backend
        record_batch_dispatched(metadata.len());
        assert_eq!(queue_size(), 0.0);
    }

    #[test]
    fn test_queue_size_accounts_for_partial_batches_and_dropped_requests() {
        let _lock = queue_size_lock();
        let runtime = Runtime::new().unwrap();
        // Room for two 8 tokens requests per batch
        let queue = Queue::new(false, 16, None, 16);

        let _first = append(&queue, 8);
        let dropped = append(&queue, 8);
        let _second = append(&queue, 8);
        let _third = append(&queue, 8);
        wait_for_queue_size(4);

        // The client of the second request went away before it could be batched
        drop(dropped);

        let (metadata, _) = runtime.block_on(queue.next_batch()).unwrap();
        assert_eq!(metadata.len(), 2);
        record_batch_dispatched(metadata.len());

        // Only the third request is left in the queue
        let (metadata, _) = runtime.block_on(queue.next_batch()).unwrap();
        assert_eq!(metadata.len(), 1);
        assert!(runtime.block_on(queue.next_batch()).is_none());
        assert_eq!(queue_size(), 1.0);

        record_batch_dispatched(metadata.len());
        assert_eq!(queue_size(), 0.0);
    }
}
