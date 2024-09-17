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
        let (queue_sender, queue_receiver) = mpsc::channel(max_concurrent_requests);

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
                let gauge = metrics::gauge!("te_queue_size");
                gauge.set(entries.len() as f64)
            }
        }
    }
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
