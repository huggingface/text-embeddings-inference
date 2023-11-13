use crate::infer::InferResponse;
use crate::tokenization::Encoding;
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
    pub encoding: Encoding,
    /// Entry metadata
    pub metadata: Metadata,
}

/// Entry metadata
#[derive(Debug)]
pub struct Metadata {
    /// InferResponse sender to communicate between the Infer struct and the batching_task
    pub response_tx: oneshot::Sender<Result<InferResponse, BackendError>>,
    /// Span that will live as long as entry
    pub span: Span,
    /// Tokenization duration
    pub tokenization: Duration,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,
    /// Normalize the embeddings
    pub normalize: bool,
}

/// Request Queue
#[derive(Debug, Clone)]
pub struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommand>,
}

impl Queue {
    pub fn new(
        max_batch_tokens: usize,
        max_batch_requests: Option<usize>,
        max_concurrent_requests: usize,
    ) -> Self {
        // Create channels
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        // Launch background queue task
        tokio::task::spawn_blocking(move || {
            queue_blocking_task(
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
            .send(QueueCommand::Append(Box::new(entry), Span::current()))
            .expect("Queue background task dropped the receiver. This is a bug.");
    }

    /// Get the next batch from the queue
    #[instrument(skip(self))]
    pub async fn next_batch(&self) -> Option<NextBatch> {
        let (response_sender, response_receiver) = oneshot::channel();

        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::NextBatch {
                response_sender,
                span: Span::current(),
            })
            .expect("Queue background task dropped the receiver. This is a bug.");
        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.expect(
            "Queue background task dropped the sender without sending a new batch. This is a bug.",
        )
    }
}

// Background task responsible of the queue state
fn queue_blocking_task(
    max_batch_tokens: usize,
    max_batch_requests: Option<usize>,
    max_concurrent_requests: usize,
    mut queue_receiver: mpsc::UnboundedReceiver<QueueCommand>,
) {
    let capacity = max_batch_requests.unwrap_or(max_concurrent_requests);

    let mut entries: VecDeque<Entry> = VecDeque::with_capacity(max_concurrent_requests);

    while let Some(cmd) = queue_receiver.blocking_recv() {
        match cmd {
            QueueCommand::Append(entry, span) => {
                let _span = span.entered();
                entries.push_back(*entry);
                metrics::increment_gauge!("te_queue_size", 1.0);
            }
            QueueCommand::NextBatch {
                response_sender,
                span,
            } => {
                let _span = span.entered();

                let mut input_ids = Vec::with_capacity(max_batch_tokens);
                let mut token_type_ids = Vec::with_capacity(max_batch_tokens);
                let mut position_ids = Vec::with_capacity(max_batch_tokens);

                let mut metadata = Vec::with_capacity(capacity);
                let mut cu_seq_lengths = Vec::with_capacity(capacity);
                cu_seq_lengths.push(0);

                let mut current_tokens = 0;
                let mut max_length = 0;

                while let Some(entry) = entries.pop_front() {
                    // Filter entries where the response receiver was dropped (== entries where the request
                    // was dropped by the client)
                    if entry.metadata.response_tx.is_closed() {
                        metrics::increment_counter!("te_request_failure", "err" => "dropped");
                        continue;
                    }

                    let entry_tokens = entry.encoding.input_ids.len();

                    if current_tokens + entry_tokens > max_batch_tokens {
                        entries.push_front(entry);
                        break;
                    }

                    max_length = max(max_length, entry_tokens as u32);

                    input_ids.extend(entry.encoding.input_ids);
                    token_type_ids.extend(entry.encoding.token_type_ids);
                    position_ids.extend(entry.encoding.position_ids);

                    current_tokens += entry_tokens;
                    metadata.push(entry.metadata);
                    cu_seq_lengths.push(current_tokens as u32);

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
                        },
                    ))
                };

                let _ = response_sender.send(next_batch);

                metrics::histogram!("te_batch_next_size", batch_size as f64);
                metrics::histogram!("te_batch_next_tokens", current_tokens as f64);
                metrics::gauge!("te_queue_size", entries.len() as f64);
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
