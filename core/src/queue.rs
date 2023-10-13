use crate::infer::InferResponse;
use crate::tokenization::Encoding;
use std::cmp::max;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use text_embeddings_backend::{BackendError, Batch};
use tokio::sync::oneshot;
use tracing::{info_span, instrument, Span};

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
    /// Temporary span used as a guard when logging inference, wait times...
    pub temp_span: Option<Span>,
    /// Tokenization duration
    pub tokenization: Duration,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,
}

/// Request Queue
#[derive(Debug, Clone)]
pub struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: flume::Sender<QueueCommand>,
}

impl Queue {
    pub fn new(max_batch_tokens: usize, max_batch_requests: Option<usize>) -> Self {
        // Create channels
        let (queue_sender, queue_receiver) = flume::unbounded();

        // Launch background queue task
        tokio::spawn(queue_task(
            max_batch_tokens,
            max_batch_requests,
            queue_receiver,
        ));

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
async fn queue_task(
    max_batch_tokens: usize,
    max_batch_requests: Option<usize>,
    queue_receiver: flume::Receiver<QueueCommand>,
) {
    let capacity = max_batch_requests.unwrap_or(512);

    let mut entries: VecDeque<Entry> = VecDeque::with_capacity(512);

    while let Ok(cmd) = queue_receiver.recv_async().await {
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

                let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty, tokens = tracing::field::Empty);
                next_batch_span.follows_from(Span::current());

                let mut metadata = Vec::with_capacity(capacity);

                let mut input_ids = Vec::with_capacity(max_batch_tokens);
                let mut token_type_ids = Vec::with_capacity(max_batch_tokens);
                let mut position_ids = Vec::with_capacity(max_batch_tokens);
                let mut cu_seq_lengths = Vec::with_capacity(capacity);
                cu_seq_lengths.push(0);

                let mut current_tokens = 0;
                let mut max_length = 0;

                while let Some(mut entry) = entries.pop_front() {
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
                    current_tokens += entry_tokens;

                    // Create a new span to link the batch back to this entry
                    let entry_batch_span = info_span!(parent: &entry.metadata.span, "infer");
                    // Add relationships
                    next_batch_span.follows_from(&entry_batch_span);
                    entry_batch_span.follows_from(&next_batch_span);

                    entry.metadata.batch_time = Some(Instant::now());
                    entry.metadata.temp_span = Some(entry_batch_span);

                    metadata.push(entry.metadata);
                    input_ids.extend(entry.encoding.input_ids);
                    token_type_ids.extend(entry.encoding.token_type_ids);
                    position_ids.extend(entry.encoding.position_ids);
                    cu_seq_lengths.push(current_tokens as u32);

                    if Some(metadata.len()) == max_batch_requests {
                        break;
                    }
                }

                let next_batch = if metadata.is_empty() {
                    None
                } else {
                    next_batch_span.record("batch_size", metadata.len() as u32);
                    next_batch_span.record("tokens", current_tokens as u32);

                    metrics::histogram!("te_batch_next_size", metadata.len() as f64);
                    metrics::histogram!("te_batch_next_tokens", current_tokens as f64);

                    Some((
                        metadata,
                        Batch {
                            input_ids,
                            token_type_ids,
                            position_ids,
                            cumulative_seq_lengths: cu_seq_lengths,
                            max_length,
                        },
                        next_batch_span,
                    ))
                };

                let _ = response_sender.send(next_batch);
                metrics::gauge!("te_queue_size", entries.len() as f64);
            }
        }
    }
}

type NextBatch = (Vec<Metadata>, Batch, Span);

#[derive(Debug)]
enum QueueCommand {
    Append(Box<Entry>, Span),
    NextBatch {
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
}
