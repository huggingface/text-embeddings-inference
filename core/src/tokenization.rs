use crate::TextEmbeddingsError;
/// Payload tokenization logic
use tokenizers::tokenizer::Tokenizer;
use tokenizers::TruncationDirection;
use tokio::sync::oneshot;
use tracing::{instrument, Span};

/// Validation
#[derive(Debug, Clone)]
pub struct Tokenization {
    /// Channel to communicate with the background tokenization task
    sender: flume::Sender<TokenizerRequest>,
}

impl Tokenization {
    pub fn new(
        workers: usize,
        tokenizer: Tokenizer,
        max_input_length: usize,
        position_offset: usize,
    ) -> Self {
        // Create channel
        let (sender, receiver) = flume::unbounded();

        // Create workers
        for _ in 0..workers {
            let tokenizer_clone = tokenizer.clone();
            let receiver_clone = receiver.clone();

            // Spawn worker
            tokio::task::spawn_blocking(move || {
                tokenizer_worker(
                    tokenizer_clone,
                    max_input_length,
                    position_offset,
                    receiver_clone,
                )
            });
        }

        Self { sender }
    }

    #[instrument(skip_all)]
    pub async fn encode(
        &self,
        inputs: String,
        truncate: bool,
    ) -> Result<Encoding, TextEmbeddingsError> {
        // Check if inputs is empty
        if inputs.is_empty() {
            return Err(TextEmbeddingsError::Validation(
                "`inputs` cannot be empty".to_string(),
            ));
        }

        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send request to the background validation task
        // Unwrap is safe here
        self.sender
            .send((inputs, truncate, response_sender, Span::current()))
            .expect("Tokenization background task dropped the receiver. This is a bug.");

        // Await on response channel
        // Unwrap is safe here
        let payload = response_receiver.await.expect("Tokenization background task dropped the sender without sending a response. This is a bug.")?;

        metrics::histogram!("te_request_input_length", payload.input_ids.len() as f64);

        Ok(payload)
    }
}

/// Start tokenization workers
fn tokenizer_worker(
    tokenizer: Tokenizer,
    max_input_length: usize,
    position_offset: usize,
    receiver: flume::Receiver<TokenizerRequest>,
) {
    // Loop over requests
    while let Ok((inputs, truncate, response_tx, parent_span)) = receiver.recv() {
        parent_span.in_scope(|| {
            if !response_tx.is_closed() {
                // It's possible that the user dropped its request resulting in a send error.
                // We just discard the error
                let _ = response_tx.send(encode_input(
                    inputs,
                    truncate,
                    max_input_length,
                    position_offset,
                    &tokenizer,
                ));
            }
        })
    }
}

/// Get input length and optionally truncate it
fn encode_input(
    inputs: String,
    truncate: bool,
    max_input_length: usize,
    position_offset: usize,
    tokenizer: &Tokenizer,
) -> Result<Encoding, TextEmbeddingsError> {
    // Get the number of tokens in the input
    let mut encoding = tokenizer.encode(inputs.clone(), true)?;

    let mut seq_len = encoding.len();

    if seq_len > max_input_length {
        if truncate {
            encoding.truncate(max_input_length, 0, TruncationDirection::Right);
            seq_len = max_input_length;
        } else {
            return Err(TextEmbeddingsError::Validation(format!(
                "`inputs` must have less than {max_input_length} tokens. Given: {seq_len}"
            )));
        }
    }

    metrics::histogram!("te_request_input_length", seq_len as f64);

    Ok(Encoding {
        input_ids: encoding.get_ids().to_vec(),
        token_type_ids: encoding.get_type_ids().to_vec(),
        position_ids: (position_offset as u32..(seq_len + position_offset) as u32)
            .collect::<Vec<_>>(),
    })
}

#[derive(Debug)]
pub struct Encoding {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
}

type TokenizerRequest = (
    String,
    bool,
    oneshot::Sender<Result<Encoding, TextEmbeddingsError>>,
    Span,
);
