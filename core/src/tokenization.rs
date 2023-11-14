/// Payload tokenization logic
use crate::TextEmbeddingsError;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::{EncodeInput, TruncationDirection, TruncationParams, TruncationStrategy};
use tokio::sync::{mpsc, oneshot};
use tracing::{instrument, Span};

/// Validation
#[derive(Debug, Clone)]
pub struct Tokenization {
    /// Channel to communicate with the background tokenization task
    sender: mpsc::UnboundedSender<TokenizerRequest>,
}

impl Tokenization {
    pub fn new(
        workers: usize,
        tokenizer: Tokenizer,
        max_input_length: usize,
        position_offset: usize,
    ) -> Self {
        tracing::info!("Starting {workers} tokenization workers");

        // Create channel
        let (sender, mut round_robin_receiver) = mpsc::unbounded_channel();
        let mut senders = Vec::with_capacity(workers);

        // Create workers
        for _ in 0..workers {
            let tokenizer_clone = tokenizer.clone();
            let (tokenizer_sender, tokenizer_receiver) = mpsc::unbounded_channel();
            senders.push(tokenizer_sender);

            // Spawn worker
            tokio::task::spawn_blocking(move || {
                tokenizer_worker(
                    tokenizer_clone,
                    max_input_length,
                    position_offset,
                    tokenizer_receiver,
                )
            });
        }

        // Create tokenization round robin task
        tokio::spawn(async move {
            // Loop over requests
            loop {
                for sender in &senders {
                    match round_robin_receiver.recv().await {
                        None => return,
                        Some(request) => sender.send(request).unwrap(),
                    };
                }
            }
        });

        Self { sender }
    }

    #[instrument(skip_all)]
    pub async fn encode(
        &self,
        inputs: EncodingInput,
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
        response_receiver.await.expect("Tokenization background task dropped the sender without sending a response. This is a bug.")
    }
}

/// Start tokenization workers
fn tokenizer_worker(
    mut tokenizer: Tokenizer,
    max_input_length: usize,
    position_offset: usize,
    mut receiver: mpsc::UnboundedReceiver<TokenizerRequest>,
) {
    // Loop over requests
    while let Some((inputs, truncate, response_tx, parent_span)) = receiver.blocking_recv() {
        parent_span.in_scope(|| {
            if !response_tx.is_closed() {
                // It's possible that the user dropped its request resulting in a send error.
                // We just discard the error
                let _ = response_tx.send(encode_input(
                    inputs,
                    truncate,
                    max_input_length,
                    position_offset,
                    &mut tokenizer,
                ));
            }
        })
    }
}

/// Get input length and optionally truncate it
fn encode_input(
    inputs: EncodingInput,
    truncate: bool,
    max_input_length: usize,
    position_offset: usize,
    tokenizer: &mut Tokenizer,
) -> Result<Encoding, TextEmbeddingsError> {
    // Default truncation params
    let truncate_params = truncate.then_some(TruncationParams {
        direction: TruncationDirection::Right,
        max_length: max_input_length,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
    });

    let inputs: EncodeInput = match inputs {
        EncodingInput::Single(s) => s.into(),
        EncodingInput::Dual(s1, s2) => (s1, s2).into(),
    };

    let encoding = tokenizer
        .with_truncation(truncate_params)?
        .encode(inputs, true)?;
    let seq_len = encoding.len();

    if seq_len > max_input_length {
        return Err(TextEmbeddingsError::Validation(format!(
            "`inputs` must have less than {max_input_length} tokens. Given: {seq_len}"
        )));
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

#[derive(Debug)]
pub enum EncodingInput {
    Single(String),
    Dual(String, String),
}

impl EncodingInput {
    fn is_empty(&self) -> bool {
        match self {
            EncodingInput::Single(s) => s.is_empty(),
            EncodingInput::Dual(s1, s2) => s1.is_empty() && s2.is_empty(),
        }
    }
}

impl From<String> for EncodingInput {
    fn from(value: String) -> Self {
        Self::Single(value)
    }
}

type TokenizerRequest = (
    EncodingInput,
    bool,
    oneshot::Sender<Result<Encoding, TextEmbeddingsError>>,
    Span,
);
