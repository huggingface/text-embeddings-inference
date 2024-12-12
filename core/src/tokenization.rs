/// Payload tokenization logic
use crate::TextEmbeddingsError;
use std::collections::HashMap;
use tokenizers::tokenizer::Tokenizer;
pub use tokenizers::Encoding as RawEncoding;
use tokenizers::{TruncationDirection, TruncationParams, TruncationStrategy};
use tokio::sync::oneshot;
use tracing::{instrument, Span};

static MAX_CHAR_MULTIPLIER: usize = 250;

/// Validation
#[derive(Debug, Clone)]
pub struct Tokenization {
    /// Channel to communicate with the background tokenization task
    sender: async_channel::Sender<TokenizerRequest>,
}

impl Tokenization {
    pub fn new(
        workers: usize,
        tokenizer: Tokenizer,
        max_input_length: usize,
        position_offset: usize,
        default_prompt: Option<String>,
        prompts: Option<HashMap<String, String>>,
    ) -> Self {
        tracing::info!("Starting {workers} tokenization workers");

        // Create channel
        let (sender, receiver) = async_channel::bounded(workers * 4);

        // Create workers
        for _ in 0..workers {
            let tokenizer_clone = tokenizer.clone();
            let receiver_clone = receiver.clone();
            let default_prompt_clone = default_prompt.clone();
            let prompts_clone = prompts.clone();
            // Spawn worker
            std::thread::spawn(move || {
                tokenizer_worker(
                    tokenizer_clone,
                    max_input_length,
                    position_offset,
                    default_prompt_clone,
                    prompts_clone,
                    receiver_clone,
                )
            });
        }

        Self { sender }
    }

    #[instrument(skip_all)]
    pub async fn encode(
        &self,
        inputs: EncodingInput,
        truncate: bool,
        truncation_direction: TruncationDirection,
        prompt_name: Option<String>,
    ) -> Result<ValidEncoding, TextEmbeddingsError> {
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
            .send(TokenizerRequest::Encode(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                response_sender,
                Span::current(),
            ))
            .await
            .expect("Tokenization background task dropped the receiver. This is a bug.");

        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.expect("Tokenization background task dropped the sender without sending a response. This is a bug.")
    }

    #[instrument(skip_all)]
    pub async fn tokenize(
        &self,
        inputs: EncodingInput,
        add_special_tokens: bool,
        prompt_name: Option<String>,
    ) -> Result<(Option<String>, RawEncoding), TextEmbeddingsError> {
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
            .send(TokenizerRequest::Tokenize(
                inputs,
                add_special_tokens,
                prompt_name,
                response_sender,
                Span::current(),
            ))
            .await
            .expect("Tokenization background task dropped the receiver. This is a bug.");

        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.expect("Tokenization background task dropped the sender without sending a response. This is a bug.")
    }

    #[instrument(skip_all)]
    pub async fn decode(
        &self,
        ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> Result<String, TextEmbeddingsError> {
        // Check if inputs is empty
        if ids.is_empty() {
            return Err(TextEmbeddingsError::Validation(
                "`input_ids` cannot be empty".to_string(),
            ));
        }

        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send request to the background validation task
        // Unwrap is safe here
        self.sender
            .send(TokenizerRequest::Decode(
                ids,
                skip_special_tokens,
                response_sender,
                Span::current(),
            ))
            .await
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
    default_prompt: Option<String>,
    prompts: Option<HashMap<String, String>>,
    receiver: async_channel::Receiver<TokenizerRequest>,
) {
    // Loop over requests
    while let Ok(request) = receiver.recv_blocking() {
        match request {
            TokenizerRequest::Encode(
                inputs,
                truncate,
                truncation_direction,
                prompt_name,
                response_tx,
                parent_span,
            ) => {
                parent_span.in_scope(|| {
                    if !response_tx.is_closed() {
                        let default_prompt_clone = match prompt_name {
                            None => default_prompt.clone(),
                            Some(_) => None,
                        };

                        // It's possible that the user dropped its request resulting in a send error.
                        // We just discard the error
                        let _ = response_tx.send(encode_input(
                            inputs,
                            truncate,
                            truncation_direction,
                            max_input_length,
                            position_offset,
                            default_prompt_clone,
                            prompt_name,
                            prompts.as_ref(),
                            &mut tokenizer,
                        ));
                    }
                })
            }
            TokenizerRequest::Tokenize(
                inputs,
                add_special_tokens,
                prompt_name,
                response_tx,
                parent_span,
            ) => {
                parent_span.in_scope(|| {
                    if !response_tx.is_closed() {
                        let default_prompt_clone = match prompt_name {
                            None => default_prompt.clone(),
                            Some(_) => None,
                        };

                        // It's possible that the user dropped its request resulting in a send error.
                        // We just discard the error
                        let _ = response_tx.send(tokenize_input(
                            inputs,
                            add_special_tokens,
                            max_input_length,
                            None,
                            default_prompt_clone,
                            prompt_name,
                            prompts.as_ref(),
                            &mut tokenizer,
                        ));
                    }
                })
            }
            TokenizerRequest::Decode(ids, skip_special_tokens, response_tx, parent_span) => {
                parent_span.in_scope(|| {
                    if !response_tx.is_closed() {
                        // It's possible that the user dropped its request resulting in a send error.
                        // We just discard the error
                        let _ =
                            response_tx.send(decode_ids(ids, skip_special_tokens, &mut tokenizer));
                    }
                })
            }
        }
    }
}

fn decode_ids(
    ids: Vec<u32>,
    skip_special_tokens: bool,
    tokenizer: &mut Tokenizer,
) -> Result<String, TextEmbeddingsError> {
    Ok(tokenizer
        .with_truncation(None)?
        .decode(&ids, skip_special_tokens)?)
}

fn prepare_pre_prompt(
    default_prompt: Option<String>,
    prompt_name: Option<String>,
    prompts: Option<&HashMap<String, String>>,
) -> Result<Option<String>, TextEmbeddingsError> {
    let pre_prompt = if let Some(prompt_name) = prompt_name.as_ref() {
        match prompts {
            None => {
                return Err(TextEmbeddingsError::Validation(format!("`default-prompt-name` is set to `{prompt_name}` but no prompts were found in the Sentence Transformers configuration")));
            }
            Some(prompts) if !prompts.contains_key(prompt_name) => {
                return Err(TextEmbeddingsError::Validation(format!("`default-prompt-name` is set to `{prompt_name}` but it was not found in the Sentence Transformers prompts. Available prompts: {:?}", prompts.keys())));
            }
            Some(prompts) => prompts.get(prompt_name).cloned(),
        }
    } else {
        default_prompt
    };
    Ok(pre_prompt)
}

#[allow(clippy::too_many_arguments)]
fn tokenize_input(
    mut inputs: EncodingInput,
    add_special_tokens: bool,
    max_input_length: usize,
    truncate_params: Option<TruncationParams>,
    default_prompt: Option<String>,
    prompt_name: Option<String>,
    prompts: Option<&HashMap<String, String>>,
    tokenizer: &mut Tokenizer,
) -> Result<(Option<String>, RawEncoding), TextEmbeddingsError> {
    let pre_prompt = prepare_pre_prompt(default_prompt, prompt_name, prompts)?;

    let input_chars = inputs.count_chars();
    let limit = max_input_length * MAX_CHAR_MULTIPLIER;
    if input_chars > limit {
        if truncate_params.is_none() {
            return Err(TextEmbeddingsError::Validation(format!(
                "`inputs` must have less than {limit} characters. Given: {input_chars}"
            )));
        }
        inputs.apply_limit(limit);
    }

    let encoding = match inputs {
        // encode input
        EncodingInput::Single(s) => {
            let s = if let Some(mut pre_prompt) = pre_prompt {
                pre_prompt.push_str(&s);
                pre_prompt
            } else {
                s
            };

            let encoding = tokenizer
                .with_truncation(truncate_params)?
                .encode::<&str>(&s, add_special_tokens)?;

            (Some(s), encoding)
        }
        EncodingInput::Dual(s1, s2) => {
            if pre_prompt.is_some() {
                return Err(TextEmbeddingsError::Validation(
                    "`prompt_name` cannot be set with dual inputs".to_string(),
                ));
            }

            (
                None,
                tokenizer
                    .with_truncation(truncate_params)?
                    .encode::<(String, String)>((s1, s2), add_special_tokens)?,
            )
        }
        // input is encoded -> convert to tokenizers Encoding
        EncodingInput::Ids(ids) => {
            if let Some(mut pre_prompt) = pre_prompt {
                let text = tokenizer.decode(&ids, true)?;
                pre_prompt.push_str(&text);

                let encoding = tokenizer
                    .with_truncation(truncate_params)?
                    .encode::<&str>(&pre_prompt, true)?;

                (Some(pre_prompt), encoding)
            } else {
                let text = tokenizer.decode(&ids, false)?;

                let encoding = tokenizer
                    .with_truncation(truncate_params)?
                    .encode::<&str>(&text, false)?;

                (Some(text), encoding)
            }
        }
    };
    Ok(encoding)
}

/// Get input length and optionally truncate it
#[allow(clippy::too_many_arguments)]
fn encode_input(
    inputs: EncodingInput,
    truncate: bool,
    truncation_direction: TruncationDirection,
    max_input_length: usize,
    position_offset: usize,
    default_prompt: Option<String>,
    prompt_name: Option<String>,
    prompts: Option<&HashMap<String, String>>,
    tokenizer: &mut Tokenizer,
) -> Result<ValidEncoding, TextEmbeddingsError> {
    // Default truncation params
    let truncate_params = truncate.then_some(TruncationParams {
        direction: truncation_direction,
        max_length: max_input_length,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
    });

    let (_, encoding) = tokenize_input(
        inputs,
        true,
        max_input_length,
        truncate_params,
        default_prompt,
        prompt_name,
        prompts,
        tokenizer,
    )?;
    let seq_len = encoding.len();

    if seq_len > max_input_length {
        return Err(TextEmbeddingsError::Validation(format!(
            "`inputs` must have less than {max_input_length} tokens. Given: {seq_len}"
        )));
    }
    let histogram = metrics::histogram!("te_request_input_length");
    histogram.record(seq_len as f64);
    Ok(ValidEncoding {
        input_ids: encoding.get_ids().to_vec(),
        token_type_ids: encoding.get_type_ids().to_vec(),
        position_ids: (position_offset as u32..(seq_len + position_offset) as u32)
            .collect::<Vec<_>>(),
    })
}

#[derive(Debug)]
pub struct ValidEncoding {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
}

#[derive(Debug)]
pub enum EncodingInput {
    Single(String),
    Dual(String, String),
    Ids(Vec<u32>),
}

impl EncodingInput {
    fn is_empty(&self) -> bool {
        match self {
            EncodingInput::Single(s) => s.is_empty(),
            EncodingInput::Dual(s1, s2) => s1.is_empty() && s2.is_empty(),
            EncodingInput::Ids(v) => v.is_empty(),
        }
    }

    fn count_chars(&self) -> usize {
        match self {
            EncodingInput::Single(s) => s.chars().count(),
            EncodingInput::Dual(s1, s2) => s1.chars().count() + s2.chars().count(),
            EncodingInput::Ids(v) => v.len(),
        }
    }

    fn apply_limit(&mut self, limit: usize) {
        let truncate_string = |s: &mut String, limit: usize| {
            if s.is_char_boundary(limit) {
                s.truncate(limit)
            }
        };

        match self {
            EncodingInput::Single(s) => {
                truncate_string(s, limit);
            }
            EncodingInput::Dual(s1, s2) => {
                truncate_string(s1, limit / 2);
                truncate_string(s2, limit / 2);
            }
            EncodingInput::Ids(_) => {}
        }
    }
}

impl From<String> for EncodingInput {
    fn from(value: String) -> Self {
        Self::Single(value)
    }
}

impl From<(String, String)> for EncodingInput {
    fn from(value: (String, String)) -> Self {
        Self::Dual(value.0, value.1)
    }
}

enum TokenizerRequest {
    Encode(
        EncodingInput,
        bool,
        TruncationDirection,
        Option<String>,
        oneshot::Sender<Result<ValidEncoding, TextEmbeddingsError>>,
        Span,
    ),
    Tokenize(
        EncodingInput,
        bool,
        Option<String>,
        oneshot::Sender<Result<(Option<String>, RawEncoding), TextEmbeddingsError>>,
        Span,
    ),
    Decode(
        Vec<u32>,
        bool,
        oneshot::Sender<Result<String, TextEmbeddingsError>>,
        Span,
    ),
}
