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

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct SimpleToken {
    pub id: u32,
    pub text: String,
    pub special: bool,
    pub start: Option<usize>,
    pub stop: Option<usize>,
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

pub fn into_tokens(encoding: tokenizers::Encoding, input: &str) -> Vec<SimpleToken> {
    encoding
        .get_ids()
        .iter()
        .zip(encoding.get_offsets())
        .zip(encoding.get_special_tokens_mask())
        .zip(encoding.get_tokens())
        .map(|(((&id, &(start, stop)), special), token)| {
            let special = *special == 1;
            match special {
                true => SimpleToken {
                    id,
                    text: token.clone(),
                    special,
                    start: None,
                    stop: None,
                },
                false => {
                    let text: Vec<u8> = input.bytes().skip(start).take(stop - start).collect();
                    let text: String = String::from_utf8_lossy(&text).to_string();
                    SimpleToken {
                        id,
                        text,
                        special,
                        start: Some(start),
                        stop: Some(stop),
                    }
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: This test requires hf-hub with ureq feature enabled
    // Disabled for now as it's not critical for Milestone 3 and causes build failures
    // TODO: Re-enable when proper feature flags are configured
    /*
    #[test]
    fn tokenizer() {
        use hf_hub::api::sync::ApiBuilder;
        let api = ApiBuilder::from_env().build().unwrap();
        let filename = api
            .model("BAAI/bge-m3".to_string())
            .get("tokenizer.json")
            .unwrap();
        let string = "这是一个文本向量化的测试句子";
        let tokenizer = Tokenizer::from_file(filename).unwrap();

        let encoded = tokenizer.encode(string, true).unwrap();
        assert_eq!(
            encoded.get_offsets(),
            vec![
                (0, 0),
                (0, 3),
                (0, 12),
                (12, 18),
                (18, 21),
                (21, 24),
                (24, 30),
                (30, 36),
                (36, 39),
                (39, 42),
                (0, 0)
            ]
        );

        let tokens = into_tokens(encoded, &string);
        assert_eq!(
            tokens,
            vec![
                SimpleToken {
                    id: 0,
                    text: "<s>".to_string(),
                    special: true,
                    start: None,
                    stop: None
                },
                SimpleToken {
                    id: 6,
                    text: "这".to_string(),
                    special: false,
                    start: Some(0),
                    stop: Some(3)
                },
                SimpleToken {
                    id: 100013,
                    text: "这是一个".to_string(),
                    special: false,
                    start: Some(0),
                    stop: Some(12)
                },
                SimpleToken {
                    id: 189061,
                    text: "文本".to_string(),
                    special: false,
                    start: Some(12),
                    stop: Some(18)
                },
                SimpleToken {
                    id: 2110,
                    text: "向".to_string(),
                    special: false,
                    start: Some(18),
                    stop: Some(21)
                },
                SimpleToken {
                    id: 3272,
                    text: "量".to_string(),
                    special: false,
                    start: Some(21),
                    stop: Some(24)
                },
                SimpleToken {
                    id: 41904,
                    text: "化的".to_string(),
                    special: false,
                    start: Some(24),
                    stop: Some(30)
                },
                SimpleToken {
                    id: 49125,
                    text: "测试".to_string(),
                    special: false,
                    start: Some(30),
                    stop: Some(36)
                },
                SimpleToken {
                    id: 27683,
                    text: "句".to_string(),
                    special: false,
                    start: Some(36),
                    stop: Some(39)
                },
                SimpleToken {
                    id: 1344,
                    text: "子".to_string(),
                    special: false,
                    start: Some(39),
                    stop: Some(42)
                },
                SimpleToken {
                    id: 2,
                    text: "</s>".to_string(),
                    special: true,
                    start: None,
                    stop: None
                }
            ]
        );
    }
    */
}

/// Listwise reranking을 위한 left padding으로 프롬프트 인코딩
///
/// Qwen3 모델은 인과성을 유지하기 위해 left padding이 필요합니다.
///
/// ⚠️ **SHOULD-FIX S2: 향상된 문서화**
/// - 이것은 단일 샘플을 인코딩합니다 (배치 없음), 따라서 패딩이 적용되지 않습니다
/// - Attention mask는 패드 토큰이 없으므로 모두 1입니다
/// - 패딩은 서로 다른 길이의 여러 시퀀스를 배치할 때만 필요합니다
/// - `add_special_tokens=true`는 HuggingFace Transformers 기본 동작과 일치합니다
///
/// # 인자
/// * `tokenizer` - 토크나이저 인스턴스 (left padding으로 설정되어야 함)
/// * `prompt` - 완전한 프롬프트 문자열 (이미 모든 특수 토큰 포함)
/// * `max_length` - 최대 시퀀스 길이 (선택적, 검증용)
///
/// # 반환
/// attention_mask=모두 1인 토큰화된 인코딩 (단일 샘플의 경우 패딩 없음)
pub fn encode_listwise(
    tokenizer: &Tokenizer,
    prompt: &str,
    max_length: Option<usize>,
) -> anyhow::Result<tokenizers::Encoding> {
    use anyhow::anyhow;

    // 인코딩 정책 (S2): 단일 샘플 (배치 없음), 패딩 불필요
    // 단일 시퀀스 인코딩에는 패딩이 없으므로 모든 attention mask 값은 1
    // 패딩은 여러 시퀀스를 배치할 때만 적용됨

    // 중요: add_special_tokens=true는 Python Transformers 기본값과 일치
    // 정확한 블록 청킹을 위해 modeling.py와 토큰 카운트가 일치하도록 보장
    // ChatML 토큰(<|im_start|>, <|im_end|>)을 인코딩에 포함
    let encoding = tokenizer
        .encode(prompt, true) // false였음 - 토큰 길이 불일치 발생!
        .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

    // 길이 검증
    if let Some(max_len) = max_length {
        if encoding.len() > max_len {
            return Err(anyhow!(
                "Prompt exceeds max length: {} > {}. Try reducing document count or length.",
                encoding.len(),
                max_len
            ));
        }
    }

    Ok(encoding)
}

// 주의: Padding side/token은 모델 로드 중에 설정되어야 합니다 (Milestone 3.2 참조).

/// 토큰 제한을 적용하기 위해 텍스트 절단 및 디코딩
///
/// Python 참조 `_truncate_texts` 동작과 일치:
/// - 쿼리는 max_query_length로 절단 (기본값 512)
/// - 각 문서는 max_doc_length로 절단 (기본값 2048)
/// - 디코딩된 문자열과 토큰 길이 반환
///
/// 토크나이제이션 정책:
/// - HuggingFace Transformers 기본값과 일치하는 `add_special_tokens=false` 사용
/// - 이것은 encode/decode 사이클의 표준 동작
/// - 특수 토큰(<|embed_token|>, <|rerank_token|>)은 토크나이저가 아닌 프롬프트 빌더에 의해 추가됨
///
/// # 반환
/// (truncated_query, truncated_docs, doc_token_lengths, query_token_length)
pub fn truncate_texts(
    tokenizer: &Tokenizer,
    query: &str,
    documents: &[String],
    max_query_length: usize,
    max_doc_length: usize,
) -> anyhow::Result<(String, Vec<String>, Vec<usize>, usize)> {
    use anyhow::anyhow;

    // 중요 토크나이제이션 정책 (modeling.py 패리티):
    // - encode(..., true): 특수 토큰 추가 (HF Transformers 기본값과 일치)
    // - decode(..., true): 디코딩시 특수 토큰 건너뛰기 (프롬프트에 BOS/EOS 방지)
    // 완전한 HF 패리티를 위해 둘 다 TRUE로 설정

    // 성능: clone 불필요 - 이 함수 동안 토크나이저는 불변
    let tk = tokenizer;

    // 쿼리
    let q_enc = tk
        .encode(query, true)
        .map_err(|e| anyhow!("encode(query): {}", e))?;
    let mut query_ids = q_enc.get_ids().to_vec();
    let mut query_trunc = query.to_string();
    if query_ids.len() > max_query_length {
        query_ids.truncate(max_query_length);
        // skip_special_tokens=true는 HF decode 기본값과 일치
        query_trunc = tk
            .decode(&query_ids, true)
            .map_err(|e| anyhow!("decode(query): {}", e))?;
    }
    let query_len = query_ids.len();

    // 문서들
    let mut docs_trunc = Vec::with_capacity(documents.len());
    let mut doc_lens = Vec::with_capacity(documents.len());
    for d in documents {
        let d_enc = tk
            .encode(d.as_str(), true)
            .map_err(|e| anyhow!("encode(doc): {}", e))?;
        let mut ids = d_enc.get_ids().to_vec();
        if ids.len() > max_doc_length {
            ids.truncate(max_doc_length);
            // skip_special_tokens=true는 HF decode 기본값과 일치
            docs_trunc.push(
                tk.decode(&ids, true)
                    .map_err(|e| anyhow!("decode(doc): {}", e))?,
            );
        } else {
            docs_trunc.push(d.clone());
        }
        doc_lens.push(ids.len());
    }

    Ok((query_trunc, docs_trunc, doc_lens, query_len))
}
