use crate::ErrorType;
use serde::de::{SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::json;
use std::fmt::Formatter;
use text_embeddings_core::tokenization::EncodingInput;
use utoipa::openapi::{RefOr, Schema};
use utoipa::ToSchema;

#[derive(Debug)]
pub(crate) enum Sequence {
    Single(String),
    Pair(String, String),
}

impl Sequence {
    pub(crate) fn count_chars(&self) -> usize {
        match self {
            Sequence::Single(s) => s.chars().count(),
            Sequence::Pair(s1, s2) => s1.chars().count() + s2.chars().count(),
        }
    }
}

impl From<Sequence> for EncodingInput {
    fn from(value: Sequence) -> Self {
        match value {
            Sequence::Single(s) => Self::Single(s),
            Sequence::Pair(s1, s2) => Self::Dual(s1, s2),
        }
    }
}

#[derive(Debug)]
pub(crate) enum PredictInput {
    Single(Sequence),
    Batch(Vec<Sequence>),
}

impl<'de> Deserialize<'de> for PredictInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Internal {
            Single(String),
            Multiple(Vec<String>),
        }

        struct PredictInputVisitor;

        impl<'de> Visitor<'de> for PredictInputVisitor {
            type Value = PredictInput;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter.write_str(
                    "a string, \
                    a pair of strings [string, string] \
                    or a batch of mixed strings and pairs [[string], [string, string], ...]",
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(PredictInput::Single(Sequence::Single(v.to_string())))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let sequence_from_vec = |mut value: Vec<String>| {
                    // Validate that value is correct
                    match value.len() {
                        1 => Ok(Sequence::Single(value.pop().unwrap())),
                        2 => {
                            // Second element is last
                            let second = value.pop().unwrap();
                            let first = value.pop().unwrap();
                            Ok(Sequence::Pair(first, second))
                        }
                        // Sequence can only be a single string or a pair of strings
                        _ => Err(de::Error::invalid_length(value.len(), &self)),
                    }
                };

                // Get first element
                // This will determine if input is a batch or not
                let s = match seq
                    .next_element::<Internal>()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?
                {
                    // Input is not a batch
                    // Return early
                    Internal::Single(value) => {
                        // Option get second element
                        let second = seq.next_element()?;

                        if seq.next_element::<String>()?.is_some() {
                            // Error as we do not accept > 2 elements
                            return Err(de::Error::invalid_length(3, &self));
                        }

                        if let Some(second) = second {
                            // Second element exists
                            // This is a pair
                            return Ok(PredictInput::Single(Sequence::Pair(value, second)));
                        } else {
                            // Second element does not exist
                            return Ok(PredictInput::Single(Sequence::Single(value)));
                        }
                    }
                    // Input is a batch
                    Internal::Multiple(value) => sequence_from_vec(value),
                }?;

                let mut batch = Vec::with_capacity(32);
                // Push first sequence
                batch.push(s);

                // Iterate on all sequences
                while let Some(value) = seq.next_element::<Vec<String>>()? {
                    // Validate sequence
                    let s = sequence_from_vec(value)?;
                    // Push to batch
                    batch.push(s);
                }
                Ok(PredictInput::Batch(batch))
            }
        }

        deserializer.deserialize_any(PredictInputVisitor)
    }
}

impl<'__s> ToSchema<'__s> for PredictInput {
    fn schema() -> (&'__s str, RefOr<Schema>) {
        (
            "PredictInput",
            utoipa::openapi::OneOfBuilder::new()
                .item(
                    utoipa::openapi::ObjectBuilder::new()
                        .schema_type(utoipa::openapi::SchemaType::String)
                        .description(Some("A single string")),
                )
                .item(
                    utoipa::openapi::ArrayBuilder::new()
                        .items(
                            utoipa::openapi::ObjectBuilder::new()
                                .schema_type(utoipa::openapi::SchemaType::String),
                        )
                        .description(Some("A pair of strings"))
                        .min_items(Some(2))
                        .max_items(Some(2)),
                )
                .item(
                    utoipa::openapi::ArrayBuilder::new().items(
                        utoipa::openapi::OneOfBuilder::new()
                            .item(
                                utoipa::openapi::ArrayBuilder::new()
                                    .items(
                                        utoipa::openapi::ObjectBuilder::new()
                                            .schema_type(utoipa::openapi::SchemaType::String),
                                    )
                                    .description(Some("A single string"))
                                    .min_items(Some(1))
                                    .max_items(Some(1)),
                            )
                            .item(
                                utoipa::openapi::ArrayBuilder::new()
                                    .items(
                                        utoipa::openapi::ObjectBuilder::new()
                                            .schema_type(utoipa::openapi::SchemaType::String),
                                    )
                                    .description(Some("A pair of strings"))
                                    .min_items(Some(2))
                                    .max_items(Some(2)),
                            )
                    ).description(Some("A batch")),
                )
                .description(Some(
                    "Model input. \
                Can be either a single string, a pair of strings or a batch of mixed single and pairs \
                of strings.",
                ))
                .example(Some(json!("What is Deep Learning?")))
                .into(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, ToSchema, Eq, Default)]
pub(crate) enum TruncationDirection {
    Left,
    #[default]
    Right,
}

impl From<TruncationDirection> for tokenizers::TruncationDirection {
    fn from(value: TruncationDirection) -> Self {
        match value {
            TruncationDirection::Left => Self::Left,
            TruncationDirection::Right => Self::Right,
        }
    }
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct PredictRequest {
    pub inputs: PredictInput,
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub raw_scores: bool,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Prediction {
    #[schema(example = "0.5")]
    pub score: f32,
    #[schema(example = "admiration")]
    pub label: String,
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum PredictResponse {
    Single(Vec<Prediction>),
    Batch(Vec<Vec<Prediction>>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct RerankRequest {
    #[schema(example = "What is Deep Learning?")]
    pub query: String,
    #[schema(example = json!(["Deep Learning is ..."]))]
    pub texts: Vec<String>,
    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub raw_scores: bool,
    #[serde(default)]
    #[schema(default = "false", example = "false")]
    pub return_text: bool,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Rank {
    #[schema(example = "0")]
    pub index: usize,
    #[schema(nullable = true, example = "Deep Learning is ...", default = "null")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[schema(example = "1.0")]
    pub score: f32,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct RerankResponse(pub Vec<Rank>);

#[derive(Deserialize, ToSchema, Debug)]
#[serde(untagged)]
pub(crate) enum InputType {
    String(String),
    Ids(Vec<u32>),
}

impl InputType {
    pub(crate) fn count_chars(&self) -> usize {
        match self {
            InputType::String(s) => s.chars().count(),
            InputType::Ids(v) => v.len(),
        }
    }
}

impl From<InputType> for EncodingInput {
    fn from(value: InputType) -> Self {
        match value {
            InputType::String(s) => Self::Single(s),
            InputType::Ids(v) => Self::Ids(v),
        }
    }
}

#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum Input {
    Single(InputType),
    Batch(Vec<InputType>),
}

#[derive(Deserialize, ToSchema, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct OpenAICompatRequest {
    pub input: Input,
    #[allow(dead_code)]
    #[schema(nullable = true, example = "null")]
    pub model: Option<String>,
    #[allow(dead_code)]
    #[schema(nullable = true, example = "null")]
    pub user: Option<String>,
    #[schema(default = "float", example = "float")]
    #[serde(default)]
    pub encoding_format: EncodingFormat,
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum Embedding {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatEmbedding {
    #[schema(example = "embedding")]
    pub object: &'static str,
    #[schema(example = json!([0.0, 1.0, 2.0]))]
    pub embedding: Embedding,
    #[schema(example = "0")]
    pub index: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatUsage {
    #[schema(example = "512")]
    pub prompt_tokens: usize,
    #[schema(example = "512")]
    pub total_tokens: usize,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatResponse {
    #[schema(example = "list")]
    pub object: &'static str,
    pub data: Vec<OpenAICompatEmbedding>,
    #[schema(example = "thenlper/gte-base")]
    pub model: String,
    pub usage: OpenAICompatUsage,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct SimilarityInput {
    /// The string that you wish to compare the other strings with. This can be a phrase, sentence,
    /// or longer passage, depending on the model being used.
    #[schema(example = "What is Deep Learning?")]
    pub source_sentence: String,
    /// A list of strings which will be compared against the source_sentence.
    #[schema(example = json!(["What is Machine Learning?"]))]
    pub sentences: Vec<String>,
}

#[derive(Deserialize, ToSchema, Default)]
pub(crate) struct SimilarityParameters {
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct SimilarityRequest {
    pub inputs: SimilarityInput,
    /// Additional inference parameters for Sentence Similarity
    #[schema(default = "null", example = "null", nullable = true)]
    pub parameters: Option<SimilarityParameters>,
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([0.0, 1.0, 0.5]))]
pub(crate) struct SimilarityResponse(pub Vec<f32>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedRequest {
    pub inputs: Input,
    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
    #[serde(default = "default_normalize")]
    #[schema(default = "true", example = "true")]
    pub normalize: bool,
}

fn default_normalize() -> bool {
    true
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([[0.0, 1.0, 2.0]]))]
pub(crate) struct EmbedResponse(pub Vec<Vec<f32>>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedSparseRequest {
    pub inputs: Input,
    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct SparseValue {
    pub index: usize,
    pub value: f32,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct EmbedSparseResponse(pub Vec<Vec<SparseValue>>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct EmbedAllRequest {
    pub inputs: Input,
    #[serde(default)]
    #[schema(default = "false", example = "false", nullable = true)]
    pub truncate: Option<bool>,
    #[serde(default)]
    #[schema(default = "right", example = "right")]
    pub truncation_direction: TruncationDirection,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([[[0.0, 1.0, 2.0]]]))]
pub(crate) struct EmbedAllResponse(pub Vec<Vec<Vec<f32>>>);

#[derive(Serialize, ToSchema)]
pub(crate) struct OpenAICompatErrorResponse {
    pub message: String,
    pub code: u16,
    #[serde(rename(serialize = "type"))]
    pub error_type: ErrorType,
}

#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum TokenizeInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct TokenizeRequest {
    pub inputs: TokenizeInput,
    #[serde(default = "default_add_special_tokens")]
    #[schema(default = "true", example = "true")]
    pub add_special_tokens: bool,
    /// The name of the prompt that should be used by for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    #[schema(default = "null", example = "null", nullable = true)]
    pub prompt_name: Option<String>,
}

fn default_add_special_tokens() -> bool {
    true
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct SimpleToken {
    #[schema(example = 0)]
    pub id: u32,
    #[schema(example = "test")]
    pub text: String,
    #[schema(example = "false")]
    pub special: bool,
    #[schema(example = 0)]
    pub start: Option<usize>,
    #[schema(example = 2)]
    pub stop: Option<usize>,
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!([[{"id": 0, "text": "test", "special": false, "start": 0, "stop": 2}]]))]
pub(crate) struct TokenizeResponse(pub Vec<Vec<SimpleToken>>);

#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum InputIds {
    Single(Vec<u32>),
    Batch(Vec<Vec<u32>>),
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct DecodeRequest {
    pub ids: InputIds,
    #[serde(default = "default_skip_special_tokens")]
    #[schema(default = "true", example = "true")]
    pub skip_special_tokens: bool,
}

fn default_skip_special_tokens() -> bool {
    true
}

#[derive(Serialize, ToSchema)]
#[schema(example = json!(["test"]))]
pub(crate) struct DecodeResponse(pub Vec<String>);

#[derive(Deserialize, ToSchema)]
pub(crate) struct VertexRequest {
    pub instances: Vec<serde_json::Value>,
}

#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum VertexPrediction {
    Embed(EmbedResponse),
    EmbedSparse(EmbedSparseResponse),
    Predict(PredictResponse),
    Rerank(RerankResponse),
}

#[derive(Serialize, ToSchema)]
pub(crate) struct VertexResponse {
    pub predictions: Vec<VertexPrediction>,
}
