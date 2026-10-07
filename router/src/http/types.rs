use crate::ErrorType;
use serde::de::{SeqAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::json;
use serde_json::Value;
use std::collections::BTreeMap;
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
    #[serde(alias = "left", alias = "Left")]
    Left,
    #[serde(alias = "right", alias = "Right")]
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

#[derive(Debug)]
pub(crate) enum DecisionQuestion {
    Choice {
        instructions: String,
        criteria: BTreeMap<String, Value>,
    },
    Score {
        instructions: String,
        criteria: Vec<Value>,
    },
    Noul {
        instructions: String,
        criteria: Option<BTreeMap<String, Value>>,
    },
}

impl DecisionQuestion {
    pub(crate) fn qtype(&self) -> u32 {
        match self {
            Self::Choice { .. } => 0,
            Self::Score { .. } => 1,
            Self::Noul { .. } => 2,
        }
    }

    pub(crate) fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Noul { instructions, .. } => instructions,
        }
    }

    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
            Self::Noul { .. } => "noul",
        }
    }

    pub(crate) fn options(&self) -> Vec<DecisionOption> {
        match self {
            Self::Choice { criteria, .. } => criteria
                .iter()
                .map(|(label, description)| DecisionOption {
                    label: label.clone(),
                    description: description_value(description),
                })
                .collect(),
            Self::Score { criteria, .. } => criteria
                .iter()
                .enumerate()
                .map(|(index, value)| DecisionOption {
                    label: format!("level {index}"),
                    description: value_text(value),
                })
                .collect(),
            Self::Noul { criteria, .. } => ["false", "true"]
                .into_iter()
                .map(|label| DecisionOption {
                    label: label.to_string(),
                    description: criteria
                        .as_ref()
                        .and_then(|values| values.get(label))
                        .and_then(description_value),
                })
                .enumerate()
                .map(|(index, mut option)| {
                    if option.description.is_none() {
                        option.description = Some(match index {
                            0 => "no, the statement does not hold".to_string(),
                            _ => "yes, the statement holds".to_string(),
                        });
                    }
                    option
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DecisionOption {
    pub label: String,
    pub description: Option<String>,
}

fn value_text(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .or_else(|| (!value.is_null()).then(|| value.to_string()))
}

fn description_value(value: &Value) -> Option<String> {
    value
        .as_object()
        .and_then(|object| object.get("description"))
        .and_then(value_text)
        .or_else(|| value_text(value))
}

#[derive(Debug)]
pub(crate) enum DecisionAnswer {
    Choice {
        label: String,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
    Score {
        score: f32,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
    Noul {
        noul: f32,
    },
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum OpenAIDecisionInput {
    Text(String),
    Messages(Vec<OpenAIDecisionMessage>),
}

impl OpenAIDecisionInput {
    pub(crate) fn into_text(self) -> Result<String, String> {
        match self {
            Self::Text(text) => Ok(text),
            Self::Messages(messages) => {
                let mut text = Vec::new();
                for message in messages {
                    if message.role != "user" {
                        return Err("decision input messages must use the user role".to_string());
                    }
                    match message.content {
                        OpenAIDecisionMessageContent::Text(content) => text.push(content),
                        OpenAIDecisionMessageContent::Parts(parts) => {
                            for part in parts {
                                match part {
                                    OpenAIDecisionInputPart::Text { text: part_text } => {
                                        text.push(part_text);
                                    }
                                    OpenAIDecisionInputPart::Image { image_url: _ } => {
                                        return Err(
                                            "this decision model does not support image input"
                                                .to_string(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(text.join("\n"))
            }
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct OpenAIDecisionMessage {
    pub role: String,
    pub content: OpenAIDecisionMessageContent,
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum OpenAIDecisionMessageContent {
    Text(String),
    Parts(Vec<OpenAIDecisionInputPart>),
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(tag = "type")]
pub(crate) enum OpenAIDecisionInputPart {
    #[serde(rename = "input_text")]
    Text { text: String },
    #[serde(rename = "input_image")]
    #[allow(dead_code)]
    Image { image_url: Value },
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct OpenAIDecisionRequest {
    pub model: String,
    pub input: OpenAIDecisionInput,
    pub questions: Vec<OpenAIDecisionQuestion>,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(tag = "type")]
pub(crate) enum OpenAIDecisionQuestion {
    #[serde(rename = "predicate")]
    Predicate { name: String, instructions: String },
    #[serde(rename = "choice")]
    Choice {
        name: String,
        instructions: String,
        choices: Vec<OpenAIDecisionChoice>,
    },
    #[serde(rename = "score")]
    Score {
        name: String,
        instructions: String,
        levels: Vec<OpenAIDecisionLevel>,
    },
}

impl OpenAIDecisionQuestion {
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Predicate { name, .. } | Self::Choice { name, .. } | Self::Score { name, .. } => {
                name
            }
        }
    }

    pub(crate) fn to_model_question(&self) -> Result<DecisionQuestion, String> {
        match self {
            Self::Predicate { instructions, .. } => Ok(DecisionQuestion::Noul {
                instructions: instructions.clone(),
                criteria: None,
            }),
            Self::Choice {
                instructions,
                choices,
                ..
            } => {
                let criteria = choices
                    .iter()
                    .map(|choice| {
                        (
                            choice.value.clone(),
                            choice
                                .description
                                .clone()
                                .map(Value::String)
                                .unwrap_or(Value::Null),
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
                if choices.is_empty() || criteria.len() != choices.len() {
                    return Err(
                        "choice questions require at least one distinct choice value".to_string(),
                    );
                }
                Ok(DecisionQuestion::Choice {
                    instructions: instructions.clone(),
                    criteria,
                })
            }
            Self::Score {
                instructions,
                levels,
                ..
            } => {
                if levels.is_empty() {
                    return Err("score questions require at least one level".to_string());
                }
                Ok(DecisionQuestion::Score {
                    instructions: instructions.clone(),
                    criteria: levels
                        .iter()
                        .map(|level| match &level.description {
                            Some(description) => {
                                Value::String(format!("{}: {}", level.label, description))
                            }
                            None => Value::String(level.label.clone()),
                        })
                        .collect(),
                })
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub(crate) struct OpenAIDecisionChoice {
    pub value: String,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub(crate) struct OpenAIDecisionLevel {
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct OpenAIDecisionResponse {
    pub answers: Vec<OpenAIDecisionAnswer>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type")]
pub(crate) enum OpenAIDecisionAnswer {
    #[serde(rename = "predicate")]
    Predicate { name: String, probability: f32 },
    #[serde(rename = "choice")]
    Choice {
        name: String,
        choice: String,
        probabilities: Vec<OpenAIDecisionChoiceProbability>,
        confidence: f32,
    },
    #[serde(rename = "score")]
    Score {
        name: String,
        score: f32,
        probabilities: Vec<OpenAIDecisionScoreProbability>,
        confidence: f32,
    },
    #[serde(rename = "refusal")]
    #[allow(dead_code)]
    Refusal { name: String },
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct OpenAIDecisionChoiceProbability {
    pub value: String,
    pub probability: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct OpenAIDecisionScoreProbability {
    pub value: usize,
    pub label: String,
    pub probability: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct OpenAIDecisionErrorResponse {
    pub error: OpenAIDecisionError,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct OpenAIDecisionError {
    pub message: String,
    pub r#type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

pub(crate) fn decision_prompt(
    state: &str,
    qtype: &str,
    instructions: &str,
    options: &[DecisionOption],
) -> String {
    let state = state.replace("[MASK]", " ");
    let instructions = instructions.replace("[MASK]", " ");
    let options = options
        .iter()
        .map(|option| match &option.description {
            Some(description) => format!("{}: {}", option.label, description),
            None => option.label.clone(),
        })
        .map(|option| format!("[MASK] {}", option.replace("[MASK]", " ")))
        .collect::<Vec<_>>()
        .join(" ");
    return format!("[CLS] {qtype} question: {instructions} [SEP] {options} [SEP] {state} [SEP]");
}

#[cfg(test)]
mod tests {
    use super::{
        decision_prompt, DecisionOption, DecisionQuestion, OpenAIDecisionAnswer,
        OpenAIDecisionChoiceProbability, OpenAIDecisionQuestion, OpenAIDecisionRequest,
        OpenAIDecisionResponse, OpenAIDecisionScoreProbability,
    };
    use std::collections::BTreeMap;

    #[test]
    fn decision_prompt_preserves_text_state() {
        assert_eq!(
            decision_prompt(
                "ready",
                "choice",
                "open door",
                &[DecisionOption {
                    label: "yes".to_string(),
                    description: Some("open it".to_string()),
                }],
            ),
            "[CLS] choice question: open door [SEP] [MASK] yes: open it [SEP] ready [SEP]"
        );
    }

    #[test]
    fn decision_question_supports_shorthand_and_typed_forms() {
        assert_eq!(
            DecisionQuestion::Noul {
                instructions: "is it urgent?".to_string(),
                criteria: None,
            }
            .qtype(),
            2
        );
        assert_eq!(
            DecisionQuestion::Choice {
                instructions: "custom".to_string(),
                criteria: BTreeMap::new(),
            }
            .instructions(),
            "custom"
        );
    }

    #[test]
    fn choice_options_are_sorted_and_render_descriptions() {
        let question = DecisionQuestion::Choice {
            instructions: "where?".to_string(),
            criteria: BTreeMap::from([
                ("z".to_string(), serde_json::json!("last")),
                ("a".to_string(), serde_json::json!("first")),
            ]),
        };

        assert_eq!(
            question.options(),
            vec![
                DecisionOption {
                    label: "a".to_string(),
                    description: Some("first".to_string()),
                },
                DecisionOption {
                    label: "z".to_string(),
                    description: Some("last".to_string()),
                },
            ]
        );
    }

    #[test]
    fn score_and_noul_options_have_stable_semantics() {
        let score = DecisionQuestion::Score {
            instructions: "how urgent?".to_string(),
            criteria: vec![
                serde_json::json!("low"),
                serde_json::json!({"description": "high"}),
            ],
        };
        assert_eq!(score.options()[0].label, "level 0");
        assert_eq!(
            score.options()[1].description.as_deref(),
            Some("{\"description\":\"high\"}")
        );

        let noul = DecisionQuestion::Noul {
            instructions: "is it urgent?".to_string(),
            criteria: Some(BTreeMap::from([
                ("false".to_string(), serde_json::json!("no")),
                ("true".to_string(), serde_json::json!("yes")),
            ])),
        };
        assert_eq!(noul.options()[0].description.as_deref(), Some("no"));
        assert_eq!(noul.options()[1].description.as_deref(), Some("yes"));

        let default_noul = DecisionQuestion::Noul {
            instructions: "is it urgent?".to_string(),
            criteria: None,
        };
        assert_eq!(
            default_noul.options()[0].description.as_deref(),
            Some("no, the statement does not hold")
        );
        assert_eq!(
            default_noul.options()[1].description.as_deref(),
            Some("yes, the statement holds")
        );
    }

    #[test]
    fn decision_prompt_allows_only_one_marker_per_option() {
        let prompt = decision_prompt(
            "state [MASK]",
            "choice",
            "question [MASK]",
            &[
                DecisionOption {
                    label: "yes [MASK]".to_string(),
                    description: Some("accept [MASK]".to_string()),
                },
                DecisionOption {
                    label: "no".to_string(),
                    description: None,
                },
            ],
        );

        assert_eq!(prompt.matches("[MASK]").count(), 2);
        assert!(!prompt.contains("question [MASK]"));
        assert!(!prompt.contains("state [MASK]"));
    }

    #[test]
    fn choice_deserialization_uses_btree_order() {
        let criteria = BTreeMap::from([
            ("b".to_string(), serde_json::json!("second")),
            ("a".to_string(), serde_json::json!("first")),
        ]);
        let question = DecisionQuestion::Choice {
            instructions: "pick".to_string(),
            criteria,
        };
        assert_eq!(question.options()[0].label, "a");
    }

    #[test]
    fn openai_decision_request_accepts_text_and_text_messages() {
        let request: OpenAIDecisionRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-6-luna",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Inspect this item."},
                    {"type": "input_text", "text": "It arrived today."}
                ]
            }],
            "questions": [{
                "type": "predicate",
                "name": "arrived",
                "instructions": "Has the item arrived?"
            }]
        }))
        .unwrap();

        assert_eq!(
            request.input.into_text().unwrap(),
            "Inspect this item.\nIt arrived today."
        );
        assert_eq!(request.questions[0].name(), "arrived");
    }

    #[test]
    fn openai_questions_convert_to_model_decision_types() {
        let choice: OpenAIDecisionQuestion = serde_json::from_value(serde_json::json!({
            "type": "choice",
            "name": "department",
            "instructions": "Choose a department.",
            "choices": [
                {"value": "billing", "description": "Payments and refunds."},
                {"value": "other", "description": "Everything else."}
            ]
        }))
        .unwrap();
        assert_eq!(choice.to_model_question().unwrap().options().len(), 2);

        let score: OpenAIDecisionQuestion = serde_json::from_value(serde_json::json!({
            "type": "score",
            "name": "severity",
            "instructions": "Rate severity.",
            "levels": [
                {"label": "low", "description": "No loss of function."},
                {"label": "high", "description": "Completely blocked."}
            ]
        }))
        .unwrap();
        assert_eq!(
            score.to_model_question().unwrap().options()[1]
                .description
                .as_deref(),
            Some("high: Completely blocked.")
        );
    }

    #[test]
    fn openai_decision_answers_serialize_with_published_field_names() {
        let response = OpenAIDecisionResponse {
            answers: vec![
                OpenAIDecisionAnswer::Predicate {
                    name: "visible_damage".to_string(),
                    probability: 0.92,
                },
                OpenAIDecisionAnswer::Choice {
                    name: "department".to_string(),
                    choice: "billing".to_string(),
                    probabilities: vec![OpenAIDecisionChoiceProbability {
                        value: "billing".to_string(),
                        probability: 0.95,
                    }],
                    confidence: 0.93,
                },
                OpenAIDecisionAnswer::Score {
                    name: "severity".to_string(),
                    score: 1.1,
                    probabilities: vec![OpenAIDecisionScoreProbability {
                        value: 1,
                        label: "Workaround available".to_string(),
                        probability: 0.7,
                    }],
                    confidence: 0.55,
                },
            ],
        };

        assert_eq!(
            serde_json::to_value(response).unwrap(),
            serde_json::json!({
                "answers": [
                    {"type": "predicate", "name": "visible_damage", "probability": 0.92},
                    {
                        "type": "choice",
                        "name": "department",
                        "choice": "billing",
                        "probabilities": [{"value": "billing", "probability": 0.95}],
                        "confidence": 0.93
                    },
                    {
                        "type": "score",
                        "name": "severity",
                        "score": 1.1,
                        "probabilities": [{"value": 1, "label": "Workaround available", "probability": 0.7}],
                        "confidence": 0.55
                    }
                ]
            })
        );
    }
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
    #[schema(default = "null", example = "null", nullable = true)]
    pub dimensions: Option<usize>,
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

    /// The number of dimensions that the output embeddings should have. If not set, the original
    /// shape of the representation will be returned instead.
    #[schema(default = "null", example = "null", nullable = true)]
    pub dimensions: Option<usize>,
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
