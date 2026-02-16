// aggregation_strategy (str, optional, defaults to "none") — The strategy to fuse (or not) tokens based on the model prediction.
// "none" : Will simply not do any aggregation and simply return raw results from the model
// "simple" : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{"word": ABC, "entity": "TAG"}, {"word": "D", "entity": "TAG2"}, {"word": "E", "entity": "TAG2"}] Notice that two consecutive B tags will end up as different entities. On word based languages, we might end up splitting words undesirably : Imagine Microsoft being tagged as [{"word": "Micro", "entity": "ENTERPRISE"}, {"word": "soft", "entity": "NAME"}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages that support that meaning, which is basically tokens separated by a space). These mitigations will only work on real words, "New york" might still be tagged with two different entities.
// "first" : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. Words will simply use the tag of the first token of the word when there is ambiguity.
// "average" : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. scores will be averaged first across tokens, and then the maximum label is applied.
// "max" : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. Word entity will simply be the token with the maximum score.

#[derive(serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub(crate) enum AggregationStrategy {
    #[default]
    None,
    Simple,
    First,
    Average,
    Max,
}

pub fn apply_aggregation(
    sentence: &str,
    predictions: Vec<crate::http::types::TokenPrediction>,
    strategy: &AggregationStrategy,
    id2label: &std::collections::HashMap<String, String>,
    ignore_labels: &[String],
) -> Vec<crate::http::types::TokenPrediction> {
    // First, filter out special tokens for ALL strategies (like HF does)
    let filtered_predictions: Vec<crate::http::types::TokenPrediction> = predictions
        .into_iter()
        .filter(|prediction| !is_special_token(&prediction.token, prediction.start, prediction.end))
        .collect();

    let mut results = match strategy {
        AggregationStrategy::None => {
            // For None strategy, we need to compute the best label using HF's approach
            // Build score vector in id2label index order
            let mut labels: Vec<(usize, String)> = id2label
                .iter()
                .filter_map(|(k, v)| k.parse::<usize>().ok().map(|i| (i, v.clone())))
                .collect();
            labels.sort_by_key(|(i, _)| *i);

            filtered_predictions
                .into_iter()
                .map(|mut prediction| {
                    // Compute best label using HF's approach: argmax over score vector in index order
                    let scores: Vec<f32> = labels
                        .iter()
                        .map(|(_, label)| *prediction.results.get(label).unwrap_or(&0.0))
                        .collect();

                    let entity_idx = scores
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    let best_label = id2label
                        .get(&entity_idx.to_string())
                        .cloned()
                        .unwrap_or_else(|| format!("UNKNOWN_{}", entity_idx));

                    // Update results to contain only the best label (HF-like format)
                    prediction.results =
                        std::collections::HashMap::from([(best_label, scores[entity_idx])]);
                    prediction
                })
                .collect()
        }
        AggregationStrategy::Simple => simple_aggregation(sentence, filtered_predictions, id2label),
        AggregationStrategy::First => first_aggregation(sentence, filtered_predictions, id2label),
        AggregationStrategy::Average => {
            average_aggregation(sentence, filtered_predictions, id2label)
        }
        AggregationStrategy::Max => max_aggregation(sentence, filtered_predictions, id2label),
    };

    // Filter out ignored labels (like Hugging Face does)
    if !ignore_labels.is_empty() {
        results.retain(|prediction| {
            if let Some((best_label, _)) = prediction
                .results
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                !ignore_labels.contains(best_label)
            } else {
                true
            }
        });
    }

    results
}

/// Helper structure to represent pre-entities for aggregation
#[derive(Debug, Clone)]
struct PreEntity {
    word: String,
    scores: Vec<f32>,
    start: Option<usize>,
    end: Option<usize>,
    index: usize,
    is_subword: bool,
}

/// Safe detokenization from tokens (BERT-friendly version)
fn detok(tokens: impl Iterator<Item = String>) -> String {
    let mut out = String::new();
    for (i, tok) in tokens.enumerate() {
        if let Some(rest) = tok.strip_prefix("##") {
            out.push_str(rest);
            continue;
        }
        if let Some(rest) = tok.strip_prefix("Ġ").or_else(|| tok.strip_prefix("▁")) {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(rest);
            continue;
        }
        // BERT/plain token (word start): put a space if not first and previous wasn't empty
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&tok);
    }
    out
}

/// Detect if a token is a special token that should be filtered out
fn is_special_token(token: &str, start: Option<usize>, end: Option<usize>) -> bool {
    // Hard-coded special token patterns
    if token == "[CLS]"
        || token == "[SEP]"
        || token == "[PAD]"
        || token == "[MASK]"
        || token == "[UNK]"
    {
        return true;
    }

    // Length-based detection: tokens with zero length are special tokens
    if let (Some(start), Some(end)) = (start, end) {
        if start >= end {
            return true;
        }
    }

    // Additional common special token patterns
    if token.starts_with("<") && token.ends_with(">") {
        // <s>, </s>, <pad>, etc.
        return true;
    }

    false
}

/// Detect if a token is a subword using marker-only approach (HF-compatible without sentence text)
fn is_subword_token(token: &str, _start: Option<usize>, _end: Option<usize>) -> bool {
    // BERT WordPiece: ## marks continuation
    if token.starts_with("##") {
        return true;
    }

    // RoBERTa/SentencePiece: Ġ/▁ marks word start, so subword = !starts_with("Ġ/▁")
    if token.starts_with('Ġ') || token.starts_with('▁') {
        return false;
    }

    // For all other tokens without markers, treat as word start (not subword)
    // This is the least harmful default without sentence text for whitespace checking
    false
}

/// Gather pre-entities from predictions
fn gather_pre_entities(
    predictions: Vec<crate::http::types::TokenPrediction>,
    id2label: &std::collections::HashMap<String, String>,
) -> Vec<PreEntity> {
    // Build label list in index order: 0..N-1
    let mut labels: Vec<(usize, String)> = id2label
        .iter()
        .filter_map(|(k, v)| k.parse::<usize>().ok().map(|i| (i, v.clone())))
        .collect();
    labels.sort_by_key(|(i, _)| *i);

    predictions
        .into_iter()
        .enumerate()
        .map(|(idx, p)| {
            let scores: Vec<f32> = labels
                .iter()
                .map(|(_, label)| *p.results.get(label).unwrap_or(&0.0))
                .collect();

            let is_subword = is_subword_token(&p.token, p.start, p.end);

            PreEntity {
                word: p.token,
                scores,
                start: p.start,
                end: p.end,
                index: idx,
                is_subword,
            }
        })
        .collect()
}

/// Get the BIO tag and entity type from a label
fn get_tag(entity_name: &str) -> (String, String) {
    if entity_name.starts_with("B-") {
        ("B".to_string(), entity_name[2..].to_string())
    } else if entity_name.starts_with("I-") {
        ("I".to_string(), entity_name[2..].to_string())
    } else {
        ("I".to_string(), entity_name.to_string())
    }
}

/// Entity structure matching Hugging Face's format
#[derive(Debug, Clone)]
struct Entity {
    entity: String,
    score: f32,
    index: usize,
    word: String,
    start: Option<usize>,
    end: Option<usize>,
}

/// EntityGroup structure matching Hugging Face's format
#[derive(Debug, Clone)]
struct EntityGroup {
    entity_group: String,
    score: f32,
    word: String,
    start: Option<usize>,
    end: Option<usize>,
}

/// Aggregate pre-entities based on the strategy
fn aggregate(
    sentence: &str,
    pre_entities: Vec<PreEntity>,
    aggregation_strategy: &AggregationStrategy,
    id2label: &std::collections::HashMap<String, String>,
) -> Vec<EntityGroup> {
    if matches!(
        aggregation_strategy,
        AggregationStrategy::None | AggregationStrategy::Simple
    ) {
        let mut entities = Vec::new();
        for pre_entity in pre_entities {
            let entity_idx = pre_entity
                .scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let score = pre_entity.scores[entity_idx];

            let entity_label = id2label
                .get(&entity_idx.to_string())
                .cloned()
                .unwrap_or_else(|| format!("UNKNOWN_{}", entity_idx));

            // Use offset slicing for None strategy as well
            let word = match (pre_entity.start, pre_entity.end) {
                (Some(start), Some(end)) if start <= end => sentence
                    .get(start..end)
                    .map(str::to_string)
                    .unwrap_or_else(|| pre_entity.word.clone()),
                _ => pre_entity.word.clone(),
            };

            let entity = Entity {
                entity: entity_label,
                score,
                index: pre_entity.index,
                word,
                start: pre_entity.start,
                end: pre_entity.end,
            };
            entities.push(entity);
        }

        if matches!(aggregation_strategy, AggregationStrategy::None) {
            return entities
                .into_iter()
                .map(|e| EntityGroup {
                    entity_group: e.entity,
                    score: e.score,
                    word: e.word,
                    start: e.start,
                    end: e.end,
                })
                .collect();
        }

        group_entities(sentence, entities)
    } else {
        let word_entities = aggregate_words(sentence, pre_entities, aggregation_strategy, id2label);
        group_entities(sentence, word_entities)
    }
}

/// Aggregate a word using the specified strategy
fn aggregate_word(
    sentence: &str,
    entities: &[PreEntity],
    aggregation_strategy: &AggregationStrategy,
    id2label: &std::collections::HashMap<String, String>,
) -> Entity {
    // Use offset slicing instead of token reconstruction
    let word = match (
        entities.first().and_then(|e| e.start),
        entities.last().and_then(|e| e.end),
    ) {
        (Some(start), Some(end)) if start <= end => sentence
            .get(start..end)
            .map(|s| s.to_string())
            .unwrap_or_else(|| detok(entities.iter().map(|e| e.word.clone()))),
        _ => detok(entities.iter().map(|e| e.word.clone())),
    };

    let (entity, score) = match aggregation_strategy {
        AggregationStrategy::First => {
            if entities.is_empty() {
                return Entity {
                    entity: "UNKNOWN".to_string(),
                    score: 0.0,
                    word: String::new(),
                    start: None,
                    end: None,
                    index: 0,
                };
            }
            let scores = &entities[0].scores;
            let idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let score = scores[idx];
            let entity = id2label
                .get(&idx.to_string())
                .cloned()
                .unwrap_or_else(|| format!("UNKNOWN_{}", idx));
            (entity, score)
        }
        AggregationStrategy::Max => {
            if entities.is_empty() {
                return Entity {
                    entity: "O".to_string(),
                    score: 0.0,
                    word: String::new(),
                    start: None,
                    end: None,
                    index: 0,
                };
            }
            let max_entity = entities
                .iter()
                .max_by(|a, b| {
                    let a_max = a
                        .scores
                        .iter()
                        .filter(|score| score.is_finite())
                        .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(&0.0);
                    let b_max = b
                        .scores
                        .iter()
                        .filter(|score| score.is_finite())
                        .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(&0.0);
                    a_max
                        .partial_cmp(b_max)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(&entities[0]);

            let scores = &max_entity.scores;
            let idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let score = scores[idx];
            let entity = id2label
                .get(&idx.to_string())
                .cloned()
                .unwrap_or_else(|| format!("UNKNOWN_{}", idx));
            (entity, score)
        }
        AggregationStrategy::Average => {
            if entities.is_empty() {
                return Entity {
                    entity: "O".to_string(),
                    score: 0.0,
                    word: String::new(),
                    start: None,
                    end: None,
                    index: 0,
                };
            }
            let mut averaged_scores = Vec::new();
            let num_scores = entities[0].scores.len();
            averaged_scores.resize(num_scores, 0.0);

            for entity in entities {
                for (i, score) in entity.scores.iter().enumerate() {
                    if i < averaged_scores.len() {
                        averaged_scores[i] += score;
                    }
                }
            }

            for score in &mut averaged_scores {
                if !entities.is_empty() {
                    *score /= entities.len() as f32;
                }
            }

            let entity_idx = averaged_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let score = averaged_scores[entity_idx];
            let entity = id2label
                .get(&entity_idx.to_string())
                .cloned()
                .unwrap_or_else(|| format!("UNKNOWN_{}", entity_idx));
            (entity, score)
        }
        _ => ("O".to_string(), 0.0),
    };

    Entity {
        entity,
        score,
        word,
        start: entities.first().and_then(|e| e.start),
        end: entities.last().and_then(|e| e.end),
        index: entities.first().map(|e| e.index).unwrap_or(0),
    }
}

/// Aggregate words for FIRST, AVERAGE, and MAX strategies using offset-based grouping
fn aggregate_words(
    sentence: &str,
    pre_entities: Vec<PreEntity>,
    aggregation_strategy: &AggregationStrategy,
    id2label: &std::collections::HashMap<String, String>,
) -> Vec<Entity> {
    if matches!(
        aggregation_strategy,
        AggregationStrategy::None | AggregationStrategy::Simple
    ) {
        panic!("NONE and SIMPLE strategies are invalid for word aggregation");
    }

    let mut word_entities = Vec::new();
    let mut word_group: Option<Vec<PreEntity>> = None;

    for entity in pre_entities {
        if word_group.is_none() {
            word_group = Some(vec![entity]);
        } else {
            let group = word_group.as_mut().unwrap();
            let last_entity = group.last().unwrap();

            // Check if we should start a new word group based on offsets
            let should_split = match (last_entity.end, entity.start) {
                (Some(last_end), Some(curr_start)) => {
                    if curr_start < last_end {
                        // Overlapping tokens - split
                        true
                    } else if curr_start > last_end {
                        // Gap between tokens - check if it contains whitespace
                        let gap = sentence.get(last_end..curr_start).unwrap_or("");
                        gap.chars().any(|c| c.is_whitespace())
                    } else {
                        // Adjacent tokens - check if this is a subword
                        entity.is_subword
                    }
                }
                _ => {
                    // Missing offsets - fall back to subword detection
                    !entity.is_subword
                }
            };

            if should_split {
                // Start a new word group
                word_entities.push(aggregate_word(
                    sentence,
                    group,
                    aggregation_strategy,
                    id2label,
                ));
                word_group = Some(vec![entity]);
            } else {
                // Continue current word group
                group.push(entity);
            }
        }
    }

    if let Some(group) = word_group {
        word_entities.push(aggregate_word(
            sentence,
            &group,
            aggregation_strategy,
            id2label,
        ));
    }

    word_entities
}

/// Group sub-entities together
fn group_sub_entities(sentence: &str, entities: &[Entity]) -> EntityGroup {
    if entities.is_empty() {
        return EntityGroup {
            entity_group: "O".to_string(),
            score: 0.0,
            word: String::new(),
            start: None,
            end: None,
        };
    }

    let entity = entities[0]
        .entity
        .split('-')
        .last()
        .unwrap_or(&entities[0].entity);

    let mut total_score = 0.0;
    let mut count = 0;
    for entity in entities {
        if entity.score.is_finite() {
            total_score += entity.score;
            count += 1;
        }
    }
    let score = if count > 0 {
        total_score / count as f32
    } else {
        0.0
    };

    // Use offset slicing instead of joining with spaces
    let fallback = entities
        .iter()
        .map(|e| e.word.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let word = match (
        entities.first().and_then(|e| e.start),
        entities.last().and_then(|e| e.end),
    ) {
        (Some(start), Some(end)) if start <= end => sentence
            .get(start..end)
            .map(str::to_string)
            .unwrap_or(fallback),
        _ => fallback,
    };

    EntityGroup {
        entity_group: entity.to_string(),
        score,
        word,
        start: entities[0].start,
        end: entities[entities.len() - 1].end,
    }
}

/// Group entities according to BIO tagging scheme
fn group_entities(sentence: &str, entities: Vec<Entity>) -> Vec<EntityGroup> {
    let mut entity_groups = Vec::new();
    let mut entity_group_disagg = Vec::new();

    for entity in entities {
        if entity_group_disagg.is_empty() {
            entity_group_disagg.push(entity);
            continue;
        }

        let (bi, tag) = get_tag(&entity.entity);
        let (_last_bi, last_tag) = if let Some(last_entity) = entity_group_disagg.last() {
            get_tag(&last_entity.entity)
        } else {
            ("I".to_string(), "".to_string())
        };

        if tag == last_tag && bi != "B" {
            entity_group_disagg.push(entity);
        } else {
            entity_groups.push(group_sub_entities(sentence, &entity_group_disagg));
            entity_group_disagg = vec![entity];
        }
    }

    if !entity_group_disagg.is_empty() {
        entity_groups.push(group_sub_entities(sentence, &entity_group_disagg));
    }

    entity_groups
}

/// Convert EntityGroup back to TokenPrediction for our API
fn entity_group_to_token_prediction(group: EntityGroup) -> crate::http::types::TokenPrediction {
    crate::http::types::TokenPrediction {
        token: group.word,
        token_id: 0,
        start: group.start,
        end: group.end,
        results: std::collections::HashMap::from([(group.entity_group, group.score)]),
    }
}

/// Simple aggregation strategy
fn simple_aggregation(
    sentence: &str,
    predictions: Vec<crate::http::types::TokenPrediction>,
    id2label: &std::collections::HashMap<String, String>,
) -> Vec<crate::http::types::TokenPrediction> {
    let pre_entities = gather_pre_entities(predictions, id2label);
    let entities = aggregate(
        sentence,
        pre_entities,
        &AggregationStrategy::Simple,
        id2label,
    );
    entities
        .into_iter()
        .map(entity_group_to_token_prediction)
        .collect()
}

/// First aggregation strategy
fn first_aggregation(
    sentence: &str,
    predictions: Vec<crate::http::types::TokenPrediction>,
    id2label: &std::collections::HashMap<String, String>,
) -> Vec<crate::http::types::TokenPrediction> {
    let pre_entities = gather_pre_entities(predictions, id2label);
    let entities = aggregate(
        sentence,
        pre_entities,
        &AggregationStrategy::First,
        id2label,
    );
    entities
        .into_iter()
        .map(entity_group_to_token_prediction)
        .collect()
}

/// Average aggregation strategy
fn average_aggregation(
    sentence: &str,
    predictions: Vec<crate::http::types::TokenPrediction>,
    id2label: &std::collections::HashMap<String, String>,
) -> Vec<crate::http::types::TokenPrediction> {
    let pre_entities = gather_pre_entities(predictions, id2label);
    let entities = aggregate(
        sentence,
        pre_entities,
        &AggregationStrategy::Average,
        id2label,
    );
    entities
        .into_iter()
        .map(entity_group_to_token_prediction)
        .collect()
}

/// Max aggregation strategy
fn max_aggregation(
    sentence: &str,
    predictions: Vec<crate::http::types::TokenPrediction>,
    id2label: &std::collections::HashMap<String, String>,
) -> Vec<crate::http::types::TokenPrediction> {
    let pre_entities = gather_pre_entities(predictions, id2label);
    let entities = aggregate(sentence, pre_entities, &AggregationStrategy::Max, id2label);
    entities
        .into_iter()
        .map(entity_group_to_token_prediction)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_subword_token_bert() {
        // BERT WordPiece tokens - marker-only approach
        assert!(!is_subword_token("microsoft", None, None)); // Regular word start
        assert!(is_subword_token("##soft", None, None)); // Continuation
        assert!(!is_subword_token("is", None, None)); // Regular word start
        assert!(!is_subword_token("great", None, None)); // Regular word start
        assert!(is_subword_token("##ing", None, None)); // Continuation
    }

    #[test]
    fn test_is_subword_token_roberta() {
        // RoBERTa BPE tokens - marker-only approach
        assert!(!is_subword_token("Ġmicrosoft", None, None)); // Word start with Ġ
        assert!(!is_subword_token("soft", None, None)); // No marker = word start
        assert!(!is_subword_token("Ġis", None, None)); // Word start with Ġ
    }

    #[test]
    fn test_is_subword_token_sentencepiece() {
        // SentencePiece tokens - marker-only approach
        assert!(!is_subword_token("▁microsoft", None, None)); // Word start with ▁
        assert!(!is_subword_token("soft", None, None)); // No marker = word start
        assert!(!is_subword_token("▁is", None, None)); // Word start with ▁
    }

    #[test]
    fn test_is_subword_token_edge_cases() {
        // Test edge cases with marker-only approach
        assert!(!is_subword_token("hello", None, None)); // Regular word
        assert!(!is_subword_token("world", None, None)); // Regular word
        assert!(!is_subword_token(".", None, None)); // Punctuation = word start
        assert!(!is_subword_token(",", None, None)); // Punctuation = word start
        assert!(is_subword_token("##ly", None, None)); // BERT continuation
        assert!(!is_subword_token("Ġtest", None, None)); // RoBERTa word start
        assert!(!is_subword_token("▁test", None, None)); // SentencePiece word start
    }

    #[test]
    fn test_detok_bert() {
        let tokens = vec![
            "micro".to_string(),
            "##soft".to_string(),
            "is".to_string(),
            "great".to_string(),
        ];
        let result = detok(tokens.into_iter());
        assert_eq!(result, "microsoft is great");
    }

    #[test]
    fn test_word_grouping() {
        // Test that BERT tokens are grouped correctly into words
        let tokens = vec![
            "micro".to_string(),
            "##soft".to_string(), // Should be one word: "microsoft"
            "is".to_string(),     // Should be one word: "is"
            "very".to_string(),
            "##great".to_string(), // Should be one word: "verygreat"
        ];

        let mut word_groups = Vec::new();
        let mut current_group: Option<Vec<String>> = None;

        for token in tokens {
            if is_subword_token(&token, None, None) {
                if let Some(group) = current_group.as_mut() {
                    group.push(token);
                } else {
                    current_group = Some(vec![token]);
                }
            } else {
                if let Some(group) = current_group {
                    word_groups.push(group);
                }
                current_group = Some(vec![token]);
            }
        }

        if let Some(group) = current_group {
            word_groups.push(group);
        }

        // Convert groups to words
        let words: Vec<String> = word_groups
            .iter()
            .map(|group| detok(group.iter().cloned()))
            .collect();

        assert_eq!(words, vec!["microsoft", "is", "verygreat"]);
    }

    #[test]
    fn test_special_token_detection() {
        // Test hard-coded special tokens
        assert!(is_special_token("[CLS]", Some(0), Some(5)));
        assert!(is_special_token("[SEP]", Some(10), Some(15)));
        assert!(is_special_token("[PAD]", Some(20), Some(25)));
        assert!(is_special_token("[UNK]", Some(30), Some(35)));

        // Test length-based detection (zero-length tokens)
        assert!(is_special_token("something", Some(10), Some(10))); // start == end
        assert!(is_special_token("something", Some(15), Some(10))); // start > end

        // Test angle bracket tokens
        assert!(is_special_token("<s>", Some(0), Some(3)));
        assert!(is_special_token("</s>", Some(5), Some(9)));
        assert!(is_special_token("<pad>", Some(10), Some(15)));

        // Test normal tokens (should not be filtered)
        assert!(!is_special_token("hello", Some(0), Some(5)));
        assert!(!is_special_token("world", Some(6), Some(11)));
        assert!(!is_special_token("John", Some(12), Some(16)));

        // Test tokens without position info (should not be filtered unless hard-coded)
        assert!(!is_special_token("hello", None, None));
        assert!(is_special_token("[CLS]", None, None)); // Hard-coded still works
    }

    #[test]
    fn test_apply_aggregation_filters_special_tokens() {
        // Create mock predictions including special tokens
        let predictions = vec![
            crate::http::types::TokenPrediction {
                token: "[CLS]".to_string(),
                token_id: 0,
                start: Some(0),
                end: Some(5),
                results: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("O".to_string(), 0.99);
                    map
                },
            },
            crate::http::types::TokenPrediction {
                token: "John".to_string(),
                token_id: 1,
                start: Some(6),
                end: Some(10),
                results: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("B-PER".to_string(), 0.95);
                    map.insert("O".to_string(), 0.05);
                    map
                },
            },
            crate::http::types::TokenPrediction {
                token: "[SEP]".to_string(),
                token_id: 2,
                start: Some(11),
                end: Some(15),
                results: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("O".to_string(), 0.98);
                    map
                },
            },
        ];

        // Create mock id2label
        let mut id2label = std::collections::HashMap::new();
        id2label.insert("0".to_string(), "O".to_string());
        id2label.insert("1".to_string(), "B-PER".to_string());

        // Test None strategy (should filter special tokens and compute best label)
        let results = apply_aggregation(
            "Hello John",
            predictions.clone(),
            &AggregationStrategy::None,
            &id2label,
            &["O".to_string()],
        );

        // Should only contain "John" with B-PER, not "[CLS]" or "[SEP]"
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].token, "John");
        assert_eq!(results[0].results.len(), 1);
        assert!(results[0].results.contains_key("B-PER"));

        // Test Simple strategy (should also filter special tokens)
        let results_simple = apply_aggregation(
            "Hello John",
            predictions.clone(),
            &AggregationStrategy::Simple,
            &id2label,
            &["O".to_string()],
        );

        // Should also only contain "John" entity
        assert_eq!(results_simple.len(), 1);
        assert_eq!(results_simple[0].token, "John");
    }

    #[test]
    fn test_detok_sentencepiece() {
        let tokens = vec![
            "▁micro".to_string(),
            "soft".to_string(),
            "▁is".to_string(),
            "▁great".to_string(),
        ];
        let results = detok(tokens.into_iter());
        assert_eq!(results, "micro soft is great");
    }

    #[test]
    fn test_detok_mixed() {
        let tokens = vec!["i".to_string(), "like".to_string(), "pizza".to_string()];
        let result = detok(tokens.into_iter());
        assert_eq!(result, "i like pizza");
    }
}
