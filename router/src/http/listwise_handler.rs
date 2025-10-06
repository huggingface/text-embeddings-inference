//! Listwise reranking HTTP handler (Milestone 8)
//!
//! This module implements the complete listwise reranking pipeline with exact
//! modeling.py parity for numerical consistency.

use crate::http::types::{Rank, RerankRequest, RerankResponse};
use crate::listwise::math::{cosine_similarity, weighted_average};
use crate::{AppState, ErrorResponse, ErrorType};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use std::time::Instant;
use text_embeddings_backend::ListwiseBlockInput;
use text_embeddings_core::prompt::build_jina_v3_prompt;
use text_embeddings_core::tokenization::{
    encode_listwise, truncate_texts, validate_special_tokens,
};

/// Listwise reranking handler with modeling.py parity
///
/// # Algorithm (from modeling.py)
/// 1. Truncate: query to 512 tokens, each doc to 2048 tokens
/// 2. Block construction: capacity = max_len - 2*query_len, split when:
///    - Block has 125 docs, OR
///    - Remaining capacity <= 2048
/// 3. Per-block processing: prompt → tokenize → validate → infer
/// 4. Query embedding: weighted average across blocks (weights = doc_count)
/// 5. Final scores: cosine_similarity(query_emb, each_doc_emb)
#[tracing::instrument(skip_all, fields(query_len, num_docs, num_blocks, total_time_ms,))]
pub async fn rerank_listwise(
    State(state): State<AppState>,
    Json(req): Json<RerankRequest>,
) -> Result<(HeaderMap, Json<RerankResponse>), (StatusCode, Json<ErrorResponse>)> {
    let start_time = Instant::now();

    // ─────────────────────────────────────────────────────────────
    // 1. INPUT VALIDATION (Early exit before backend calls)
    // ─────────────────────────────────────────────────────────────
    if req.texts.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "texts cannot be empty".to_string(),
                error_type: ErrorType::Empty,
            }),
        ));
    }

    let config = &state.listwise_config;

    if req.texts.len() > config.max_documents_per_request {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Too many documents: {} (max: {})",
                    req.texts.len(),
                    config.max_documents_per_request
                ),
                error_type: ErrorType::Validation,
            }),
        ));
    }

    for (i, doc) in req.texts.iter().enumerate() {
        if doc.len() > config.max_document_length_bytes {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "Document {} exceeds max length: {} > {} bytes",
                        i,
                        doc.len(),
                        config.max_document_length_bytes
                    ),
                    error_type: ErrorType::Validation,
                }),
            ));
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 2. TEXT TRUNCATION (modeling.py _truncate_texts)
    // ─────────────────────────────────────────────────────────────
    let tokenizer = &*state.tokenizer;

    let (truncated_query, truncated_docs, doc_token_lengths, query_token_length) =
        truncate_texts(tokenizer, &req.query, &req.texts, 512, 2048).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Tokenization failed: {}", e),
                    error_type: ErrorType::Tokenizer,
                }),
            )
        })?;

    let model_max_length = state.info.max_input_length;

    // ─────────────────────────────────────────────────────────────
    // 3. RANDOM ORDERING (if configured)
    // ─────────────────────────────────────────────────────────────
    // Combine data into tuples: (original_index, doc, token_length)
    let mut process_items: Vec<(usize, String, usize)> = truncated_docs
        .into_iter()
        .zip(doc_token_lengths.into_iter())
        .enumerate()
        .map(|(idx, (doc, len))| (idx, doc, len))
        .collect();

    // Apply random shuffle if configured (using ChaCha8Rng for reproducibility)
    if config.ordering == crate::strategy::RerankOrdering::Random {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = match config.random_seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };

        process_items.shuffle(&mut rng);

        tracing::info!(
            "Applied random ordering to {} documents (seed: {:?})",
            process_items.len(),
            config.random_seed
        );
    }

    // Extract shuffled/ordered data
    let doc_original_indices: Vec<usize> = process_items.iter().map(|(idx, _, _)| *idx).collect();
    let truncated_docs: Vec<String> = process_items
        .iter()
        .map(|(_, doc, _)| doc.clone())
        .collect();
    let doc_token_lengths: Vec<usize> = process_items.iter().map(|(_, _, len)| *len).collect();

    // ─────────────────────────────────────────────────────────────
    // 4. BLOCK CONSTRUCTION (modeling.py algorithm)
    // ─────────────────────────────────────────────────────────────
    const MAX_BLOCK_SIZE: usize = 125;
    const LENGTH_CAPACITY_MARGIN: usize = 2048;

    let initial_capacity = model_max_length.saturating_sub(2 * query_token_length);
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let mut current_block: Vec<usize> = Vec::new();
    let mut length_capacity = initial_capacity;

    for (idx, &doc_len) in doc_token_lengths.iter().enumerate() {
        current_block.push(idx);
        length_capacity = length_capacity.saturating_sub(doc_len);

        // Split condition: max docs reached OR capacity exhausted
        if current_block.len() >= MAX_BLOCK_SIZE || length_capacity <= LENGTH_CAPACITY_MARGIN {
            blocks.push(current_block);
            current_block = Vec::new();
            length_capacity = initial_capacity;
        }
    }

    // Add remaining documents as final block
    if !current_block.is_empty() {
        blocks.push(current_block);
    }

    tracing::Span::current().record("num_blocks", blocks.len());
    tracing::Span::current().record("num_docs", req.texts.len());

    // ─────────────────────────────────────────────────────────────
    // 4. BLOCK PROCESSING & ACCUMULATION
    // ─────────────────────────────────────────────────────────────
    let embed_token_id = tokenizer.token_to_id("<|embed_token|>").ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Missing <|embed_token|> in tokenizer".to_string(),
                error_type: ErrorType::Tokenizer,
            }),
        )
    })?;

    let rerank_token_id = tokenizer.token_to_id("<|rerank_token|>").ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Missing <|rerank_token|> in tokenizer".to_string(),
                error_type: ErrorType::Tokenizer,
            }),
        )
    })?;

    // Storage for aggregation
    let mut all_doc_embeddings: Vec<Option<Vec<f32>>> = vec![None; req.texts.len()];
    let mut all_query_embeddings: Vec<Vec<f32>> = Vec::with_capacity(blocks.len());
    let mut block_weights: Vec<f32> = Vec::with_capacity(blocks.len());

    for block_indices in blocks {
        let block_start = Instant::now();

        // Build doc refs for this block (use truncated strings)
        let block_docs: Vec<&str> = block_indices
            .iter()
            .map(|&idx| truncated_docs[idx].as_str())
            .collect();

        // Build prompt with truncated texts
        let prompt =
            build_jina_v3_prompt(&truncated_query, &block_docs, config.instruction.as_deref());

        // Tokenize
        let encoding =
            encode_listwise(tokenizer, &prompt, Some(model_max_length)).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Encoding failed: {}", e),
                        error_type: ErrorType::Tokenizer,
                    }),
                )
            })?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        // Validate special tokens (prevent index OOB panics)
        validate_special_tokens(
            &input_ids,
            embed_token_id,
            rerank_token_id,
            block_docs.len(),
        )
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST, // 400 - this is input validation failure
                Json(ErrorResponse {
                    error: format!("Special token validation failed: {}", e),
                    error_type: ErrorType::Validation,
                }),
            )
        })?;

        // Backend inference
        let block_input = ListwiseBlockInput {
            input_ids,
            attention_mask,
            embed_token_id,
            rerank_token_id,
            doc_count: block_docs.len(),
        };

        let block_output = state
            .infer
            .embed_listwise_block(block_input)
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Inference failed: {}", e),
                        error_type: ErrorType::Backend,
                    }),
                )
            })?;

        // Record metrics
        let block_duration = block_start.elapsed();
        metrics::histogram!("tei_lbnl_ms_per_group").record(block_duration.as_millis() as f64);
        metrics::histogram!("tei_lbnl_seq_tokens").record(encoding.len() as f64);
        metrics::histogram!("tei_lbnl_group_size").record(block_docs.len() as f64);

        if block_duration.as_millis() > config.block_timeout_ms as u128 {
            metrics::counter!("tei_lbnl_block_timeout_total").increment(1);
        }

        // Compute block-level scores for weight calculation (modeling.py parity)
        let mut block_scores = Vec::with_capacity(block_docs.len());
        for doc_emb in &block_output.doc_embeddings {
            let score = cosine_similarity(&block_output.query_embedding, doc_emb).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Block score computation failed: {}", e),
                        error_type: ErrorType::Backend,
                    }),
                )
            })?;
            block_scores.push(score);
        }

        // Weight = max((1 + scores) / 2.0) - normalize [-1,1] to [0,1] then take max
        // This matches modeling.py line ~180
        let weight = block_scores
            .iter()
            .map(|&s| (1.0 + s) / 2.0)
            .fold(f32::NEG_INFINITY, f32::max);

        // Store query embedding and weight for final aggregation
        all_query_embeddings.push(block_output.query_embedding);
        block_weights.push(weight);

        // Store document embeddings (map back to original indices)
        // block_indices contains shuffled indices, doc_original_indices maps them to original
        for (i, &shuffled_idx) in block_indices.iter().enumerate() {
            let original_idx = doc_original_indices[shuffled_idx];
            all_doc_embeddings[original_idx] = Some(block_output.doc_embeddings[i].clone());
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 5. QUERY EMBEDDING AGGREGATION (BLOCKER FIX)
    // ─────────────────────────────────────────────────────────────
    // Use weighted average where weights = max((1 + block_scores) / 2)
    // This matches modeling.py line ~180

    // Safety: Handle zero-weight case (all blocks have max_score = -1.0)
    const MIN_WEIGHT: f32 = 1e-6;
    let total_weight: f32 = block_weights.iter().sum();
    if total_weight < MIN_WEIGHT {
        // All blocks had terrible scores - use equal weighting as fallback
        tracing::warn!(
            "All block weights near zero (total={}), using equal weighting",
            total_weight
        );
        block_weights.iter_mut().for_each(|w| *w = 1.0);
    }

    let final_query_embedding =
        weighted_average(&all_query_embeddings, &block_weights).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to aggregate query embeddings: {}", e),
                    error_type: ErrorType::Backend,
                }),
            )
        })?;

    // ─────────────────────────────────────────────────────────────
    // 6. FINAL SCORE COMPUTATION
    // ─────────────────────────────────────────────────────────────
    let mut ranks = Vec::with_capacity(req.texts.len());

    for (idx, doc_emb_opt) in all_doc_embeddings.iter().enumerate() {
        if let Some(doc_emb) = doc_emb_opt {
            // cosine_similarity handles L2 normalization internally
            let score = cosine_similarity(&final_query_embedding, doc_emb).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Cosine similarity failed: {}", e),
                        error_type: ErrorType::Backend,
                    }),
                )
            })?;

            // NaN check
            if score.is_nan() {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Score is NaN for document {}", idx),
                        error_type: ErrorType::Backend,
                    }),
                ));
            }

            let text = if req.return_text {
                Some(req.texts[idx].clone())
            } else {
                None
            };

            ranks.push(Rank {
                index: idx,
                text,
                score,
            });
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 7. SORT & RESPONSE
    // ─────────────────────────────────────────────────────────────
    // Sort descending by score
    ranks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let total_time = start_time.elapsed();
    tracing::Span::current().record("total_time_ms", total_time.as_millis() as f64);

    // Headers for response metadata
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-listwise-blocks",
        block_weights.len().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-total-time-ms",
        total_time.as_millis().to_string().parse().unwrap(),
    );

    Ok((headers, Json(RerankResponse(ranks))))
}
