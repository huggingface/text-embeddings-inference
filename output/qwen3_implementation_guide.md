# Qwen3 Reranker Fix - Implementation Guide

## Problem Summary
Qwen3-Reranker returns identical scores (0.7310586) for all inputs because:
1. The Candle backend has hardcoded token IDs that don't match the actual Qwen3 tokenizer
2. The router doesn't format prompts in the specific format Qwen3 reranker expects

## Minimal Solution Overview

### Changes Required:
1. **Candle Backend**: Add dynamic token ID detection
2. **Router**: Add Qwen3-specific prompt formatting
3. **Cargo.toml**: Add serde_json dependency

## Detailed Implementation

### 1. Candle Backend Changes

**File**: `backends/candle/src/models/flash_qwen3.rs`

#### Step 1.1: Add imports at the top of the file
```rust
use serde_json;
use std::path::Path;
```

#### Step 1.2: Add token detection method
Add this method inside the `impl FlashQwen3` block (around line 400):

```rust
/// Detect yes/no token IDs from tokenizer files
fn detect_token_ids(&self) -> candle::Result<(u32, u32)> {
    // Try to read tokenizer.json from model path
    let model_path = std::env::var("MODEL_PATH").unwrap_or_default();
    let tokenizer_path = Path::new(&model_path).join("tokenizer.json");
    
    if tokenizer_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&tokenizer_path) {
            // Look for token IDs in added_tokens or vocab
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                // Check added_tokens first
                if let Some(added_tokens) = json.get("added_tokens") {
                    if let Some(tokens) = added_tokens.as_array() {
                        let mut yes_id = None;
                        let mut no_id = None;
                        
                        for token in tokens {
                            if let Some(content) = token.get("content").and_then(|c| c.as_str()) {
                                if let Some(id) = token.get("id").and_then(|i| i.as_u64()) {
                                    match content {
                                        "yes" => yes_id = Some(id as u32),
                                        "no" => no_id = Some(id as u32),
                                        _ => {}
                                    }
                                }
                            }
                        }
                        
                        if let (Some(yes), Some(no)) = (yes_id, no_id) {
                            tracing::info!("Detected token IDs from tokenizer.json - yes: {}, no: {}", yes, no);
                            return Ok((yes, no));
                        }
                    }
                }
                
                // Check vocab
                if let Some(model) = json.get("model") {
                    if let Some(vocab) = model.get("vocab") {
                        if let Some(vocab_obj) = vocab.as_object() {
                            let yes_id = vocab_obj.get("yes")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as u32);
                            let no_id = vocab_obj.get("no")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as u32);
                                
                            if let (Some(yes), Some(no)) = (yes_id, no_id) {
                                tracing::info!("Detected token IDs from vocab - yes: {}, no: {}", yes, no);
                                return Ok((yes, no));
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Fallback to known Qwen3 defaults
    tracing::warn!("Using default Qwen3 token IDs - yes: 9454, no: 2901");
    Ok((9454u32, 2901u32))
}
```

#### Step 1.3: Replace hardcoded token IDs
Find these lines (around line 598-600):
```rust
// Correct token IDs for Qwen3 (verified from tokenizer)
let yes_id = 9454u32; // "yes" token ID
let no_id = 2901u32; // "no" token ID
```

Replace with:
```rust
// Dynamic token detection for Qwen3
let (yes_id, no_id) = self.detect_token_ids()?;
```

### 2. Router Changes

**File**: `router/src/http/server.rs`

#### Step 2.1: Add helper function
Add this function at the end of the file (before the final closing brace):

```rust
/// Format input for Qwen3 reranker model
fn format_qwen3_rerank_input(query: &str, document: &str) -> (String, String) {
    // Qwen3 reranker expects a specific prompt format
    let instruction = "Judge whether the Document meets the Query, answer with yes or no.";
    
    let formatted_query = format!(
        "<|im_start|>system\n{}\n<|im_start|>user\n<Instruct>: {}\n<Query>: {}\n<Document>: {}\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        instruction,
        instruction,
        query,
        document
    );
    
    // Return formatted query with empty document (all content is in query)
    (formatted_query, String::new())
}
```

#### Step 2.2: Modify rerank function
Find the `rerank_inner` closure (around line 1050):
```rust
let rerank_inner = move |query: String, text: String, truncate: bool, infer: Infer| async move {
    let permit = infer.acquire_permit().await;
    let response = infer
        .predict(
            (query, text),
            truncate,
            req.truncation_direction.into(),
            req.raw_scores,
            permit,
        )
```

You need to modify it to check if we're using a Qwen3 reranker and format accordingly. The challenge is that `info` is not available in the closure. We need to capture it:

First, before the `rerank_inner` definition, add:
```rust
let is_qwen3_reranker = info.model_id.contains("qwen") && info.model_id.contains("rerank");
```

Then modify the closure:
```rust
let rerank_inner = move |query: String, text: String, truncate: bool, infer: Infer| async move {
    let permit = infer.acquire_permit().await;
    
    // Format input for Qwen3 reranker if needed
    let input = if is_qwen3_reranker {
        format_qwen3_rerank_input(&query, &text)
    } else {
        (query, text)
    };
    
    let response = infer
        .predict(
            input,
            truncate,
            truncation_direction, // Note: not req.truncation_direction since we're in the closure
            raw_scores,           // Note: not req.raw_scores
            permit,
        )
```

### 3. Cargo.toml Changes

**File**: `backends/candle/Cargo.toml`

Add to the `[dependencies]` section:
```toml
serde_json = "1.0"
```

## Building and Testing

### Build Commands:
```bash
# Build Candle backend
cd backends/candle
cargo build --release

# Build router
cd ../../router
cargo build --release

# Build the full project
cd ..
cargo build --release
```

### Testing:
1. Start TEI with a Qwen3 reranker model:
```bash
text-embeddings-router --model-id Alibaba-NLP/gte-Qwen2.5-7B-instruct-reranker --port 8080
```

2. Test with curl:
```bash
curl -X POST http://localhost:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "texts": [
      "Machine learning is a subset of artificial intelligence.",
      "The weather today is sunny and warm.",
      "Deep learning uses neural networks to learn patterns."
    ]
  }'
```

### Expected Results:
- Scores should vary based on relevance (not all 0.7310586)
- The most relevant document should have the highest score
- Logs should show dynamic token ID detection

## Troubleshooting

### If token detection fails:
1. Check that `MODEL_PATH` environment variable is set correctly
2. Verify `tokenizer.json` exists in the model directory
3. Check logs for the fallback message

### If scores are still constant:
1. Verify the prompt formatting is being applied (add debug logging)
2. Check that the model name contains both "qwen" and "rerank"
3. Ensure the Candle backend was rebuilt and is being used

## Notes

1. This solution is minimal and focused on fixing the immediate issue
2. For production, consider adding:
   - Token ID caching for performance
   - More robust error handling
   - Support for other reranker models
   - Metrics and monitoring
3. The Python backend already has correct implementation and can be used as reference