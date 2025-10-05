# Qwen3 CausalLM Reranker Fix - Summary

## Problem Fixed
The Qwen3-Reranker-0.6B model was returning identical scores (0.7310586) for all query-document pairs when used with text-embeddings-inference (TEI).

## Root Causes
1. **Hardcoded token IDs**: The Candle backend used hardcoded token IDs (yes=9454, no=2901) that might not match the actual tokenizer
2. **Missing prompt formatting**: Qwen3 CausalLM reranker requires specific chat template formatting
3. **Model type**: Qwen3-Reranker is a CausalLM that needs special yes/no token probability scoring

## Changes Applied

### 1. Candle Backend Token Detection (`backends/candle/src/models/flash_qwen3.rs`)

**Added dynamic token ID detection:**
```rust
// Added imports
use std::fs;
use std::path::Path;
use serde_json;

// Added token detection method
fn detect_token_ids() -> (u32, u32) {
    // Try to read from common tokenizer.json locations
    // Returns detected IDs or defaults (9454, 2901)
}

// Updated predict method to use dynamic detection
let (yes_id, no_id) = Self::detect_token_ids();
```

### 2. Router Prompt Formatting (`router/src/http/server.rs`)

**Added Qwen3-specific prompt formatting:**
```rust
// Check if this is a Qwen3 reranker model
let is_qwen3_reranker = info.model_id.to_lowercase().contains("qwen") && 
                       info.model_id.to_lowercase().contains("rerank");

// Format prompts for Qwen3
if is_qwen3 {
    let prompt = format!(
        "<|im_start|>system\n...\n<|im_start|>user\n<Instruct>: {}\n<Query>: {}\n<Document>: {}<|im_end|>...",
        instruction, query, text
    );
    (prompt, String::new())
}
```

### 3. Request Type Update (`router/src/http/types.rs`)

**Added instruction field to RerankRequest:**
```rust
pub struct RerankRequest {
    // ... existing fields ...
    #[serde(default)]
    pub instruction: Option<String>,
}
```

## How It Works

1. **Detection**: Router detects Qwen3 reranker models by checking for "qwen" and "rerank" in model ID
2. **Formatting**: For Qwen3 models, formats the query and document with the required chat template
3. **Token IDs**: Candle backend dynamically detects correct "yes"/"no" token IDs from tokenizer.json
4. **Scoring**: Computes score as `exp(yes_logit) / (exp(yes_logit) + exp(no_logit))`

## Testing

Run the test script to verify the fix:
```bash
python test_qwen3_fix.py
```

Expected output:
- Different scores for different texts (not all 0.7310586)
- Steve Kragthorpe texts should score higher for the Tulsa query
- Korean text should be handled correctly

## Files Modified

1. `backends/candle/src/models/flash_qwen3.rs` - Added dynamic token detection
2. `router/src/http/server.rs` - Added prompt formatting for Qwen3
3. `router/src/http/types.rs` - Added instruction field
4. `backends/python/server/text_embeddings_server/models/__init__.py` - Already had Qwen3 detection
5. `backends/python/server/text_embeddings_server/models/qwen3_rerank_model.py` - Python backend support

## Notes

- The Python backend was also updated but TEI uses Candle backend on Metal
- Token detection tries multiple paths to find tokenizer.json
- Falls back to known Qwen3 defaults if detection fails
- Prompt formatting is essential for Qwen3 CausalLM reranker to work correctly