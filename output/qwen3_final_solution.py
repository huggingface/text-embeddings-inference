"""
Qwen3 CausalLM Reranker - MINIMAL Final Solution
================================================

This solution addresses the critical issue where Qwen3-Reranker returns identical scores (0.7310586)
for all inputs due to:
1. Hardcoded token IDs in Candle backend
2. Missing prompt formatting in router

MINIMAL PATCHES REQUIRED:

====================
PATCH 1: Candle Backend - Dynamic Token Detection
====================

File: backends/candle/src/models/flash_qwen3.rs

Replace lines 598-600 (the hardcoded token IDs) with dynamic detection:

```rust
// OLD CODE (REMOVE):
// Correct token IDs for Qwen3 (verified from tokenizer)
let yes_id = 9454u32; // "yes" token ID  
let no_id = 2901u32; // "no" token ID

// NEW CODE (ADD):
// Dynamic token detection for Qwen3
let (yes_id, no_id) = self.detect_token_ids()?;
```

Add this method to the FlashQwen3 impl block (around line 400):

```rust
impl FlashQwen3 {
    /// Detect yes/no token IDs from tokenizer files
    fn detect_token_ids(&self) -> candle::Result<(u32, u32)> {
        // Try to read tokenizer.json from model path
        let model_path = std::env::var("MODEL_PATH").unwrap_or_default();
        let tokenizer_path = std::path::Path::new(&model_path).join("tokenizer.json");
        
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
}
```

====================
PATCH 2: Router - Add Qwen3 Prompt Formatting
====================

File: router/src/http/server.rs

In the rerank function, replace the direct tuple passing with formatted prompt:

```rust
// Find this code around line 1050-1060:
let rerank_inner = move |query: String, text: String, truncate: bool, infer: Infer| async move {
    let permit = infer.acquire_permit().await;
    let response = infer
        .predict(
            (query, text),  // <-- REPLACE THIS LINE
            truncate,
            req.truncation_direction.into(),
            req.raw_scores,
            permit,
        )

// Replace with:
let rerank_inner = move |query: String, text: String, truncate: bool, infer: Infer| async move {
    let permit = infer.acquire_permit().await;
    
    // Format input for Qwen3 reranker if needed
    let input = if info.model_id.contains("qwen") && info.model_id.contains("rerank") {
        format_qwen3_rerank_input(&query, &text)
    } else {
        (query, text)
    };
    
    let response = infer
        .predict(
            input,
            truncate,
            req.truncation_direction.into(),
            req.raw_scores,
            permit,
        )
```

Add this helper function at the end of the file (before the last closing brace):

```rust
/// Format input for Qwen3 reranker model
fn format_qwen3_rerank_input(query: &str, document: &str) -> (String, String) {
    // Qwen3 reranker expects a specific prompt format
    let instruction = "Judge whether the Document meets the Query, answer with yes or no.";
    
    let formatted_query = format!(
        "<|im_start|>system\\n{}\\n<|im_start|>user\\n<Instruct>: {}\\n<Query>: {}\\n<Document>: {}\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n",
        instruction,
        instruction,
        query,
        document
    );
    
    // Return formatted query with empty document (all content is in query)
    (formatted_query, String::new())
}
```

====================
PATCH 3: Add serde_json dependency to Candle backend
====================

File: backends/candle/Cargo.toml

Add serde_json to dependencies if not already present:

```toml
[dependencies]
serde_json = "1.0"
```

====================
VERIFICATION STEPS
====================

1. Apply the patches above
2. Rebuild the Candle backend: `cd backends/candle && cargo build --release`
3. Rebuild the router: `cd router && cargo build --release`
4. Test with a Qwen3 reranker model

Expected behavior:
- Token IDs will be detected dynamically from tokenizer files
- Prompts will be formatted correctly for Qwen3 
- Scores will vary based on query-document relevance (not constant 0.7310586)

====================
IMPLEMENTATION NOTES
====================

1. This is the MINIMAL solution that fixes the immediate issue
2. The token detection will first try to read from tokenizer.json
3. If detection fails, it falls back to the known Qwen3 defaults
4. The router only formats prompts for models with "qwen" and "rerank" in the name
5. No complex caching or monitoring is added to keep changes minimal

The solution prioritizes:
- Immediate fix for the constant score issue
- Minimal code changes
- Compatibility with existing TEI infrastructure
- Working with Candle backend on Metal

For production deployment, consider adding:
- More robust error handling
- Token ID caching for performance
- Metrics and monitoring
- Support for other reranker models
"""