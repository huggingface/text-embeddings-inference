# Qwen3 CausalLM Reranker - Full Context

## Original Problem
- Qwen3-Reranker returns identical scores (0.7310586) for all inputs
- Root cause: Hardcoded token IDs in Candle backend + missing prompt formatting
- TEI uses Candle backend on Metal, not Python backend

## Solution Evolution

### V1 (Problem-Solver)
- Identified need for dynamic token detection
- Proposed prompt formatting in router
- Basic implementation for both backends

### V2 (Solution-Improver)  
- Added robust 4-method token detection
- Implemented caching and performance optimizations
- Added monitoring, error handling, circuit breakers
- Production-ready features

### Verification Results

#### First Verification
- Solution correctly addresses the issue
- Minor gaps: cache algorithm, metric exposure
- Approved for deployment

#### Second Verification
- CRITICAL: Current code still has hardcoded token IDs
- CRITICAL: Router doesn't format prompts
- Solution design is correct but not implemented

## Key Insights

1. **Two Critical Changes Needed**:
   - Router must format prompts with Qwen3 template
   - Candle backend must detect token IDs dynamically

2. **Token Detection Priority**:
   - tokenizer.json → vocab.json → tokenizer_config.json → hardcoded fallback
   - Must handle different tokenizer formats

3. **Prompt Format**:
   ```
   <|im_start|>system
   Judge whether the Document meets...
   <|im_start|>user
   <Instruct>: {instruction}
   <Query>: {query}
   <Document>: {doc}
   <|im_start|>assistant
   <think>
   
   </think>
   
   ```

4. **Score Calculation**:
   - Get logits for last token
   - Extract yes/no logits
   - score = exp(yes) / (exp(yes) + exp(no))

## Implementation Status
- Python backend: Has dynamic detection but not used
- Candle backend: Hardcoded IDs, needs update
- Router: Missing prompt formatting
- Tests: Show correct usage pattern