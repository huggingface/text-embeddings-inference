# Listwise Reranker Integration – Detailed Implementation Plan

## 0. Objective & Constraints
- Add first-class support for Jina-style listwise reranking to TEI with Candle backend parity to Python reference in `jina-reranker-v3/modeling.py`.
- Preserve existing pairwise reranker path as default; listwise activates only for compatible models or when forced via CLI.
- No regression for embedding/sequence-classifier models. HTTP and gRPC APIs keep the same request/response schema.
- Respect `workspace-write` sandbox (only edit project files) and keep fast tokenizer requirement.

## 1. High-Level Milestones
1. **Model Detection & CLI Controls** – extend router metadata, CLI args, runtime state.
2. **Prompt & Tokenization Layer** – reusable prompt builder, payload guard rails, block estimator.
3. **Inference Pipeline** – listwise-aware queueing + backend dispatch wrappers.
4. **Candle Backend Implementation** – hidden-state projector, embedding extraction.
5. **Testing & Telemetry** – unit/integration tests, Prometheus metrics, docs updates.

Each milestone below enumerates concrete tasks, owner files, acceptance tests, and notes.

---

## 2. Tasks & Deliverables

### 2.1 Router: Model Detection & CLI
**Files**: `router/src/lib.rs`, `router/src/main.rs`, `router/src/http/server.rs`, `router/src/grpc/server.rs`, `router/src/http/types.rs`

1. **Extend model kind metadata**
   - Introduce `ModelKind::ListwiseReranker` alongside existing variants.
   - Implement `detect_model_kind(repo, tokenizer)` that prioritises listwise detection before falling back to classifier/embedding.
   - Detection must simultaneously satisfy:
     1. `projector.0.weight` and `projector.2.weight` tensors exist, while the corresponding biases do **not**.
     2. Tokenizer exposes `<|embed_token|>` and `<|rerank_token|>` IDs.
     3. Architecture matches Qwen-style causal LM (`JinaForRanking`, `Qwen3ForCausalLM`, etc.).
   - Example snippet (lib.rs):
     ```rust
     fn detect_model_kind(repo: &ModelRepo, tokenizer: &Tokenizer) -> Result<ModelKind> {
         if has_lbnl_signature(repo, tokenizer)? {
             return Ok(ModelKind::ListwiseReranker);
         }
         if is_sequence_classifier(repo)? {
             return Ok(ModelKind::SequenceClassifier);
         }
         Ok(ModelKind::Embedding)
     }
     ```

2. **CLI knobs & AppState wiring**
   - Add `--reranker-mode`, `--max-listwise-docs-per-pass`, `--rerank-ordering`, `--rerank-instruction`, `--listwise-payload-limit-bytes` flags (per TECHSPEC §5.2).
   - Add `--listwise-block-timeout-ms` (default: `30000`) to bound per-block processing time.
   - Update `Args`, propagate to new `CliFlags` struct inside `AppState`; include defaults from spec.
   - Guard mutually exclusive prompt options (existing defaults vs listwise overrides).

3. **Strategy selection**
   - Introduce `Strategy::{Pairwise,Listwise}` plus `determine_strategy(cli_mode, model_kind)`.
   - HTTP/gRPC handlers select strategy once per request, rejecting mismatches with `ApiError::ModelNotSupported`.
   - Inject early payload guard (byte-size) in HTTP path before queuing.

4. **HTTP & gRPC dispatch**
   - Wrap existing pairwise work in helper `rerank_pairwise(req, infer, info)`.
   - Create new `rerank_listwise(req, state)` that calls into core/backend abstraction (see §2.3/§2.5).
   - Ensure `return_text` behavior is preserved.

5. **Types & Validation extensions**
   - Keep external schema stable; track ordering preference inside server state unless future API changes require exposure.
   - Add validation error for `[texts]` exceeding `max_listwise_docs_per_pass` once listwise mode is active.

### 2.2 Core Prompt & Tokenization Layer
**Files**: `core/src/prompt.rs` (new), `core/src/tokenization.rs`, `core/src/infer.rs`, `core/src/queue.rs`

1. **New prompt module**
   - Add `core::prompt::listwise::{build_prompt, sanitize, estimate_tokens}`.
   - `sanitize` strips user-provided occurrences of `<|im_start|>`, `<|rerank_token|>`, `<|embed_token|>`, etc., preventing prompt injection.
   - Mirror Python template (system/user/assistant sections, optional instruction, doc sandwich pattern).
   - Example constructor:
     ```rust
     pub fn build_prompt(ctx: &PromptContext) -> String {
         format!(
             "{system}{user}\n{docs}\n<query>\n{query}{query_token}\n</query>{assistant}",
             system = ctx.system_block(),
             // ...
         )
     }
     ```

2. **Token guards & estimation**
   - Two-phase validation:
     1) Pre-queue fast rejection: optional `estimate_tokens(char_len, num_docs)` heuristic (char count × avg token ratio) to drop obviously oversized requests early.
     2) Block assembly: authoritative incremental tokenization while building each block (correctness gate; determines final chunk boundaries).
   - The heuristic is purely an optimization; correctness relies on phase (2).
   - Expose `encode_listwise(tokenizer, &prompt, max_seq_len)` returning `RawEncoding` with left padding; errors out if limit exceeded.

3. **Queue isolation**
   - Introduce `BatchScope::Listwise` and `BatchScope::Pairwise`.
   - Semantics (unambiguous):
     - Listwise blocks from the SAME request are processed sequentially (algorithm requirement).
     - Listwise blocks from DIFFERENT requests MUST NOT share the same backend batch (no cross-request co-batching).
     - Pairwise requests continue using existing batching logic unchanged.
   - Implementation options:
     - Either use a separate internal queue for listwise traffic, or tag entries with `BatchScope::Listwise` and ensure the queue builder never mixes scopes within a batch.

4. **Infer facade**
   - Introduce `Infer::dispatch_listwise_query` / `Infer::dispatch_listwise_block` returning embeddings + metadata.
   - Pairwise path continues using `predict` to minimize regression risk.

### 2.3 Backend Abstraction
**Files**: `backends/src/lib.rs`, `backends/core/src/lib.rs`

1. **Trait updates**
   - Define `pub trait ListwiseBackend` with method signature similar to:
     ```rust
     async fn embed_listwise_query(&self, query: ListwiseQueryInput) -> Result<Vec<f32>>;
     async fn embed_listwise_block(&self, block: ListwiseBlockInput) -> Result<Vec<Vec<f32>>>;
     ```
     Default implementations return `Err(BackendError::Inference("Listwise reranking not supported".into()))`.
   - Candle backend implements the trait; Python backend returns unsupported.

2. **Command channel & warmup**
   - Extend `BackendCommand` enum with `ListwiseQuery(ListwiseQueryInput, sender)` and `ListwiseBlock(ListwiseBlockInput, sender)` and handle in thread loop using new Candle runner.
   - Warmup requirements for listwise (mandatory on CUDA/HPU):
     - Build a representative listwise sample: 1 query + up to `max_listwise_docs_per_pass` (default 125) documents with left padding.
     - Tokenize and run a full forward pass through the Qwen3 backbone and projector once to allocate/cuDNN/attention caches and peak working memory for the max batch length.
     - Record resulting shapes to guide runtime bucket selection if applicable.
     - Health check must accept `ModelType::Embedding` with listwise backend after warmup completes.

3. **Data contracts**
   - `ListwiseQueryInput` carries tokenized query-only prompt (input ids, positions, mask).
   - `ListwiseBlockInput` carries fully tokenized block data:
     ```rust
     pub struct ListwiseBlockInput {
         pub input_ids: Vec<u32>,
         pub position_ids: Vec<u32>,
         pub attention_mask: Vec<u8>,
         pub cumulative_seq_lengths: Vec<u32>,
         pub embed_token_id: u32,
         pub rerank_token_id: u32,
         pub doc_offsets: Vec<usize>,
     }
     ```
   - `embed_listwise_query` returns the projected query embedding (`Vec<f32>`). `embed_listwise_block` returns projected document embeddings for that block (`Vec<Vec<f32>>`). Router performs cosine scoring + query update.

### 2.4 Candle Backend Implementation
**Files**: `backends/candle/src/lib.rs`, `backends/candle/src/lbnl_reranker.rs` (new), `backends/candle/src/layers/projector.rs` (new), `backends/candle/src/models/qwen3.rs`

1. **Projector layer**
   - Create reusable `Projector` struct (`Linear -> ReLU -> Linear`, no bias) that loads weights dynamically using tensor shapes rather than hardcoded dimensions. Derive hidden/latent sizes from weight tensors or `config.hidden_size`.
   - REQUIRED validation in loader (shape, dtype, no bias):
     ```rust
     impl Projector {
         pub fn load(vb: &VarBuilder, config: &Qwen3Config) -> Result<Self> {
             let w1 = vb.pp("projector.0").get((), "weight")?;
             ensure!(w1.dims() == [config.hidden_size, config.hidden_size / 2],
                 "Invalid projector.0.weight shape: expected [{}, {}], got {:?}",
                 config.hidden_size, config.hidden_size / 2, w1.dims());
             ensure!(matches!(w1.dtype(), DType::F32 | DType::BF16 | DType::F16),
                 "Unexpected projector.0.weight dtype: {:?}", w1.dtype());
             ensure_no_bias(vb.pp("projector.0"))?;

             let w2 = vb.pp("projector.2").get((), "weight")?;
             ensure!(w2.dims() == [config.hidden_size / 2, 512],
                 "Invalid projector.2.weight shape: expected [{}, 512], got {:?}",
                 config.hidden_size / 2, w2.dims());
             ensure!(matches!(w2.dtype(), DType::F32 | DType::BF16 | DType::F16),
                 "Unexpected projector.2.weight dtype: {:?}", w2.dtype());
             ensure_no_bias(vb.pp("projector.2"))?;

             Ok(Self::from_weights(w1, w2))
         }
     }
     ```

2. **Hidden-state extractor**
   - Extend `Qwen3Model` with `forward_hidden(&self, batch: Batch) -> Result<Tensor>` returning final layer hidden states (`[bs, seq, hidden]`).
   - Reuse existing attention mask + left padding logic to maintain parity with pairwise embeddings.

3. **Embedding extraction helpers**
   - Implement `LbnlRerankerCandle::embed_query(query_input)` returning the projected query embedding.
   - Implement `LbnlRerankerCandle::embed_block(block_input)` returning projected document embeddings for the block (Vec<Vec<f32>>), leaving scoring to router.

4. **Backend integration**
   - Register new module in `CandleBackend::new`: when `ModelType::Embedding` and `has_lbnl_signature` true, instantiate `LbnlRerankerCandle` instead of `Qwen3Model`.
   - Implement the `ListwiseBackend` trait by delegating to these helpers.

### 2.5 Router ↔ Backend Glue
**Files**: `router/src/http/server.rs`, `router/src/grpc/server.rs`, `router/src/listwise/math.rs` (new), `core/src/infer.rs`

1. **Listwise execution path (sequential orchestration)**
   - Build `ListwiseExecutionContext` for each request:
     1. Determine ordering (input vs random).
     2. Build and tokenize the query-only prompt; call `Infer::dispatch_listwise_query` to obtain the initial projected query embedding.
     3. Iterate through documents, assembling blocks until token estimate exceeds capacity.
     4. For each block *in sequence*: build prompt, sanitize, tokenise via `encode_listwise`, validate special token counts (see below), send `ListwiseBlockInput` to backend using `Infer::dispatch_listwise_block`, await embeddings with per‑block timeout (`--listwise-block-timeout-ms`), compute cosine scores via math helpers, update the running query embedding via weighted average `(1 + score) / 2` heuristic (normalize result), then proceed to the next block.
     5. After all blocks processed, produce final ranking from accumulated scores.

2. **Special token validation (after tokenization, before backend dispatch)**
   ```rust
   fn validate_special_tokens(
       input_ids: &[u32],
       embed_token_id: u32,
       rerank_token_id: u32,
       expected_doc_count: usize,
   ) -> anyhow::Result<()> {
       let embed_count = input_ids.iter().filter(|&&id| id == embed_token_id).count();
       anyhow::ensure!(embed_count == expected_doc_count,
           "Expected {} embed tokens, found {}", expected_doc_count, embed_count);

       let rerank_count = input_ids.iter().filter(|&&id| id == rerank_token_id).count();
       anyhow::ensure!(rerank_count == 1,
           "Expected 1 rerank token, found {}", rerank_count);
       Ok(())
   }
   ```

3. **Vector math utilities**
   - Add `router/src/listwise/math.rs` housing pure functions (e.g., `cosine_similarity`, `weighted_average`, `normalize`, `add_scaled`) operating on `&[f32]` / `Vec<f32>`.
   - Unit-test these helpers independently to guarantee numerical correctness and maintain readability in the main handler.

4. **Metadata aggregation**
   - Accumulate tokenization/queue/inference durations per block, computing averages for headers.
   - Populate `RerankResponse(Vec<Rank>)` sorted by score; reattach optional `text` respecting `return_text`.

5. **Error handling**
   - Treat backend error in any block as fatal for the overall request; propagate the error and include context about the failing block.
   - Map backend `TokenLimitExceeded` to HTTP 413 with hint (`"Try reducing document count or length"`). Distinguish between pairwise/listwise strategy mismatches for diagnostics.

### 2.6 Telemetry & Observability
**Files**: `router/src/prometheus.rs`, `router/src/http/server.rs`

- Register histograms during server bootstrap:
  - `tei_lbnl_group_size`
  - `tei_lbnl_seq_tokens`
  - `tei_lbnl_blocks_per_request`
  - `tei_lbnl_block_inference_duration_ms`
  - `tei_lbnl_ms_per_group`
- Record metrics inside listwise orchestration loop (per block) before responding.

### 2.7 Testing Strategy
1. **Rust unit tests**
   - Prompt builder tests validating token placement and sanitization.
   - Queue isolation tests: ensure listwise batches are never coalesced.
   - Detection tests verifying failure cases (missing projector weights, missing tokens).
   - Math helper tests covering cosine similarity, weighted average, normalization, and query update logic (including normalization and weight application).

2. **Integration tests**
   - `integration_tests/tests/rerank_listwise.rs`: spin TEI with stub Candle backend verifying `/rerank` returns sorted ranks, honors ordering, enforces payload/document limits.
   - Extend existing `test_qwen3_reranker.py` to cover listwise path once backend available.

3. **Golden comparisons**
   - Compare Candle embeddings + router math vs Python reference on fixed fixture (small query + docs).
   - Use relative tolerance with absolute fallback for near‑zero values:
     ```rust
     fn assert_approx_equal(a: f32, b: f32, rel_tol: f32, abs_tol: f32) {
         let abs_diff = (a - b).abs();
         let denom = a.abs().max(b.abs()).max(1e-8);
         let rel_diff = abs_diff / denom;
         assert!(rel_diff < rel_tol || abs_diff < abs_tol,
             "Values differ: {} vs {} (rel: {:.2e}, abs: {:.2e})", a, b, rel_diff, abs_diff);
     }
     // Use: assert_approx_equal(expected, actual, 1e-5, 1e-8)
     ```

---

## 2.8 Design Rationale & Trade-offs

- Scoring in router: Pros — algorithm visible, backend simple/portable; Con — router holds ML math details. Mitigation: document coupling and keep math isolated in `router/src/listwise/math.rs` with unit tests.
- Sequential blocks: Pros — correct per algorithm; Con — higher latency. Mitigation: per‑block timeout, documented latency (`total ≈ blocks × block_latency`), optional progress headers later.
- Vec<Vec<f32>>: Pros — simpler code; Con — more allocations. Mitigation: profile first; optimize to flat buffers/SIMD if needed.
- Queue isolation: Prevent cross‑request mixing for listwise; pairwise unchanged. Mitigation: explicit `BatchScope` semantics and/or separate internal queue.

### 2.9 Documentation Updates
- Update `README.md` (top-level) to highlight `--reranker-mode listwise` usage.
- Produce new section in `docs/TECHSPEC.md` referencing implementation details for maintainers (link to modules).
- Document Prometheus metrics.

---

## 3. Sequencing & Dependencies
1. **Detection & CLI** (enables feature flagging) – merge early to unblock downstream work.
2. **Prompt/Tokenization** – provides shared layer for backend & router.
3. **Backend Trait & Candle Implementation** – heavy lift; build behind feature guard.
4. **Router wiring** – integrate once backend interface is stable.
5. **Telemetry & Tests** – finalize before release.

Workstreams can proceed in parallel after milestone 1 with mocks (e.g., stub listwise backend for router development).

---

## 4. Risk Mitigation
- **Token budget miscalculation**: add unit tests for estimation and enforce 0.9 × max_seq_len guard.
- **Performance regression**: keep pairwise path untouched, ensure queue isolation, add benchmarks (optional) before enabling auto-mode.
- **Model variance**: runtime check for projector weights & special tokens; emit actionable error if missing.
- **Numeric stability**: normalize embeddings before cosine, clamp results to [-1, 1].

---

## 5. Acceptance Checklist
- `ModelKind::ListwiseReranker` automatically detected for Jina v3 weights.
- `TEI` serves `/rerank` in listwise mode producing deterministic ranking identical to Python reference on sample fixtures.
- Pairwise reranker still functional; existing tests pass.
- New Prometheus metrics and CLI flags documented.
- Integration tests green; lint & fmt clean.
