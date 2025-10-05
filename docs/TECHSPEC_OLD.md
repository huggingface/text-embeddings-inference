
# TECHSPEC_PLAN.md — Add Jina v3 Listwise Reranker (LBNL) to TEI

**Owner:** @you
**Repo:** huggingface/text-embeddings-inference
**Target branch:** `main`
**Scope:** HTTP + gRPC + Candle backend 우선(ORT는 후속), 동적 배칭은 1차 버전에서 제한
**Related:**

* TEI가 `/rerank` HTTP 엔드포인트를 제공(SequenceClassification 크로스 인코더 기반)한다는 README 및 API 문서 확인. ([GitHub][1])
* TEI 내부는 `axum` 기반 라우터와 `router/src/lib.rs`/`main.rs` 구성이며, 백엔드는 `backends` 크레이트(캔들/ORT)이다(컴파일 로그 및 이슈에서 경로/특징 확인). ([GitHub][2])
* Qwen3 계열/리랭커 지원 요구 이슈/PR들이 존재(배경/호환성 참고). ([GitHub][3])

---

## 0) 목표(Goals) / 비목표(Non‑Goals)

**Goals**

* `jina-reranker-v3`와 동형인 **LBNL(listwise) 재랭커**를 TEI에서 1st‑class로 서빙
* TEI의 **/rerank** API와 완전 호환(요청·응답 스키마 유지), 단 **모델 특성에 따라 내부적으로 listwise 수행**
* Qwen3(CausalLM) 백본에서 **특수 토큰 위치 히든을 집계 → MLP projector(1024→512→512, ReLU, bias 없음) → cosine(q,d)** 스코어 계산(코사인 내부에서 L2 정규화 수행)
* **좌측 패딩(left padding)** 및 **프롬프트 템플릿**을 모델 카드/논문 정의와 일치
* 긴 입력/다문서에서 **listwise 그룹 처리**(문서 묶음 단위로 1패스) + **성능/메모리 최적화**
1. **LBNL(listwise) 리랭커**를 **1회 전방패스에서 다문서+쿼리 공동 인코딩** 후, 특수토큰 위치( `<|embed_token|>`, `<|rerank_token|>` )의 **마지막 히든스테이트 → 2층 MLP(1024→512→512, ReLU, bias 없음)** → **코사인 유사도**로 점수를 산출하고, 블록별 쿼리 임베딩은 가중 평균으로 결합.
2. 기존 `/rerank` API와 **하위 호환** 유지. **새 모드(`strategy`)** 추가만으로 동일 엔드포인트에서 listwise 가동.
3. **Candle backend 우선** 구현. (Qwen3 0.6B 기반 모델 이미 TEI 임베딩에서 지원되므로 재사용 가능) ([GitHub][1])
4. **Prompt builder** 내장: Qwen instruction 형식(system/user/assistant) + 문서 샌드위치 + 특수토큰 배치.

**Non‑Goals**

* TEI의 기존 **SequenceClassification 크로스 인코더** 경로를 교체하지 않음(병존)
* 학습/미세튜닝 기능 추가는 범위 외(서빙 전용)
* 모델 가중치 변환(예: LM‑head→CLS 1‑logit 치환)은 목표 아님(점수 정의가 다름)
* ORT/ONNX 경로, VLLM 통합, 가이우디(HPU) 전개는 2차 목표로 미룬다.
* Listwise **크로스‑리퀘스트** 동적 배칭(서로 다른 요청의 문서가 같은 컨텍스트 창에서 상호 주의하지 않도록 하는 세그먼트 마스크)은 1차에서 **비활성화**.
* LoRA/합성곱 추론 가속 등 모델 그래프 컴파일 최적화는 범위 외.

---

## 1) API/프로토콜(HTTP & gRPC)

### 1.1 HTTP (기존과 동일)

* **Endpoint**: `POST /rerank` (유지) ([GitHub][1])
* **Request JSON**:

  ```json
  {
    "query": "string",
    "texts": ["string", "..."],
    "instruction": "string (optional)",
  "max_listwise_docs_per_pass": 125, // optional, default 125 (<=125)
    "ordering": "input|random",       // optional, default "input"
    "strategy": "auto|pairwise|listwise"  // optional, default "auto"
  }
  ```

  * `strategy=auto`: 로드된 모델의 타입이 listwise면 listwise로, 아니면 기존 pairwise 경로 사용
  * 참고: Jina reference `JinaForRanking.rerank()`는 `instruction` 인자를 노출하지 않으므로 TEI 통합 시 별도 래핑/파라미터 전달이 필요함.
* **Response JSON**(동일):

  ```json
  { "results": [ { "index": 0, "score": 0.123 }, ... ] }
  ```

  * `index`는 입력 `texts` 인덱스(원래 순서 기준)

### 1.2 gRPC

* 1차 반영 범위에서는 gRPC 스키마를 변경하지 않는다. 기존 `Rerank` RPC는 그대로 유지하고, 서버 내부에서 **모델 자동 감지(auto)** 로 pairwise/listwise 경로를 분기한다. (gRPC 확장은 후속 제안으로 분리)

---

## 2) 모델 어댑터: `LbnlReranker` (신규 백엔드)

### 2.1 로더/탐지

* **탐지 규칙**(모두 만족 시 LBNL 경로 진입):

  1. `config.json`의 **architectures**가 `QwenForCausalLM`, `Qwen3ForCausalLM`, **또는 `JinaForRanking`**(Qwen3 기반 커스텀 헤드)이며 `model_type`이 `qwen3`
  2. 모델 리포에 **프로젝터 가중치**가 존재 (`projector.*.weight` 패턴 전체 탐색. 샤딩 인덱스 `model.safetensors.index.json` 포함)
  3. 토크나이저 구성(`tokenizer.json`, `special_tokens_map.json`, `added_tokens.json` 등)에서 **특수 토큰 문자열**(`<|embed_token|>`·`<|rerank_token|>` 또는 향후 파생형)을 찾고, 이를 통해 **런타임에 토큰 ID를 조회**(문자열 → ID 매핑. ID 하드코딩 금지)
  4. 조회된 토큰 ID를 이용해 추론 중 히든스테이트 위치를 결정(하드코딩된 151670/151671 값은 참조용일 뿐, 감지 시에는 문자열→ID 매핑 사용)
* 토크나이저: **fast tokenizer 필수**(TEI 제약). 모델에 없으면 런타임 추가 불가로 **로드 실패 처리**. ([Hugging Face][2])
* Long‑RoPE/rope‑scaling 등 **Qwen3 롱컨텍스트 설정**은 기존 Qwen3 임베딩 경로의 설정 방식을 재사용(가능 범위 내). ([GitHub][1])

### 2.2 프롬프트 빌더(기본 템플릿)

* 시스템/유저/어시스턴트 롤 태그는 모델 카드에 준수. 기본값(논문 3.2에 준함):

  ```
  <|im_start|>system
  You are a search relevance expert...
  <|im_end|>
  <|im_start|>user
  I will provide you with {k} passages...
  Rank the passages based on their relevance to query: {QUERY}
  <instruct>
  {INSTRUCTION?}
  </instruct>
  <passage id="i">
  {DOC_i}<|embed_token|>
  </passage> ...
  <query>
  {QUERY}<|rerank_token|>
  </query>
  <|im_end|>
  <|im_start|>assistant
  <think>

  </think>
  ```
* `instruction`이 없으면 `<instruct>` 블록은 생략되며, `{k}`와 `{QUERY}` 문장은 그대로 유지된다. `</think>` 뒤에는 레퍼런스 구현과 동일하게 공백 줄 두 개(`\n\n`)를 남긴다.
* 프롬프트를 만들기 전에 `query`와 각 문서에서 `<|embed_token|>`, `<|rerank_token|>` (및 fallback 표기)의 등장 여부를 제거해 특수토큰 인젝션을 방지한다.
* **좌측 패딩** 강제, pad_token 미존재 시 reference 구현은 **pad=unk**로 강제(토크나이저가 pad 토큰을 제공하지 않음). TEI에서 필요하면 pad=eos로 재정의 가능. ([GitHub][1])

### 2.3 스코어 계산(핵심 로직)

* 인퍼런스: Qwen3 CausalLM **최종 레이어 히든** `H`에서

  * `h_q = H[t_{<|rerank_token|>}]`, `h_{d_i} = H[t_{<|embed_token|>_i}]`
  * `z_q = MLP(h_q)`, `z_{d_i} = MLP(h_{d_i})` (MLP=1024→512→512, ReLU, bias 없음)
  * 동일 쿼리에 대해 여러 블록을 처리한 경우, 각 블록의 `z_q`를 **블록 가중치** `w_b = (1 + max(score_b)) / 2`로 가중 평균(`q_bar = sum(w_b * z_q_b) / sum(w_b)`)하고, 문서 벡터는 블록별 `z_{d_i}`를 그대로 사용
  * `score_i = cosine(q_bar, z_{d_i})` (cosine에서 L2 정규화 수행)
* **정확한 토큰 위치**는 tokenizer로 인덱싱한 `<|embed_token|>`, `<|rerank_token|>`의 id로 탐색(멀티 발생 시 마지막 매치)
  * 참고: Jina reference 구현은 `doc_embed_token_id=151670`, `query_embed_token_id=151671`로 상수 선언되어 있으며, Hub `added_tokens.json`의 값과 일치해야 한다.
* Candle Qwen 백엔드는 listwise 경로에서 **`output_hidden_states=true`를 지원하는 전용 함수**(예: `forward_with_hidden_states`)를 제공해야 하며, 이를 통해 마지막 히든스테이트에서 특수 토큰 위치의 벡터를 추출한다.
* **출력**: 입력 순서 `texts`와 동일한 `index` 매핑으로 `score_i` 반환

---

## 3) 실행 계획(서빙 파이프라인)

### 3.1 리스트와이즈 그룹 처리

* 요청의 `texts`를 길이/토큰 수 기준으로 **그룹화**하여 한 컨텍스트에 배치(`max_listwise_docs_per_pass`, 기본 125). 그룹 크기는 토큰 길이에 따라 자동 조정(상한 125).
* `length_capacity = max_seq_len - 2 * query_length`를 추적하며(프롬프트 본문과 `<query>` 블록 모두에 쿼리가 삽입되기 때문), 다음 조건 중 하나가 참이면 새 블록을 시작: `len(block_docs) >= max_listwise_docs_per_pass` 또는 `length_capacity <= max_doc_len`
* 각 그룹에 대해 **단일 포워드**로 **k개 점수** 산출하고, 필요 시 여러 그룹 결과를 결합해 최종 스코어 배열 생성(**리스트 의존성**으로 그룹간 상호작용은 불가; 문서화)
* 참고: Jina reference 구현은 `block_size=125`로 하드코딩되어 있으며, TEI는 동일 기본값을 유지하되 CLI/환경설정으로 노출한다.
* **주의**: 블록별로 반환된 점수는 **가중치 계산용(보조 정보)**이며, 최종 응답 전에 반드시 `q_bar`와 모든 문서 임베딩으로 한 번 더 cosine을 계산해 결과를 덮어쓴다. (레퍼런스와 완전 동일한 식을 강제)

### 3.2 동적 배칭과 격리

* TEI의 **토큰 기반 동적 배칭**은 **요청 단위(=하나의 쿼리+문서 리스트)**로만 결합(각 요청은 독립 시퀀스)
  → 서로 다른 쿼리의 리스트를 같은 포워드에 섞지 않음(리스트와이즈 상호작용 보존)

---

## 4) 라우터/백엔드 변경점

### 4.1 라우터

* `router`의 `/rerank` 핸들러에 **model‑kind 분기** 추가:

  * SequenceClassification(기존): 문서별 pairwise 호출
  * **LbnlReranker(신규)**: 그룹화→listwise 빌더→단일 포워드→k 로짓 반환
* OpenAPI 문서 `/docs`에 선택 파라미터(`max_listwise_docs_per_pass`, `ordering`, `strategy`) 설명 추가. ([GitHub][1])

### 4.2 백엔드

* `backends/lbnl_reranker/` (신규)

  * Qwen3 CausalLM 호출, 히든 추출, projector 적용, cosine 계산
  * Flash‑Attention·cuBLASLt 경로 재사용(가용 시) ([GitHub][1])
* 토크나이저: fast tokenizer 로드 확인(미존재 시 에러 반환 – 기존 정책과 동일) ([Hugging Face][2])

---

## 5) 성능/메모리/튜닝

* **복잡도**: 한 그룹 길이 `T = L_q + Σ L_doc_i`, self‑attn 비용 O(T²).

  * 권장 기본: `max_listwise_docs_per_pass=125`, **쿼리는 512 토큰, 문서는 2048 토큰 이하**로 클립(모델 카드와 동일)
* **옵션**:

* `--max-listwise-docs-per-pass`(기본 125)
  * `--max-seq-len` 안전 가드
  * `--rope-scaling`/`--flash-attn` 플래그 상속
* **메트릭 노출**(Prometheus):

  * `tei_lbnl_group_size`, `tei_lbnl_seq_tokens`, `tei_lbnl_ms_per_group`, `tei_lbnl_groups_per_req`
  * **ordering**에 따른 평균 스코어/latency 비교용 라벨

---

## 6) 호환성/폴백

* 사용자가 `strategy=pairwise`로 강제하면 기존 크로스 인코더 경로로 폴백(성능↓ 가능)
* 모델이 **필수 스펙 미충족** 시(특수 토큰 없음, projector 없음, fast tokenizer 아님) → **로드 실패** + 에러 메시지:
  `"model is not a supported LBNL reranker (missing special tokens or projector weights)"`

---

## 7) 보안/에러 처리

* 입력 길이·문서 수 상한 체크(HTTP 413)
* 특수 토큰 누락 시 422
* 그룹화 중 어느 그룹도 토큰 한도를 넘지 않도록 **보수적 컷**(over‑length 문서 drop 옵션: `drop_long_docs=true|false`)

---

## 8) 테스트 플랜

1. **유닛 테스트**

   * 토큰 위치 탐색: `<|embed_token|>`, `<|rerank_token|>` 인덱스 추출 정확성
   * projector 출력과 cosine 계산 검증(고정 입력→고정 점수)
   * 블록 가중 평균(`block_weights`)이 기대한 방식으로 query 임베딩을 결합하는지 확인
   * 좌패딩/없을 때 **pad=unk**로 동작 확인(override 시 pad=eos 테스트 추가)
2. **통합 테스트**

   * `/rerank`에 `texts=K`(K∈{1,4,16,64,125}) 요청 → **k개 스코어 반환**
   * `max_listwise_docs_per_pass`에 따라 **포워드 호출 수**가 변하는지 검증
   * `ordering=input|random` 설정 시 결과 안정성(랜덤 모드의 통계적 편차는 허용)
3. **로드/안정성**

   * fast tokenizer 없음, projector 누락, 특수 토큰 누락 모델에 대해 **적절한 에러**
4. **성능 회귀**

   * A10/A100에서 group size·토큰 길이 스윕 → p50/p95 latency, VRAM 사용량 로그

---

## 9) 문서화(README / docs)

* `Using Re-rankers models` 섹션에 **“Listwise (LBNL) Reranker”** 하위 절 추가:

  * 어떤 모델이 여기에 해당하는지(예: `jina‑reranker‑v3`)
  * 요청 옵션(`strategy`, `max_listwise_docs_per_pass`, `ordering`)
  * **주의**: 리스트 구성·순서에 스코어가 **의존**할 수 있음(모델 특성)
* OpenAPI에 필드 설명 반영

---

## 10) 오픈 이슈·제약(명시)

* **Qwen3 Reranker 지원 요청 이슈** 해결과 연계(모델 타입 감지/로더 공유) ([GitHub][3])
* **Tokenizer fast 필수**: TEI 정책 상 요구됨(관련 토론 있음) ([Hugging Face][2])
* 아주 큰 `k`(예: 100+)에서는 **pairwise가 더 빠를 수 있음**(T² 스케일). 운영 가이드에 권장 k~8–32 범위 명시.
* 한 요청 내 **다중 쿼리**는 미지원(리스트와이즈 특성상 쿼리 1개 전제)

---

## 11) 사용 예시

```bash
# Docker
model=jinaai/jina-reranker-v3   # (가칭, projector/특수토큰 포함 리포)
volume=$PWD/data
docker run --gpus all -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-embeddings-inference:1.8 \
  --model-id $model --reranker-mode auto

# HTTP
curl 127.0.0.1:8080/rerank -X POST -H 'Content-Type: application/json' -d '{
  "query": "등기 우편 반송 처리 기준",
  "texts": ["반송되는 경우에는...", "반송되었을 때는...", "배달 불가 사유..."],
  "max_listwise_docs_per_pass": 125,
  "ordering": "input",
  "strategy": "auto"
}'
```
---

## 1. 외부 인터페이스(API/CLI/Proto) 변경

### 1.1 HTTP `/rerank`(기존 유지 + 확장)

* **요청(JSON)**

```json
{
  "query": "What is Deep Learning?",
  "texts": ["doc A ...", "doc B ...", "doc C ..."],
  "strategy": "listwise",              // [optional] "pairwise"(default) | "listwise"
  "max_pass_docs": 125,                // [optional] listwise 1패스 최대 문서 수 (기본 125)
  "template": "jina_v3_lbnl"           // [optional] 빌트인 템플릿 키
}
```

* **응답(JSON)**: 기존 `/rerank`와 동일(스코어 배열 + 인덱스/문서), 단 listwise에서도 **문서별 점수**가 동일 형태로 반환되도록 정규화.

> 참고: `/rerank` 엔드포인트는 이미 존재하며, bge/gte 등의 시퀀스 분류형 리랭커를 지원한다. 본 PR은 **새 전략 플래그 추가**만으로 동일 엔드포인트를 재사용한다. ([GitHub][1])

### 1.2 CLI 플래그(`router`)

* `--reranker-mode {auto|pairwise|listwise}` (기본 `auto`)

  * `auto`: 모델 메타/토크나이저에 **특수토큰**(`'<|embed_token|>'`, `'<|rerank_token|>'`)과 `projector.0/2.weight` 가중치 존재 시 `listwise`로 자동 전환.
* `--max-listwise-docs-per-pass <int>` (기본 125)
* `--listwise-payload-limit-bytes <int>` (기본 2_000_000 유지, 필요시 상향 권장) — 413 방지 관련 참고. ([GitHub][2])
* `--default-prompt-name jina_v3_lbnl` 또는 `--default-prompt '{...}'`
  (TEI는 기본 프롬프트/이름 플래그를 이미 지원하며, 로그에 표시됨) ([GitHub][4])

### 1.3 gRPC (`proto/tei.proto`)

* `message RerankRequest { ... optional Strategy strategy = 4; ... }`
* `enum Strategy { STRATEGY_UNSPECIFIED = 0; PAIRWISE = 1; LISTWISE = 2; }`
* 서버/클라이언트 스텁 재생성.

---

## 2. 내부 구조 변경(파일/코드 수준)

### 2.1 라우터 (`router/`)

#### 2.1.1 `router/src/lib.rs`

* **변경점**

  1. **모델 종류 열거형**에 `ModelKind::ListwiseReranker` 추가.
  2. 모델 초기화 경로에서 **`detect_lbnl_reranker()`** 호출:

     * 조건: 토크나이저에 `<|embed_token|>`, `<|rerank_token|>` 추가토큰 존재 **그리고** safetensors에 `projector.0.weight`, `projector.2.weight` 가중치 키 존재 → `ModelKind::ListwiseReranker`.
  3. `/rerank` 핸들러에서 요청 JSON 파싱 시 `strategy` 를 읽어 **`Pairwise` vs `Listwise` 분기**. `auto`일 때는 `ModelKind` 기반 디폴트.
  4. **동적 배칭 제어**: `Listwise`일 때는 **요청 단위로 격리**(다른 요청과 컨텍스트 공유 금지). 1차 버전은 `max_batch_requests=1` 강제 또는 동일 요청 내 문서만 배치.
  5. **토큰 상한**: Qwen3 0.6B 컨텍스트(예: 131k)에서 **안전 여유**를 둔 도큐먼트 슬라이싱/청킹 로직 도입.

* **핵심 코드 스케치(개념적 diff)**

```rust
// router/src/lib.rs (개념적 코드)
enum ModelKind {
    Embedding,
    SequenceClassifier,
    ListwiseReranker, // NEW
}

fn detect_model_kind(repo: &ModelRepo) -> ModelKind {
    if detect_lbnl_reranker(repo) { return ModelKind::ListwiseReranker; }
    if detect_sequence_classifier(repo) { return ModelKind::SequenceClassifier; }
    ModelKind::Embedding
}

async fn rerank_handler(Json(req): Json<RerankRequest>, State(state): State<AppState>) -> Json<RerankResponse> {
    let strategy = req.strategy.unwrap_or_else(|| state.default_strategy_for_model());
    match (strategy, state.model_kind) {
        (Strategy::Listwise, ModelKind::ListwiseReranker) | (Strategy::Auto, ModelKind::ListwiseReranker) => {
            state.backend.rerank_listwise(req).await
        }
        _ => state.backend.rerank_pairwise(req).await
    }
}
```

> 라우터/핸들러/전역 설정은 이미 `axum` 기반으로 구성되어 있음(413 이슈 스레드에서 기본 본문 제한 설명). ([GitHub][2])

#### 2.1.2 `router/src/main.rs`

* **CLI 파서**에 새 플래그 3종 등록(§1.2).
* `Args` → `AppState` 주입 시 `default_strategy`/`max_listwise_docs_per_pass` 보관.

### 2.2 코어 (`core/`)

#### 2.2.1 `core/src/prompt.rs` (신규 파일)

* **함수**: `build_jina_v3_prompt(query: &str, docs: &[&str], instruction: Option<&str>) -> String`

  * Qwen instruction 형식:

    * `<|im_start|>system ... <|im_end|>`
    * `<|im_start|>user` ➝ `I will provide you with {k} passages...` + `Rank ... query: {query}`
    * `instruction`이 있으면 `<instruct> ... </instruct>` 블록 삽입(없으면 생략)
    * 각 문서는 `<passage id> ... {DOC_i}<|embed_token|>` 패턴으로 주입, 마지막에 `<query> {query}<|rerank_token|> </query>` 추가
    * `<|im_start|>assistant` 다음 `<think> ... </think>\n\n` 부가(레퍼런스 구현과 동일)
  * **문서 샌드위치**(쿼리 앞/뒤 배치) 및 **특수토큰 삽입**.

#### 2.2.2 `core/src/tokenization.rs` (기존 확장)

* **함수**: `encode_listwise(&tokenizer, prompt: &str, left_padding: bool) -> Encoded`

  * Qwen3 CausalLM 추론은 **left padding**을 사용하며, 토크나이저에 pad 토큰이 없으면 `unk_token`을 pad로 대체(레퍼런스 구현과 동일). TEI에서 별도 정책이 필요하면 override 가능. ([GitHub][1])
* **유틸**: 컨텍스트 길이/바이트 제한 **사전 검사** + 도큐먼트 배치 청킹(최대 `max_listwise_docs_per_pass`).

### 2.3 백엔드 (`backends/`)

#### 2.3.1 공통: `backends/src/lib.rs`

* **변경점**

  1. `BackendModel`/`ModelKind` 매핑에 `ListwiseReranker` 추가.
  2. **가중치/토크나이저 검사** 로직에 `detect_lbnl_reranker(&repo)` 구현:

    * `tokenizer.json` 또는 `added_tokens.json`에 `<|embed_token|>`, `<|rerank_token|>`
    * `model.safetensors` 키에 `projector.0.weight`, `projector.2.weight` (bias 파라미터 없음)
  3. `trait RerankBackend`에 **새 메서드** 추가:

     ```rust
     async fn rerank_listwise(&self, req: RerankRequest) -> anyhow::Result<RerankResponse>;
     ```
  4. Candle/ORT 분기 — 1차는 Candle만 `Listwise` 지원, ORT는 `unimplemented` 반환.

#### 2.3.2 Candle 구현: `backends/candle/src/lbnl_reranker.rs` (신규)

* **책임**

  * Qwen3‑0.6B CausalLM **전방패스 실행**, **마지막 히든스테이트** 획득.
  * **특수토큰 위치 인덱싱**: `<|embed_token|>` 각각과 `<|rerank_token|>` 1개.
  * **Projector MLP**: `Linear(1024→512, bias=False)` → `ReLU` → `Linear(512→512, bias=False)`(정규화는 cosine에서 처리).
  * **코사인 유사도** 산출: `score_i = dot(q, d_i)`(정규화 완료시 cos 동일).
* **문서 수 초과** 시 **청킹**: 최대 125개/패스(기본) + 토큰 용량 가드. 같은 요청 내 **쿼리 임베딩 재사용** 허용(선택) — 구현 단순화를 위해 1차는 **그룹별 전체 컨텍스트 재구성**.
  * **성능 주의**: kv‑cache 미사용(인스트럭션 고정/오토레그레시브 디코딩 무), `use_flash_attn`/`bf16` 옵션은 Candle 백엔드 기본 최적화 상속.

* **핵심 코드 스케치**

```rust
pub struct LbnlRerankerCandle { 
    model: Qwen3CausalLm, 
    tok: Tokenizer, 
    projector: Projector, 
    max_docs: usize,
}

impl LbnlRerankerCandle {
    fn forward_once(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        let prompt = build_jina_v3_prompt(query, docs); // core::prompt
        let enc = encode_listwise(&self.tok, &prompt, true)?;
        let hs = self.model.forward_hidden_states(&enc)?; // 최종 레이어 HS
        let q_idx = find_token_position(&enc, "<|rerank_token|>")?;
        let d_idx = find_all_positions(&enc, "<|embed_token|>")?;
        let q_h = hs.index(q_idx);
        let q = self.projector.forward(&q_h).l2norm();
        let mut scores = Vec::with_capacity(d_idx.len());
        for &i in &d_idx {
            let d = self.projector.forward(&hs.index(i)).l2norm();
            scores.push(q.dot(&d));
        }
        Ok(scores)
    }
}

#[async_trait]
impl RerankBackend for LbnlRerankerCandle {
    async fn rerank_listwise(&self, req: RerankRequest) -> Result<RerankResponse> {
        let chunks = docs_chunking(req.texts, self.max_docs, self.tok_ctx_limit());
        let mut out = Vec::with_capacity(req.texts.len());
        for docs in chunks {
            let s = self.forward_once(&req.query, &docs)?;
            out.extend(s);
        }
        Ok(RerankResponse::from_scores(req.texts, out))
    }
}
```

> TEI는 이미 **Qwen3 임베딩** 모델을 Candle로 구동하므로, Qwen3 텐서/로더/포지셔널 구성(rope) 등은 재사용한다. 본 구현은 **히든스테이트 반환** 경로만 추가로 노출하면 된다. ([GitHub][1])

#### 2.3.3 Projector MLP 로더: `backends/candle/src/layers/projector.rs` (신규)

* safetensors에서 `projector.0.weight`, `projector.2.weight` 로딩(편향 파라미터 없음), `linear → relu → linear`.

### 2.4 프로토콜 버퍼 (`proto/tei.proto`)

* `RerankRequest`/`Strategy` 추가(§1.3). `buf generate`/`prost` 재생성.

---

## 3. 호환성/경계조건

* **하위호환**: `strategy` 생략 혹은 `pairwise` 지정 시 기존 동작과 완벽히 동일.
* **모델 자동감지**(`--reranker-mode auto`):

  * 특수토큰(+프로젝터 가중치) 감지 실패 시 기존 시퀀스 분류형 경로로 폴백(424는 **잘못된 모델일 때만** 반환).
* **페이로드 제한(413)**: 긴 컨텍스트를 보내는 listwise 특성상 `--payload-limit` 상향 및 문서 청킹 로직으로 대응. ([GitHub][2])

---

## 4. 성능/자원

* **토큰 상한**: 모델 카드/논문 명세 기준(예: 131k)보다 **안전 마진** 10% 삭감하여 `ctx_budget` 계산. (문서 수 `k`, 평균 토큰 길이, 고정 템플릿 토큰 오버헤드 고려)
* **배칭**: 1차는 **요청 격리**(cross‑request listwise packing 미지원)로 간단/안전하게.
* **정밀도**: `bf16` 우선. 필요시 출력 BF16→FP32 라운드트립으로 점수 패리티 고정(참조 사례: 시퀀스클래시피케이션 변환 시 패리티 처리 관행).

---

## 5. 관찰성/메트릭

* Prometheus:

  * `tei_rerank_listwise_requests_total`
  * `tei_rerank_listwise_tokens_total`
  * `tei_rerank_listwise_latency_ms_bucket`
* Tracing span: `rerank_listwise.forward`, `rerank_listwise.prompt_build`, `rerank_listwise.tokenize`.

---

## 6. 테스트 계획

### 6.1 유닛

* `core/prompt.rs`: 특수토큰/샌드위치 배치/ID 매칭.
* `tokenization.rs`: left padding + `pad=unk` fallback, 특수토큰 위치 캡처.
* `layers/projector.rs`: 가중치 로드 및 projector 출력이 기대한 `cosine_similarity` 결과를 내는지 검증(고정 텐서로 골든).
* `listwise::weights`: 블록 가중 평균(`block_weights`)이 기대대로 `q`를 결합하는지 확인.

### 6.2 통합

* `integration_tests/tests/rerank_listwise.rs` (신규)

  * **모델 픽스처**: 최소형 Qwen3‑0.6B 체크포인트(또는 tiny weight) + projector weight(작은 난수) 업로드된 테스트 레포 사용.
  * **행동 검증**: `/rerank` `strategy=listwise` 호출 시 **문서 순서 불변성**, `max_pass_docs` 및 토큰 길이에 따른 청킹 트리거, `payload_limit` 근접 부하.

### 6.3 회귀

* 기존 `/rerank` pairwise 케이스가 스코어/정렬 동일 유지 확인.

---

## 7. 문서화

* `README.md` > “Using Re‑rankers models” 절에 **Listwise(LBNL) 추가 예시**:

  ```bash
  model=jinaai/jina-reranker-v3
  docker run --gpus all -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-embeddings-inference:latest \
    --model-id $model --reranker-mode auto --max-listwise-docs-per-pass 125
  ```

  ```bash
  curl 127.0.0.1:8080/rerank -X POST -H 'Content-Type: application/json' -d '{
    "query":"best way to sort arrays in python?",
    "texts":["doc1 ...", "doc2 ...", "..."],
    "strategy":"listwise"
  }'
  ```
* API 문서(gh-pages) `/rerank` 스키마에 `strategy`/`max_pass_docs` 옵션 추가. ([Hugging Face][5])

---

## 8. 마이그레이션/롤아웃

* **1차 릴리스**: Candle 전용 + 요청 격리.
* **2차**:

  * ORT 경로(ONNX 내 커스텀 `SelectHiddenStates`/`Gather` + MLP)
  * Cross‑request listwise 동적 배칭(세그먼트 마스크/블록 주의)
  * gRPC 클라이언트 샘플 추가.

---

## 9. 리스크 & 완화

* **모델 가용성**: Hugging Face Hub 상 `jina‑reranker‑v3` 공개/특수토큰 일치 필요 → 자동감지 실패 시 사용자 플래그 `--reranker-mode listwise`로 강제 가능.
* **컨텍스트 초과**: 긴 문서가 많을 때 413/토큰 초과 → 청킹 + 에러메시지/제안(문서 길이 제한/`max_pass_docs` 조정).
* **성능**: Cross‑request 배칭 미지원으로 처리량 저하 가능 → 2차에서 개선.

---

## 10. 상세 변경 파일 목록

> 아래는 **실제 리포 구조/파일명**을 기준으로 작성했습니다. 경로들은 TEI 레포의 공개 로그/이슈/README에서 확인된 구조를 따릅니다. ([GitHub][6])

```
router/
  src/
    lib.rs                     # ModelKind 추가, rerank 핸들러 listwise 분기, 감지 로직/설정
    main.rs                    # CLI 플래그 3종 추가 및 AppState 반영
core/
  src/
    prompt.rs                  # NEW: build_jina_v3_prompt()
    tokenization.rs            # encode_listwise() 추가/보강
backends/
  src/
    lib.rs                     # ModelKind::ListwiseReranker, detect_lbnl_reranker(), trait 확장
  candle/
    src/
      lbnl_reranker.rs         # NEW: Qwen3 CausalLM + hidden states + projector + cosine
      layers/projector.rs      # NEW: 2-layer MLP 로더/추론
proto/
  tei.proto                    # RerankRequest.strategy 필드/enum 추가
docs/ (또는 README)
  quick_tour / README.md       # Listwise 사용법/예시/제약 추가
integration_tests/
  tests/
    rerank_listwise.rs         # NEW
```

---

## 11. 의존성/빌드

* 새 파일만 추가(Candle/serde/axum/utoipa 기존 의존성 재사용).
* `Cargo.toml` 수정 없음(기능 게이트가 필요하다면 `feature = ["listwise"]` 가드 추가 가능).

---

## 12. 구현 메모(스코어 계산)

* 히든스테이트: 최종 레이어 `H ∈ R^{T×1024}`에서 **특수토큰 위치** `t_q, t_i`만 인덱싱.
* Projector:

  * `h → relu(h W1) → z = (· W2) ∈ R^{512}` (`W1: 1024×512`, `W2: 512×512`, bias 없음)
  * `q = z_q / ||z_q||2`, `d_i = z_i / ||z_i||2`
  * `s_i = <q, d_i>` (코사인 동일)
* 문서 > 컨텍스트 시 **125개 단위** 청킹(기본) + 토큰 용량(`length_capacity`) 검사, 동일 요청 내 결과 **연결**.

---

## 13. 레퍼런스 & 배경 링크

* TEI README(지원 모델/`/rerank` 엔드포인트/도커/플래그 등) ([GitHub][1])
* TEI API gh‑pages(`/rerank` 문서) ([Hugging Face][5])
* TEI 라우터/백엔드 경로 및 axum 사용(이슈/로그) ([GitHub][2])
* Qwen3/리랭커 지원 요구 이슈/PR(배경) ([GitHub][3])

---

### 부록 A. 간단한 Pseudo‑Diff

```diff
// router/src/lib.rs
 enum ModelKind {
   Embedding,
   SequenceClassifier,
+  ListwiseReranker,
 }

+fn detect_lbnl_reranker(repo: &ModelRepo) -> bool {
+  repo.tokenizer_has("<|embed_token|>") && repo.tokenizer_has("<|rerank_token|>")
+    && repo.weights_has("projector.0.weight")
+    && repo.weights_has("projector.2.weight")
+}

 async fn rerank(Json(req): Json<RerankRequest>, State(st): State<AppState>) -> Json<RerankResponse> {
-  backend.rerank_pairwise(req).await
+  let strategy = req.strategy.unwrap_or(st.default_strategy());
+  match (strategy, st.model_kind) {
+    (Strategy::Listwise, ModelKind::ListwiseReranker) |
+    (Strategy::Auto, ModelKind::ListwiseReranker) => st.backend.rerank_listwise(req).await?,
+    _ => st.backend.rerank_pairwise(req).await?,
+  }
 }
```

```diff
// backends/src/lib.rs
 pub enum ModelKind {
   Embedding,
   SequenceClassifier,
+  ListwiseReranker,
 }

 #[async_trait]
 pub trait RerankBackend {
   async fn rerank_pairwise(&self, req: RerankRequest) -> Result<RerankResponse>;
+  async fn rerank_listwise(&self, req: RerankRequest) -> Result<RerankResponse>;
 }
```

```diff
// proto/tei.proto
 message RerankRequest {
   string query = 1;
   repeated string texts = 2;
+  Strategy strategy = 3; // optional
 }
+enum Strategy {
+  STRATEGY_UNSPECIFIED = 0;
+  PAIRWISE = 1;
+  LISTWISE = 2;
+}
```

---

## 결론

본 PR은 **엔드포인트 변경을 최소화**하면서 **listwise(LBNL) 리랭킹**을 TEI에 자연스럽게 통합하는 것을 목표로 합니다.
1차 배포는 **Candle + 요청 격리**로 안전하게 제공하고, 2차에서 **ORT/배칭 고도화**를 진행하는 단계적 전략입니다.
이로써 **`jina‑v3‑reranker`** 같은 **문서간 조기 상호작용 기반** 모델도 TEI에서 **표준 `/rerank` API**로 운영할 수 있습니다.
