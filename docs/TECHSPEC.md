# TEI에 Jina v3 Listwise Reranker (LBNL) 추가 - 상세 기술 스펙

**Owner:** @you  
**Repository:** huggingface/text-embeddings-inference  
**Target Branch:** `main`  
**Implementation Scope:** Candle backend 우선 (ORT는 후속 작업), HTTP API 중심  
**Version:** 1.0 (초안)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [목표 및 비목표](#2-목표-및-비목표)
3. [배경 및 관련 작업](#3-배경-및-관련-작업)
4. [아키텍처 설계](#4-아키텍처-설계)
5. [API 및 CLI 인터페이스](#5-api-및-cli-인터페이스)
6. [모델 로딩 및 감지](#6-모델-로딩-및-감지)
7. [프롬프트 구성](#7-프롬프트-구성)
8. [추론 파이프라인](#8-추론-파이프라인)
9. [스코어 계산 로직](#9-스코어-계산-로직)
10. [코드 구조 및 파일 변경](#10-코드-구조-및-파일-변경)
11. [성능 및 자원 관리](#11-성능-및-자원-관리)
12. [테스트 계획](#12-테스트-계획)
13. [문서화](#13-문서화)
14. [보안 및 에러 처리](#14-보안-및-에러-처리)
15. [마이그레이션 및 롤아웃](#15-마이그레이션-및-롤아웃)
16. [위험 요소 및 완화 전략](#16-위험-요소-및-완화-전략)
17. [참고 자료](#17-참고-자료)

---

## 1. 프로젝트 개요

### 1.1 요약

본 프로젝트는 Hugging Face의 Text Embeddings Inference (TEI) 서버에 **Listwise Reranking** 방식을 지원하는 새로운 백엔드를 추가하는 작업입니다. 구체적으로 `jina-reranker-v3` 모델과 같은 **LBNL (Listwise Batch-aware Neural Listwise) Reranker**를 1st-class로 지원하여, 기존의 pairwise(문서별) 재랭킹 방식과 병존하도록 합니다.

### 1.2 핵심 특징

- **Listwise 추론**: 여러 문서를 단일 컨텍스트에서 함께 인코딩하여, 문서 간 상호작용을 통한 더 정확한 재랭킹
- **기존 API 호환**: TEI의 `/rerank` 엔드포인트를 그대로 활용하며 내부 전략만 확장
- **Qwen3 백본 기반**: CausalLM 아키텍처에 특수 토큰 위치의 hidden state를 추출하여 MLP projector를 통과시킨 후 cosine similarity로 점수 산출
- **효율적인 배칭**: 문서 그룹을 지능적으로 묶어 토큰 예산 내에서 처리

---

## 2. 목표 및 비목표

### 2.1 Goals (목표)

1. **Listwise 재랭커의 1st-class 지원**
    - `jina-reranker-v3`와 동일한 방식의 listwise 재랭킹을 TEI에서 네이티브로 제공
    - Qwen3 CausalLM 백본에서 특수 토큰 위치의 hidden state를 추출하여 점수 계산

2. **기존 `/rerank` API와 완전 호환**
    - 요청·응답 스키마 유지
    - 내부적으로만 listwise 처리 방식 적용
    - 기존 pairwise 경로와 병존

3. **정확한 모델 구현**
    - **특수 토큰**: `<|embed_token|>` (문서용), `<|rerank_token|>` (쿼리용)
    - **MLP Projector**: 1024 → 512 → 512 (ReLU, bias 없음)
    - **점수 계산**: 정규화된 벡터 간 cosine similarity
    - **좌측 패딩**: Qwen3 특성에 맞춘 left padding 적용

4. **프롬프트 템플릿 정확성**
    - 모델 카드 및 논문(Jina Reranker v3)에 정의된 형식 준수
    - 시스템/사용자/어시스턴트 역할 구분
    - 문서 샌드위치 패턴 (쿼리가 프롬프트와 끝에 모두 등장)

5. **효율적인 그룹 처리**
    - 긴 입력과 다수 문서를 처리하기 위한 listwise 그룹 배칭
    - 토큰 예산 기반 동적 청킹
    - 블록별 쿼리 임베딩 가중 평균 결합

6. **Candle 백엔드 우선 구현**
    - TEI에서 이미 지원하는 Qwen3 임베딩 인프라 재사용
    - 히든 스테이트 추출 기능 추가

### 2.2 Non-Goals (비목표)

1. **기존 경로 대체 없음**
    - SequenceClassification 기반 크로스 인코더는 그대로 유지
    - Pairwise와 listwise는 병존

2. **학습/미세튜닝 미지원**
    - 서빙(inference) 전용
    - 모델 가중치 변환 작업 범위 외

3. **1차 버전 제외 사항**
    - ORT/ONNX 백엔드 지원 (2차 목표)
    - VLLM 통합
    - Gaudi(HPU) 배포
    - Cross-request 동적 배칭 (서로 다른 요청의 문서가 같은 컨텍스트에서 상호작용하는 고급 배칭)
    - LoRA 또는 컴파일 최적화
    - gRPC 프로토콜 확장 (1차는 HTTP만)

---

## 3. 배경 및 관련 작업

### 3.1 TEI 현재 상태

- **지원 기능**: TEI는 `/rerank` HTTP 엔드포인트를 제공하며, SequenceClassification 기반 크로스 인코더를 통한 pairwise 재랭킹을 지원합니다
- **아키텍처**: `axum` 기반 라우터와 `backends` 크레이트(Candle/ORT)로 구성
- **Qwen3 지원**: TEI는 이미 Qwen3 계열 임베딩 모델을 Candle로 지원하므로 백본 인프라 재사용 가능

### 3.2 Jina Reranker v3의 특징

- **아키텍처**: Qwen3 CausalLM을 백본으로 사용
- **Listwise 방식**: 쿼리와 여러 문서를 하나의 긴 시퀀스로 인코딩하여, 문서 간 early interaction 활용
- **특수 토큰**:
    - `<|embed_token|>` (ID: 151670) - 각 문서 뒤에 배치
    - `<|rerank_token|>` (ID: 151671) - 쿼리 뒤에 배치
- **Projector**: 2층 MLP (1024→512→512, ReLU, bias 없음)
- **블록 처리**: 긴 리스트를 125개 단위로 나누어 처리한 후 가중 평균으로 결합

### 3.3 관련 이슈

- Qwen3 계열 리랭커 지원 요청
- 토크나이저 fast tokenizer 요구사항
- 페이로드 크기 제한 (413 에러) 관련 토론

---

## 4. 아키텍처 설계

### 4.1 전체 흐름도

```
사용자 요청 (HTTP /rerank)
    ↓
라우터 (strategy 분기)
    ↓
┌─────────────┬─────────────┐
│  Pairwise   │  Listwise   │
│  (기존)     │   (신규)    │
└─────────────┴─────────────┘
         ↓            ↓
   SequenceClassifier  LbnlReranker
         ↓            ↓
    문서별 점수    그룹화 → 프롬프트 구성
                    ↓
              Qwen3 CausalLM Forward
                    ↓
            Hidden States 추출
                    ↓
            특수 토큰 위치 인덱싱
                    ↓
            MLP Projector 적용
                    ↓
            Cosine Similarity
                    ↓
              블록 결합 (가중 평균)
                    ↓
              최종 점수 배열
                    ↓
               JSON 응답
```

### 4.2 컴포넌트 구조

#### 4.2.1 라우터 레이어
- 요청 파싱 및 검증
- 모델 종류 감지 (ModelKind 열거형)
- 전략 분기 (auto/pairwise/listwise)
- 동적 배칭 제어

#### 4.2.2 코어 레이어
- 프롬프트 빌더 (템플릿 관리)
- 토크나이저 래퍼 (좌측 패딩, 특수 토큰 처리)
- 입력 sanitization (특수 토큰 인젝션 방지)

#### 4.2.3 백엔드 레이어
- **LbnlReranker** (신규):
    - Qwen3 CausalLM 로딩
    - Hidden states 추출
    - Projector MLP 로딩 및 추론
    - Cosine similarity 계산
    - 블록 가중 평균

---

## 5. API 및 CLI 인터페이스

### 5.1 HTTP API

#### 5.1.1 엔드포인트

기존 `/rerank` 엔드포인트를 그대로 사용하되, 내부 처리 방식만 확장합니다.

**URL**: `POST /rerank`

#### 5.1.2 요청 스키마 (Request JSON)

```json
{
  "query": "What is Deep Learning?",
  "texts": [
    "Deep learning is a subset of machine learning...",
    "Neural networks are computing systems...",
    "Artificial intelligence encompasses..."
  ]
}
```

**필드 설명**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `query` | string | Yes | 검색 쿼리 문자열 |
| `texts` | array[string] | Yes | 재랭킹할 문서 목록 (순서 유지) |

**참고**: 기존 pairwise 방식과 완전히 동일한 요청 형식을 유지합니다.

#### 5.1.3 응답 스키마 (Response JSON)

```json
{
  "results": [
    { "index": 2, "score": 0.856 },
    { "index": 0, "score": 0.742 },
    { "index": 1, "score": 0.631 }
  ]
}
```

**필드 설명**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `results` | array[object] | 점수순으로 정렬된 결과 배열 |
| `results[].index` | integer | 원본 `texts` 배열의 인덱스 (0-based) |
| `results[].score` | float | 재랭킹 점수 (높을수록 관련성 높음) |

**참고**: 기존 응답 형식과 완전히 동일하며, listwise 처리 여부는 내부적으로만 결정됩니다.

### 5.2 CLI 플래그

TEI 서버 실행 시 다음 플래그들을 추가하여 listwise 재랭킹 동작을 제어합니다.

#### 5.2.1 전략 제어

```bash
--reranker-mode <MODE>
```

- **설명**: 재랭킹 전략을 지정합니다
- **가능한 값**:
    - `auto` (기본값): 모델 메타데이터를 분석하여 자동 선택
        - 특수 토큰(`<|embed_token|>`, `<|rerank_token|>`)과 projector 가중치가 모두 존재하면 `listwise`
        - 그렇지 않으면 기존 `pairwise`
    - `pairwise`: 강제로 pairwise 모드 사용 (문서별 개별 처리)
    - `listwise`: 강제로 listwise 모드 사용 (모델이 지원해야 함)

**예시**:
```bash
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id jinaai/jina-reranker-v3 \
  --reranker-mode auto
```

#### 5.2.2 Listwise 파라미터

```bash
--max-listwise-docs-per-pass <INTEGER>
```

- **설명**: Listwise 방식에서 한 번의 forward pass에 포함할 최대 문서 수
- **기본값**: `125`
- **제약**: `1 ≤ value ≤ 125`
- **동작**:
    - 문서가 이 값을 초과하면 여러 블록으로 나누어 처리
    - 각 블록의 결과를 가중 평균으로 결합
    - 토큰 길이 제약에 따라 실제로는 더 적은 문서가 포함될 수 있음

**예시**:
```bash
--model-id jinaai/jina-reranker-v3 \
--reranker-mode listwise \
--max-listwise-docs-per-pass 100
```

#### 5.2.3 프롬프트 템플릿

```bash
--default-prompt-name <TEMPLATE_NAME>
```

- **설명**: 사용할 프롬프트 템플릿 이름
- **기본값**: `jina_v3_lbnl` (Listwise 모델 감지 시 자동 적용)
- **가능한 값**:
    - `jina_v3_lbnl`: Jina Reranker v3 공식 템플릿
    - (향후 다른 템플릿 추가 가능)

또는 커스텀 템플릿을 직접 지정:

```bash
--default-prompt '{...custom template...}'
```

**예시**:
```bash
--model-id jinaai/jina-reranker-v3 \
--default-prompt-name jina_v3_lbnl
```

#### 5.2.4 문서 순서 제어

```bash
--rerank-ordering <MODE>
```

- **설명**: 입력 문서의 순서 처리 방식
- **기본값**: `input`
- **가능한 값**:
    - `input`: 원본 순서 유지 (deterministic)
    - `random`: 모델 입력 전 랜덤 셔플 (실험용)

**주의**: `random` 모드는 재현성을 해치므로 프로덕션에서는 권장하지 않습니다. 모델 성능 테스트 목적으로만 사용하세요. `modeling.py` 경로에는 해당 옵션이 없으므로, `random`을 사용할 경우 요청마다 **점수/순위가 달라지는 것이 정상**임을 README/API 문구로 명시합니다.

**예시**:
```bash
--model-id jinaai/jina-reranker-v3 \
--rerank-ordering input
```

#### 5.2.5 추가 지시사항 (향후 확장)

```bash
--rerank-instruction <STRING>
```

- **설명**: 모든 재랭킹 요청에 적용할 추가 지시사항
- **기본값**: (없음)
- **동작**: 프롬프트의 `<instruct>` 블록에 삽입
- **참고**: Jina reference 구현에는 없으나 TEI 확장 기능으로 추가 가능

**예시**:
```bash
--rerank-instruction "Focus on technical accuracy and relevance."
```

#### 5.2.6 페이로드 크기 제한

```bash
--listwise-payload-limit-bytes <INTEGER>
```

- **설명**: Listwise 요청의 최대 페이로드 크기 (바이트)
- **기본값**: `2000000` (2MB)
- **권장**: 긴 문서를 많이 처리하는 경우 상향 조정
- **관련**: HTTP 413 에러 방지

**예시**:
```bash
--listwise-payload-limit-bytes 5000000  # 5MB
```

### 5.3 전체 CLI 예시

```bash
model=jinaai/jina-reranker-v3
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id $model \
  --reranker-mode auto \
  --max-listwise-docs-per-pass 125 \
  --default-prompt-name jina_v3_lbnl \
  --rerank-ordering input \
  --listwise-payload-limit-bytes 2000000
```

### 5.4 gRPC 프로토콜

**1차 버전에서는 gRPC 스키마를 변경하지 않습니다.**

- 기존 `Rerank` RPC를 그대로 유지
- 서버 내부에서 모델 자동 감지(`auto` 모드)로 pairwise/listwise 경로를 분기
- gRPC 확장은 2차 목표로 후속 PR에서 별도 제안

---

## 6. 모델 로딩 및 감지

### 6.1 LBNL 모델 감지 규칙

모델이 다음 **모든 조건**을 만족할 때 Listwise Reranker로 감지합니다:

#### 6.1.1 조건 1: 아키텍처

`config.json`의 다음 필드 중 하나를 만족:

```json
{
  "architectures": ["QwenForCausalLM"],  // 또는
  "architectures": ["Qwen3ForCausalLM"], // 또는
  "architectures": ["JinaForRanking"],   // Qwen3 기반 커스텀
  "model_type": "qwen3"
}
```

#### 6.1.2 조건 2: Projector 가중치

모델 리포지토리에 다음 가중치 키가 존재해야 합니다:

- `projector.0.weight` (1024 × 512, 첫 번째 Linear layer)
- `projector.2.weight` (512 × 512, 두 번째 Linear layer)

**참고**:
- Bias 파라미터는 없습니다 (`projector.0.bias`, `projector.2.bias` 없음)
- Safetensors 샤딩된 경우 `model.safetensors.index.json`도 검사

**검사 코드 스케치**:
```rust
fn has_projector_weights(repo: &ModelRepo) -> bool {
    repo.has_weight("projector.0.weight") && 
    repo.has_weight("projector.2.weight") &&
    !repo.has_weight("projector.0.bias") &&
    !repo.has_weight("projector.2.bias")
}
```

#### 6.1.3 조건 3: 특수 토큰

토크나이저 구성 파일에서 다음 특수 토큰 문자열을 찾아야 합니다:

- `<|embed_token|>` (문서용)
- `<|rerank_token|>` (쿼리용)

**검사 파일**:
- `tokenizer.json`
- `special_tokens_map.json`
- `added_tokens.json`

**중요**:
- **원본 구현은 토큰 ID(151670, 151671)를 하드코딩하지만, TEI에서는 문자열→ID 매핑을 통해 런타임 조회** (모델 파생형/커스텀 토크나이저 대응)
- 런타임에 토크나이저 API를 통해 문자열→ID 매핑을 수행:
  ```rust
  let embed_token_id = tokenizer.token_to_id("<|embed_token|>")
      .ok_or("embed_token not found")?;
  let rerank_token_id = tokenizer.token_to_id("<|rerank_token|>")
      .ok_or("rerank_token not found")?;
  ```

### 6.2 모델 종류 열거형

`router/src/lib.rs`에 새로운 모델 타입 추가:

```rust
pub enum ModelKind {
    Embedding,
    SequenceClassifier,
    ListwiseReranker,  // 신규
}
```

### 6.3 감지 로직 구현

```rust
fn detect_model_kind(repo: &ModelRepo, tokenizer: &Tokenizer) -> Result<ModelKind> {
    // Listwise 우선 검사
    if is_qwen_architecture(repo)? 
       && has_projector_weights(repo)? 
       && has_special_tokens(tokenizer)? {
        return Ok(ModelKind::ListwiseReranker);
    }
    
    // 기존 SequenceClassifier 검사
    if is_sequence_classifier(repo)? {
        return Ok(ModelKind::SequenceClassifier);
    }
    
    // 기본값: Embedding
    Ok(ModelKind::Embedding)
}

fn is_qwen_architecture(repo: &ModelRepo) -> Result<bool> {
    let config = repo.load_config()?;
    let arch = config["architectures"].as_array()
        .and_then(|a| a.first())
        .and_then(|v| v.as_str());
    
    Ok(matches!(arch,
        Some("QwenForCausalLM") |
        Some("Qwen3ForCausalLM") |
        Some("JinaForRanking")
    ))
}

fn has_special_tokens(tokenizer: &Tokenizer) -> Result<bool> {
    Ok(tokenizer.token_to_id("<|embed_token|>").is_some() &&
       tokenizer.token_to_id("<|rerank_token|>").is_some())
}

// 참고: 일부 파생 리포에서는 `model_type`이 다른 문자열일 수 있으므로, 위 `architectures` 매칭만으로 판정한다.
```

### 6.4 토크나이저 요구사항

**Fast Tokenizer 필수**: TEI 정책상 fast tokenizer만 지원합니다.

- 모델에 fast tokenizer가 없으면 **로드 실패** 처리
- 에러 메시지: `"Fast tokenizer is required for TEI. Model does not provide a fast tokenizer."`

**패딩 설정**:

- pad 토큰이 미정의인 경우 **pad=unk**로 강제 지정합니다. 이는 `modeling.py`와 동일한 정책으로, 대부분의 Qwen3 토크나이저가 `unk`를 제공한다는 가정에 기반합니다.
- `unk`가 존재하지 않는 극히 예외적인 토크나이저만 `pad=eos`로 폴백합니다.
```rust
// Qwen3 특성에 맞춘 좌측 패딩
tokenizer.padding_side = PaddingSide::Left;

// pad_token 이 없으면 pad=unk 로 강제 설정 (모델링 구현과 동일)
if tokenizer.pad_token.is_none() {
    if let Some(unk) = tokenizer.unk_token.clone() {
        tokenizer.pad_token = Some(unk);
    } else if let Some(eos) = tokenizer.eos_token.clone() {
        // 예외적으로 unk 가 없는 토크나이저만 pad=eos 로 폴백
        tokenizer.pad_token = Some(eos);
    } else {
        return Err(anyhow!("Tokenizer must provide pad, unk, or eos token"));
    }
}
```

### 6.5 Long-Context 설정

Qwen3의 Long-RoPE 및 rope-scaling 설정은 기존 Qwen3 임베딩 경로의 방식을 재사용합니다:

- `config.json`의 `rope_scaling` 파라미터 로드
- 컨텍스트 길이: 모델의 `tokenizer.model_max_length`에 따름 (예: rope-scaling된 Qwen3는 100k 이상)
- 안전 마진: 로딩 시 사전 거부용으로 **model_max_length의 90%**를 사용합니다. 실제 청킹/길이 계산은 `modeling.py`와 동일한 조건(`block_size=125`, `length_capacity = max_len - 2 * query_len`, `length_capacity <= max_doc_len`)을 그대로 적용합니다.

---

## 7. 프롬프트 구성

### 7.1 Jina v3 LBNL 템플릿

Jina Reranker v3 논문(Section 3.2) 및 모델 카드에 명시된 공식 템플릿을 정확히 구현합니다.

#### 7.1.1 전체 템플릿 구조

```
<|im_start|>system
You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. If an instruction is provided, you should follow the instruction when determining the ranking.<|im_end|>
<|im_start|>user
I will provide you with {k} passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: {QUERY}
{INSTRUCTION_BLOCK?}
{PASSAGES}
<query>
{QUERY}<|rerank_token|>
</query>
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

#### 7.1.2 각 구성 요소 설명

**1. System 메시지** (원본 `modeling.py`와 동일):
```
<|im_start|>system
You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. If an instruction is provided, you should follow the instruction when determining the ranking.
<|im_end|>
```

**2. User 메시지 헤더**:
```
<|im_start|>user
I will provide you with {k} passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: {QUERY}
```

- `{k}`: 실제 문서 개수로 치환 (예: "5")
- `{QUERY}`: 검색 쿼리 문자열로 치환

**3. Instruction 블록 (선택사항)**:
```
<instruct>
{INSTRUCTION}
</instruct>\n
```

- CLI 플래그 `--rerank-instruction`이 제공된 경우에만 포함 (없으면 블록 자체 생략)
- 원본 구현과 동일하게 `</instruct>` 뒤에 개행이 한 줄 추가됩니다.

**4. Passages (문서 리스트)**:
```
<passage id="0">
{DOC_0}<|embed_token|>
</passage>
<passage id="1">
{DOC_1}<|embed_token|>
</passage>
...
<passage id="k-1">
{DOC_{k-1}}<|embed_token|>
</passage>
```

- 각 문서는 `<passage id="i">` 태그로 감싸며, **i는 0부터 시작**합니다.
- 각 문서 뒤에는 항상 특수 토큰 `<|embed_token|>`을 붙이고, `</passage>` 뒤에 개행(`\n`)이 추가되어 다음 문서와 구분됩니다.

**5. Query 블록**:
```
<query>
{QUERY}<|rerank_token|>
</query>
```

- 쿼리가 문서 리스트 뒤에 다시 등장 (샌드위치 패턴)
- 쿼리 끝에 특수 토큰 `<|rerank_token|>` 배치

**6. Assistant 메시지**:
```
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

- `</think>` 뒤에 빈 줄 두 개 (`\n\n`) 필수
- 이는 모델이 reasoning을 생성할 공간 (실제로는 사용 안 함)

### 7.2 프롬프트 빌더 구현

`core/src/prompt.rs` (신규 파일):

```rust
pub fn build_jina_v3_prompt(
    query: &str,
    docs: &[&str],
    instruction: Option<&str>,
) -> String {
    let k = docs.len();
    
    // Sanitize inputs (특수 토큰 제거)
    let query_clean = sanitize_input(query);
    let docs_clean: Vec<String> = docs.iter()
        .map(|d| sanitize_input(d))
        .collect();
    
    let mut prompt = String::new();
    
    // System message
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str("You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. If an instruction is provided, you should follow the instruction when determining the ranking.\n");
    prompt.push_str("<|im_end|>\n");
    
    // User message
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(&format!("I will provide you with {} passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: {}\n", k, query_clean));
    
    // Instruction block (optional)
    if let Some(instr) = instruction {
        prompt.push_str("<instruct>\n");
        prompt.push_str(instr);
        prompt.push_str("\n</instruct>\n");
    }
    
    // Passages
    for (i, doc) in docs_clean.iter().enumerate() {
    prompt.push_str(&format!("<passage id=\"{}\">\n", i));
        prompt.push_str(doc);
        prompt.push_str("<|embed_token|>\n</passage>\n");
    }
    
    // Query block
    prompt.push_str("<query>\n");
    prompt.push_str(&query_clean);
    prompt.push_str("<|rerank_token|>\n</query>\n");
    
    // Assistant message
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>assistant\n");
    prompt.push_str("<think>\n\n</think>\n\n");
    
    prompt
}

fn sanitize_input(text: &str) -> String {
    // 특수 토큰 인젝션 방지 (두 토큰만 제거)
    text.replace("<|embed_token|>", "")
        .replace("<|rerank_token|>", "")
}
```

> 라우터/상위 계층에서는 이미 sanitize된 문자열을 재사용하고, 중복 호출로 불필요한 문자열 복사를 만들지 않도록 주의합니다. (중복 호출은 기능상 문제는 없지만 퍼포먼스와 회귀 위험을 피하기 위해 **프롬프트 빌더 단일 지점**에서만 수행합니다.)

### 7.3 토큰화

`core/src/tokenization.rs`에 좌측 패딩 지원 추가:

```rust
pub fn encode_listwise(
    tokenizer: &Tokenizer,
    prompt: &str,
    max_length: Option<usize>,
) -> Result<Encoding> {
    // 좌측 패딩 설정 (이미 로더에서 설정되었지만 명시)
    let mut tokenizer = tokenizer.clone();
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Left,
        pad_id: tokenizer.token_to_id(tokenizer.pad_token.as_ref().unwrap()).unwrap(),
        ..Default::default()
    }));
    
    // 인코딩
    let encoding = tokenizer.encode(prompt, false)?;
    
    // 길이 체크
    if let Some(max_len) = max_length {
        if encoding.len() > max_len {
            return Err(anyhow!("Prompt exceeds max length: {} > {}", encoding.len(), max_len));
        }
    }
    
    Ok(encoding)
}
```

---

## 8. 추론 파이프라인

### 8.1 전체 흐름

1. **입력 수신**: HTTP 요청에서 query와 texts 추출
2. **전처리 & 클리핑**: 특수 토큰( `<|embed_token|>`, `<|rerank_token|>` 등) 제거 후, `tokenizer.encode(..., truncation=True)`로 **쿼리는 512 토큰, 각 문서는 2048 토큰**으로 클립하며, 여기서 얻은 **잘린 문자열(query′, docs′)**과 길이 정보를 구조체에 보관합니다.
3. **문서 그룹화**: 토큰 예산에 맞게 **잘린 문서 문자열(docs′)**을 기준으로 블록을 생성합니다 (`max_listwise_docs_per_pass`, 컨텍스트 용량 기준).
4. **블록별 처리**:
    - 프롬프트 구성
    - 토큰화 (좌측 패딩)
    - Qwen3 forward pass (hidden states 반환)
    - 특수 토큰 위치 인덱싱
    - Projector 적용
    - Cosine similarity 계산
5. **블록 결합**: 여러 블록의 결과를 가중 평균으로 통합
6. **최종 점수 산출**: 결합된 쿼리 임베딩과 모든 문서 임베딩으로 cosine 재계산
7. **응답 생성**: 점수를 정렬하여 JSON 반환

### 8.2 문서 그룹화 (Chunking)

`modeling.py`와 동일한 휴리스틱을 사용합니다.

1. `truncate_texts`로 쿼리를 최대 512 토큰, 각 문서를 최대 2048 토큰으로 자르고 **잘린 문자열과 길이 정보를 함께 반환**합니다.
2. `length_capacity = tokenizer.model_max_length - 2 * query_length`로 초기화합니다.
3. 문서를 순서대로 순회하며 각 문서의 토큰 수만큼 `length_capacity`를 차감하고 블록에 문서 인덱스를 추가합니다.
4. 다음 조건 중 하나라도 참이면 즉시 블록을 종료합니다.
   - 현재 블록 문서 수가 125개 이상
   - `length_capacity`가 2048 이하
5. 남은 문서가 있으면 마지막 블록으로 추가합니다.

#### 8.2.1 구현 스케치

```rust
struct TruncatedBatch {
    query: String,
    docs: Vec<String>,
    doc_lengths: Vec<usize>,
    query_length: usize,
}

fn truncate_texts(
    tokenizer: &Tokenizer,
    query: &str,
    docs: &[String],
    max_query_len: usize,
    max_doc_len: usize,
) -> TruncatedBatch {
    // ... (기존 구현과 동일하게 토큰화 후 decode)
}

struct DocumentChunker {
    tokenizer: Arc<Tokenizer>,
}

impl DocumentChunker {
    fn chunk(&self, batch: &TruncatedBatch) -> Vec<Vec<usize>> {
        let max_length = self.tokenizer.get_model_max_length();
        let max_doc_len = 2048;
        let block_size = 125;

        let mut blocks = Vec::new();
        let mut current = Vec::new();
        let mut remaining = max_length.saturating_sub(2 * batch.query_length);

        for (idx, &doc_tokens) in batch.doc_lengths.iter().enumerate() {
            current.push(idx);
            remaining = remaining.saturating_sub(doc_tokens);

            if current.len() >= block_size || remaining <= max_doc_len {
                blocks.push(current.clone());
                current.clear();
                remaining = max_length.saturating_sub(2 * batch.query_length);
            }
        }

        if !current.is_empty() {
            blocks.push(current);
        }

        blocks
    }
}
```

상위 `rerank_listwise` 경로는 잘린 텍스트를 전 구간에서 재사용합니다:

```rust
fn rerank_listwise(&self, req: RerankRequest) -> Result<RerankResponse> {
    let batch = truncate_texts(
        &self.tokenizer,
        &req.query,
        &req.texts,
        512,
        2048,
    );

    let blocks = self.chunker.chunk(&batch);

    for block in blocks {
        let block_docs: Vec<&str> = block
            .iter()
            .map(|&idx| batch.docs[idx].as_str())
            .collect();

        let block_result = self.process_block(&batch.query, &block_docs)?;
        // ... accumulate block outputs & map scores back using `idx`
    }

    // 최종 점수/정렬 단계에서만 원본 req.texts 인덱스를 사용
}
```

> **핵심**: 프롬프트 빌더에는 항상 `batch.query` 및 `batch.docs[idx]`를 전달하여 `modeling.py`와 동일한 길이/내용의 프롬프트를 보장합니다. 응답 생성 시에는 `idx` 값으로 원본 텍스트를 참조합니다.

### 8.3 블록별 Forward Pass

각 블록에 대해 (입력 `query` 및 `docs`는 모두 `truncate_texts`에서 반환된 잘린 문자열을 재사용):

```rust
fn process_block(
    &self,
    query: &str,
    docs: &[&str],  // 블록에 속한 문서만
) -> Result<BlockResult> {
    // 1. 프롬프트 구성
    let prompt = build_jina_v3_prompt(query, docs, self.instruction.as_deref());
    
    // 2. 토큰화
    let encoding = encode_listwise(&self.tokenizer, &prompt, Some(self.max_seq_len))?;
    let input_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();
    
    // 3. Forward pass (hidden states 반환)
    let hidden_states = self.model.forward_with_hidden_states(
        &Tensor::new(input_ids, &self.device)?,
        &Tensor::new(attention_mask, &self.device)?,
    )?;
    
    // 4. 특수 토큰 위치 찾기
    let embed_positions = find_all_token_positions(input_ids, self.embed_token_id);
    let rerank_position = find_single_token_position(input_ids, self.rerank_token_id)?;
    
    // 5. Hidden states 추출
    let h_query = hidden_states.index(&[rerank_position])?;  // [1, 1024]
    let h_docs: Vec<Tensor> = embed_positions.iter()
        .map(|&pos| hidden_states.index(&[pos]))
        .collect::<Result<Vec<_>>>()?;
    
    // 6. Projector 적용 (정규화는 cosine 계산 시 수행)
    let z_query = self.projector.forward(&h_query)?;          // [1, 512]
    let z_docs: Vec<Tensor> = h_docs.iter()
        .map(|h| self.projector.forward(h))
        .collect::<Result<Vec<_>>>()?;                        // 각 [1, 512]
    
    // 7. Cosine similarity (내부에서 L2 정규화 수행)
    let scores: Vec<f32> = z_docs.iter()
        .map(|z_doc| cosine_similarity(&z_query, z_doc))
        .collect::<Result<Vec<_>>>()?;
    
    Ok(BlockResult {
        query_embed: z_query,
        doc_embeds: z_docs,
        scores,
    })
}
```

---

## 9. 스코어 계산 로직

### 9.1 단일 블록 내 계산

블록 하나를 처리할 때의 정확한 수식입니다.

#### 9.1.1 Hidden State 추출

최종 레이어의 hidden states `H ∈ ℝ^(T×1024)`에서:

```
h_q = H[pos_<|rerank_token|>]     ∈ ℝ^1024
h_{d_i} = H[pos_<|embed_token|>_i] ∈ ℝ^1024  (i = 1..k)
```

**특수 토큰 위치 찾기**:
```rust
fn find_single_token_position(input_ids: &[u32], token_id: u32) -> Result<usize> {
    let positions: Vec<usize> = input_ids.iter()
        .enumerate()
        .filter_map(|(pos, &id)| if id == token_id { Some(pos) } else { None })
        .collect();
    match positions.as_slice() {
        [pos] => Ok(*pos),
        [] => Err(anyhow!("Token {} not found", token_id)),
        _ => Err(anyhow!("Token {} appears multiple times", token_id)),
    }
}

fn find_all_token_positions(input_ids: &[u32], token_id: u32) -> Vec<usize> {
    input_ids.iter()
        .enumerate()
        .filter_map(|(pos, &id)| if id == token_id { Some(pos) } else { None })
        .collect()
}
```

#### 9.1.2 MLP Projector

2층 신경망 (bias 없음):

```
Layer 1: Linear(1024 → 512, bias=False)
Layer 2: ReLU
Layer 3: Linear(512 → 512, bias=False)
```

수식:
```
a = h · W1^T              (W1: 512×1024)
r = ReLU(a)               (element-wise)
z = r · W2^T              (W2: 512×512)
```

구현:
```rust
pub struct Projector {
    linear1: Linear,  // 1024 → 512, bias=false
    linear2: Linear,  // 512 → 512, bias=false
}

impl Projector {
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let a = self.linear1.forward(h)?;
        let r = a.relu()?;
        let z = self.linear2.forward(&r)?;
        Ok(z)
    }
}
```

가중치 로딩:
```rust
fn load_projector(repo: &ModelRepo) -> Result<Projector> {
    let w1 = repo.load_tensor("projector.0.weight")?;  // [512, 1024]
    let w2 = repo.load_tensor("projector.2.weight")?;  // [512, 512]
    
    Ok(Projector {
        linear1: Linear::new(w1, None),  // bias=None
        linear2: Linear::new(w2, None),
    })
}
```

#### 9.1.3 Cosine Similarity (명시적 L2 정규화)

안정적인 결과를 위해 projector 출력에 대해 명시적으로 L2 정규화를 수행한 뒤 cosin 값을 계산합니다 (ε 안정화 포함).

```
score_i = <q̂, d̂_i> = Σ(q̂_j · d̂_{i,j}) / (‖q‖₂ · ‖d_i‖₂)
```

구현:
```rust
fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let eps = 1e-8f32;
    let an = a.sqr()?.sum_keepdim(-1)?.sqrt()?.add_scalar(eps)?;
    let bn = b.sqr()?.sum_keepdim(-1)?.sqrt()?.add_scalar(eps)?;
    let dot = (a * b)?.sum_keepdim(-1)?;
    Ok(dot.div(&an)?.div(&bn)?.to_vec1::<f32>()?[0])
}
```

> **Parity Note**: PyTorch의 `torch.nn.functional.cosine_similarity`와 동일한 정규화를 수행하지만, Candle 연산 순서 차이로 `1e-6` 내외 부동소수 오차가 발생할 수 있습니다. 이는 `modeling.py` 대비 허용 가능한 범위이며, 골든 테스트를 설계할 때 `rtol=1e-5`, `atol=1e-6` 허용치를 권장합니다.

### 9.2 다중 블록 결합

문서가 `max_listwise_docs_per_pass`를 초과하여 여러 블록으로 나뉜 경우:

#### 9.2.1 블록 가중치 계산

각 블록 `b`에 대해:

```
max_score_b = max(scores_b)
block_weight_b = (1 + max_score_b) / 2
```

**근거**:
- 점수가 높은 블록에 쿼리 임베딩이 더 정확하다고 가정
- Cosine similarity 범위 [-1, 1]을 [0, 1]로 정규화

구현:
```rust
let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
let block_weight = (1.0 + max_score) / 2.0;
```

#### 9.2.2 쿼리 임베딩 가중 평균

```
q̄ = (Σ w_b · z_q_b) / (Σ w_b)
```

where:
- `w_b` = block_weight for block b
- `z_q_b` = query embedding from block b (shape: [512])

구현:
```rust
fn combine_query_embeddings(
    query_embeds: Vec<Tensor>,     // Vec of [1, 512]
    block_weights: Vec<f32>,
) -> Result<Tensor> {
    let total_weight: f32 = block_weights.iter().sum();
    
    let weighted_sum = query_embeds.iter()
        .zip(block_weights.iter())
        .try_fold(
            Tensor::zeros(&[1, 512], DType::F32, &device)?,
            |acc, (emb, &w)| {
                acc.add(&emb.mul_scalar(w)?)
            }
        )?;
    
    weighted_sum.div_scalar(total_weight)
}
```

#### 9.2.3 최종 점수 재계산

**중요**: 블록별로 계산된 점수는 **보조 정보**(블록 가중치 계산용)일 뿐입니다.

최종 점수는 반드시 다음 방식으로 계산:

```
q̄ = weighted_average(query_embeddings)  // from step 9.2.2
final_score_i = cosine(q̄, z_{d_i})
```

**모든 문서**에 대해 결합된 쿼리 임베딩 `q̄`와 각 문서 임베딩으로 cosine을 다시 계산합니다.

구현:
```rust
fn finalize_scores(
    combined_query: &Tensor,        // [1, 512]
    all_doc_embeds: Vec<Tensor>,    // Vec of [1, 512]
) -> Result<Vec<f32>> {
    all_doc_embeds.iter()
        .map(|doc_emb| cosine_similarity(combined_query, doc_emb))
        .collect()
}
```

### 9.3 전체 `rerank_listwise` 메서드

```rust
impl LbnlReranker {
    pub async fn rerank_listwise(&self, req: RerankRequest) -> Result<RerankResponse> {
        let query = &req.query;
        let docs: Vec<&str> = req.texts.iter().map(|s| s.as_str()).collect();
        
        // 1. 문서 그룹화
        let blocks = self.chunker.chunk(query, &req.texts);
        
        // 2. 블록별 처리
        let mut all_query_embeds = Vec::new();
    let mut all_doc_embeds = Vec::new();
    let mut all_doc_indices = Vec::new();
        let mut block_weights = Vec::new();
        
        for block_indices in blocks {
            let block_docs: Vec<&str> = block_indices.iter()
                .map(|&i| docs[i])
                .collect();
            
            let result = self.process_block(query, &block_docs)?;
            
            // 블록 가중치 계산
            let max_score = result.scores.iter().cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let weight = (1.0 + max_score) / 2.0;
            
            all_query_embeds.push(result.query_embed);
        all_doc_embeds.extend(result.doc_embeds);
        all_doc_indices.extend(block_indices.iter().copied());
            block_weights.push(weight);
        }
        
        // 3. 쿼리 임베딩 결합
        let combined_query = combine_query_embeddings(all_query_embeds, block_weights)?;
        
        // 4. 최종 점수 계산
        let final_scores = finalize_scores(&combined_query, all_doc_embeds)?;
        
        // 5. 응답 생성 (점수순 정렬)
    Ok(RerankResponse::from_scores_with_indices(&req.texts, final_scores, &all_doc_indices))
    }
}
```
> `from_scores_with_indices`는 `(score, original_index)` 쌍을 사용해 응답을 구성하므로, `ordering=random`처럼 입력 문서를 셔플한 경우에도 점수와 원본 텍스트의 매핑이 정확히 복원됩니다.

---

## 10. 코드 구조 및 파일 변경

### 10.1 디렉토리 구조

```
text-embeddings-inference/
├── router/
│   └── src/
│       ├── lib.rs                    # 수정: ModelKind 추가, 핸들러 분기
│       └── main.rs                   # 수정: CLI 플래그 추가
├── core/
│   └── src/
│       ├── prompt.rs                 # 신규: 프롬프트 빌더
│       └── tokenization.rs           # 수정: encode_listwise 추가
├── backends/
│   ├── src/
│   │   └── lib.rs                    # 수정: 트레이트 확장, 감지 로직
│   └── candle/
│       └── src/
│           ├── lbnl_reranker.rs      # 신규: Listwise 백엔드 구현
│           └── layers/
│               └── projector.rs      # 신규: MLP Projector
├── proto/
│   └── tei.proto                     # 미변경 (1차)
└── integration_tests/
    └── tests/
        └── rerank_listwise.rs        # 신규: 통합 테스트
```

### 10.2 주요 파일 변경 내역

#### 10.2.1 `router/src/lib.rs`

**변경 사항**:

1. **ModelKind 열거형 확장**
```rust
pub enum ModelKind {
    Embedding,
    SequenceClassifier,
    ListwiseReranker,  // 추가
}
```

2. **모델 감지 함수 추가**
```rust
fn detect_model_kind(repo: &ModelRepo, tokenizer: &Tokenizer) -> Result<ModelKind> {
    // LBNL 감지 로직 (§6.3 참조)
    if is_listwise_reranker(repo, tokenizer)? {
        return Ok(ModelKind::ListwiseReranker);
    }
    
    // 기존 로직
    if is_sequence_classifier(repo)? {
        return Ok(ModelKind::SequenceClassifier);
    }
    
    Ok(ModelKind::Embedding)
}

fn is_listwise_reranker(repo: &ModelRepo, tokenizer: &Tokenizer) -> Result<bool> {
    Ok(
        is_qwen_architecture(repo)? &&
        has_projector_weights(repo)? &&
        has_special_tokens(tokenizer)?
    )
}
```

3. **Rerank 핸들러 수정**
```rust
async fn rerank_handler(
    Json(req): Json<RerankRequest>,
    State(state): State<AppState>,
) -> Result<Json<RerankResponse>, ApiError> {
    // 전략 결정
    let strategy = determine_strategy(
        state.cli_flags.reranker_mode,
        state.model_kind,
    )?;
    
    // 분기 처리
    match (strategy, state.model_kind) {
        (Strategy::Listwise, ModelKind::ListwiseReranker) => {
            state.backend.rerank_listwise(req).await
        }
        (Strategy::Pairwise, _) | (_, ModelKind::SequenceClassifier) => {
            state.backend.rerank_pairwise(req).await
        }
        _ => {
            Err(ApiError::InvalidConfiguration(
                "Model does not support requested strategy".to_string()
            ))
        }
    }
}

fn determine_strategy(cli_mode: CliRerankMode, model_kind: ModelKind) -> Result<Strategy> {
    match cli_mode {
        CliRerankMode::Auto => match model_kind {
            ModelKind::ListwiseReranker => Ok(Strategy::Listwise),
            _ => Ok(Strategy::Pairwise),
        },
        CliRerankMode::Pairwise => Ok(Strategy::Pairwise),
        CliRerankMode::Listwise => Ok(Strategy::Listwise),
    }
}
```

4. **동적 배칭 제어**
```rust
// Listwise 요청은 격리 (1차 버전)
if state.model_kind == ModelKind::ListwiseReranker {
    // 요청당 독립 처리 (cross-request batching 비활성화)
    state.batcher.set_max_batch_requests(1);
}
```

#### 10.2.2 `router/src/main.rs`

**CLI 플래그 추가**:

```rust
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    // ... 기존 플래그들 ...
    
    /// Reranking strategy mode
    #[clap(long, env, value_enum, default_value = "auto")]
    reranker_mode: RerankMode,
    
    /// Maximum documents per listwise pass
    #[clap(long, env, default_value = "125")]
    max_listwise_docs_per_pass: usize,
    
    /// Document ordering for reranking
    #[clap(long, env, value_enum, default_value = "input")]
    rerank_ordering: OrderingMode,
    
    /// Default prompt template name for listwise reranking
    #[clap(long, env)]
    default_prompt_name: Option<String>,
    
    /// Custom prompt template string
    #[clap(long, env)]
    default_prompt: Option<String>,
    
    /// Additional instruction for all rerank requests
    #[clap(long, env)]
    rerank_instruction: Option<String>,
    
    /// Maximum payload size for listwise requests (bytes)
    #[clap(long, env, default_value = "2000000")]
    listwise_payload_limit_bytes: usize,
}

#[derive(Debug, Clone, ValueEnum)]
enum RerankMode {
    Auto,
    Pairwise,
    Listwise,
}

#[derive(Debug, Clone, ValueEnum)]
enum OrderingMode {
    Input,
    Random,
}
```

**AppState 구성**:

```rust
let app_state = AppState {
    model_kind,
    backend,
    cli_flags: CliFlags {
        reranker_mode: args.reranker_mode,
        max_listwise_docs_per_pass: args.max_listwise_docs_per_pass,
        rerank_ordering: args.rerank_ordering,
        default_prompt_name: args.default_prompt_name,
        rerank_instruction: args.rerank_instruction,
        listwise_payload_limit_bytes: args.listwise_payload_limit_bytes,
    },
    // ...
};
```

#### 10.2.3 `core/src/prompt.rs` (신규)

전체 구현은 §7.2 참조.

주요 함수:
- `build_jina_v3_prompt()`: 메인 빌더
- `sanitize_input()`: 특수 토큰 제거
- `estimate_prompt_tokens()`: 토큰 수 예측 (청킹용)

#### 10.2.4 `core/src/tokenization.rs`

**추가 함수**:

```rust
pub fn encode_listwise(
    tokenizer: &Tokenizer,
    prompt: &str,
    max_length: Option<usize>,
) -> Result<Encoding> {
    // §7.3 참조
}

pub fn count_tokens(tokenizer: &Tokenizer, text: &str) -> usize {
    tokenizer.encode(text, false)
        .map(|enc| enc.len())
        .unwrap_or(0)
}
```

#### 10.2.5 `backends/src/lib.rs`

**트레이트 확장**:

```rust
#[async_trait]
pub trait RerankBackend: Send + Sync {
    async fn rerank_pairwise(&self, req: RerankRequest) -> Result<RerankResponse>;
    
    async fn rerank_listwise(&self, req: RerankRequest) -> Result<RerankResponse> {
        // 기본 구현: 미지원
        Err(anyhow!("Listwise reranking not supported by this backend"))
    }
}
```

**감지 함수 추가**:

```rust
pub fn detect_lbnl_reranker(repo: &ModelRepo, tokenizer: &Tokenizer) -> Result<bool> {
    // §6.3의 is_listwise_reranker와 동일
}

fn is_qwen_architecture(repo: &ModelRepo) -> Result<bool> {
    // §6.3 참조
}

fn has_projector_weights(repo: &ModelRepo) -> Result<bool> {
    repo.has_weight("projector.0.weight") &&
    repo.has_weight("projector.2.weight") &&
    !repo.has_weight("projector.0.bias") &&
    !repo.has_weight("projector.2.bias")
}

fn has_special_tokens(tokenizer: &Tokenizer) -> Result<bool> {
    tokenizer.token_to_id("<|embed_token|>").is_some() &&
    tokenizer.token_to_id("<|rerank_token|>").is_some()
}
```

#### 10.2.6 `backends/candle/src/lbnl_reranker.rs` (신규)

전체 구현은 §8.3, §9 참조.

**구조체**:

```rust
pub struct LbnlRerankerCandle {
    model: Qwen3CausalLm,
    tokenizer: Arc<Tokenizer>,
    projector: Projector,
    chunker: DocumentChunker,
    device: Device,
    
    // Config
    max_seq_len: usize,
    embed_token_id: u32,
    rerank_token_id: u32,
    instruction: Option<String>,
}
```

**주요 메서드**:

```rust
impl LbnlRerankerCandle {
    pub fn new(
        model_path: &Path,
        cli_flags: &CliFlags,
        device: Device,
    ) -> Result<Self> {
        // 모델/토크나이저/프로젝터 로딩
        // 특수 토큰 ID 조회
        // Chunker 초기화
    }
    
    fn process_block(&self, query: &str, docs: &[&str]) -> Result<BlockResult> {
        // §8.3 참조
    }
    
    async fn rerank_listwise(&self, req: RerankRequest) -> Result<RerankResponse> {
        // §9.3 참조
    }
}

#[async_trait]
impl RerankBackend for LbnlRerankerCandle {
    async fn rerank_listwise(&self, req: RerankRequest) -> Result<RerankResponse> {
        self.rerank_listwise(req).await
    }
}
```

#### 10.2.7 `backends/candle/src/layers/projector.rs` (신규)

```rust
use candle_core::{Tensor, Result};
use candle_nn::{Linear, Module};

pub struct Projector {
    linear1: Linear,  // 1024 → 512
    linear2: Linear,  // 512 → 512
}

impl Projector {
    pub fn new(linear1: Linear, linear2: Linear) -> Self {
        Self { linear1, linear2 }
    }
    
    pub fn load(repo: &ModelRepo, device: &Device) -> Result<Self> {
        let w1 = repo.load_tensor("projector.0.weight", device)?;
        let w2 = repo.load_tensor("projector.2.weight", device)?;
        
        // Bias 없음 확인
        if repo.has_tensor("projector.0.bias") || repo.has_tensor("projector.2.bias") {
            return Err(anyhow!("Projector should not have bias parameters"));
        }
        
        Ok(Self::new(
            Linear::new(w1, None),
            Linear::new(w2, None),
        ))
    }
    
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let a = self.linear1.forward(h)?;
        let r = a.relu()?;
        self.linear2.forward(&r)
    }
}
```

#### 10.2.8 Qwen3 모델 수정

기존 `backends/candle/src/models/qwen3.rs` 확장:

```rust
impl Qwen3CausalLm {
    // 기존 forward 메서드는 그대로 유지
    
    // 새 메서드 추가
    pub fn forward_with_hidden_states(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        
        // Embedding
        let mut hidden_states = self.model.embed_tokens.forward(input_ids)?;
        
        // Transformer layers
        for layer in &self.model.layers {
            hidden_states = layer.forward(
                &hidden_states,
                attention_mask,
                /* ... */
            )?;
        }
        
        // Final norm
        hidden_states = self.model.norm.forward(&hidden_states)?;
        
        // 최종 hidden states 반환 (LM head 통과하지 않음)
        Ok(hidden_states)  // [batch_size, seq_len, hidden_dim]
    }
}
```

- 반환되는 텐서는 `super().forward(..., output_hidden_states=true)`의 `hidden_states[-1]`(Transformers Python 구현에서 최종 LayerNorm을 거친 마지막 hidden state)과 1:1로 일치해야 합니다.
- dtype, device, attention mask 브로드캐스트 로직은 기존 임베딩 경로와 동일하게 유지하여 parity를 보장합니다.

---

## 11. 성능 및 자원 관리

### 11.1 계산 복잡도

#### 11.1.1 시간 복잡도

단일 블록 처리:
```
T = total_tokens = query_tokens + Σ(doc_tokens_i) + overhead

Self-Attention: O(T²)
MLP Projector: O(k × 1024 × 512) where k = num_docs
Cosine: O(k × 512)

Total: O(T²) dominant
```

#### 11.1.2 메모리 사용

```
Hidden States: T × 1024 × 4 bytes (fp32) or × 2 bytes (fp16/bf16)
Attention Cache: num_layers × T² × hidden_dim (if using KV cache, 현재는 미사용)
Projector Params: (1024×512 + 512×512) × 4 bytes ≈ 3 MB
```

**권장 설정**:
- **쿼리**: 최대 512 토큰
- **문서**: 각각 최대 2048 토큰
- **블록당 문서 수**: 125개 (기본값)
- **모델 정밀도**: bf16 (메모리 절약 + 성능 유지)

### 11.2 CLI 플래그 상호작용

기존 TEI 플래그들과의 관계:

```bash
# 시퀀스 길이 제한
--max-seq-len <길이>           # 모델의 `tokenizer.model_max_length`에 맞춘 상한 (예: 131072)

# Flash Attention (가능 시 자동 활용)
--flash-attn true

# RoPE Scaling
--rope-scaling linear        # Long-context 지원

# 정밀도
--dtype bfloat16             # 권장
```

### 11.3 메트릭 노출

Prometheus 메트릭 추가:

```rust
// registry.rs
lazy_static! {
    pub static ref LBNL_GROUP_SIZE: Histogram = register_histogram!(
        "tei_lbnl_group_size",
        "Number of documents per listwise group"
    ).unwrap();
    
    pub static ref LBNL_SEQ_TOKENS: Histogram = register_histogram!(
        "tei_lbnl_seq_tokens",
        "Total tokens in listwise sequence"
    ).unwrap();
    
    pub static ref LBNL_MS_PER_GROUP: Histogram = register_histogram!(
        "tei_lbnl_ms_per_group",
        "Processing time per group (ms)"
    ).unwrap();
    
    pub static ref LBNL_GROUPS_PER_REQ: Histogram = register_histogram!(
        "tei_lbnl_groups_per_req",
        "Number of groups per request"
    ).unwrap();
}
```

사용:
```rust
// process_block 내
let start = Instant::now();
let result = self.forward_once(query, docs)?;
LBNL_MS_PER_GROUP.observe(start.elapsed().as_millis() as f64);
LBNL_GROUP_SIZE.observe(docs.len() as f64);
LBNL_SEQ_TOKENS.observe(total_tokens as f64);
```

### 11.4 최적화 기회

**1차 버전에서 적용**:
- Flash Attention (Candle 지원 범위 내)
- bf16 정밀도
- 토큰 기반 동적 청킹

**2차 목표**:
- KV Cache 재사용 (동일 쿼리, 다른 문서 블록)
- Cross-request batching (세그먼트 마스크)
- ONNX 최적화 (ORT 백엔드)
- 컴파일 최적화 (TorchScript, TensorRT)

**추가 제안(선택)**:
- 토큰 예산 가드에서 템플릿 오버헤드를 토크나이저 기반으로 산출하여 413 false positive를 줄입니다.
- `<|embed_token|>` 검출 수가 `docs.len()`과 다를 경우 422를 반환하도록 process_block 경고 레일을 추가합니다.
- 수치 패리티 디버깅용 `--bf16-parity-roundtrip` 플래그를 실험적으로 제공하여 출력 스코어를 bf16→fp32로 강제 변환합니다 (기본 off).

---

## 12. 테스트 계획

### 12.1 유닛 테스트

#### 12.1.1 프롬프트 빌더 (`core/src/prompt.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_jina_v3_prompt_structure() {
        let query = "test query";
        let docs = vec!["doc1", "doc2"];
        let prompt = build_jina_v3_prompt(query, &docs, None);
        
        // 구조 검증
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("I will provide you with 2 passages"));
        assert!(prompt.contains("<passage id=\"0\">"));
        assert!(prompt.contains("doc1<|embed_token|>"));
        assert!(prompt.contains("<query>"));
        assert!(prompt.contains("test query<|rerank_token|>"));
        assert!(prompt.contains("<think>\n\n</think>\n\n"));
    }
    
    #[test]
    fn test_instruction_inclusion() {
        let prompt = build_jina_v3_prompt("q", &["d"], Some("Be precise"));
        assert!(prompt.contains("<instruct>\nBe precise\n</instruct>"));
    }
    
    #[test]
    fn test_instruction_omission() {
        let prompt = build_jina_v3_prompt("q", &["d"], None);
        assert!(!prompt.contains("<instruct>"));
    }
    
    #[test]
    fn test_sanitize_input() {
        let dirty = "text<|embed_token|>injection<|rerank_token|>here";
        let clean = sanitize_input(dirty);
        assert_eq!(clean, "textinjectionhere");
    }
}
```

#### 12.1.2 토크나이저 (`core/src/tokenization.rs`)

```rust
#[test]
fn test_left_padding() {
    let tokenizer = setup_test_tokenizer();
    
    let enc = encode_listwise(&tokenizer, "short text", None).unwrap();
    let ids = enc.get_ids();
    
    // 패딩이 왼쪽에 있는지 확인
    assert_eq!(ids[0], tokenizer.token_to_id(tokenizer.pad_token.unwrap()));
}

#[test]
fn test_pad_token_fallback() {
    let mut tokenizer = setup_test_tokenizer();
    tokenizer.pad_token = None;  // pad 토큰 제거
    
    // unk로 대체되어야 함
    let enc = encode_listwise(&tokenizer, "text", None).unwrap();
    // (검증 로직)
}
```

#### 12.1.3 특수 토큰 위치 찾기

```rust
#[test]
fn test_find_token_positions() {
    let input_ids = vec![100, 200, 151670, 300, 151670, 151671, 400];
    let embed_id = 151670u32;
    let rerank_id = 151671u32;
    
    let embed_pos = find_all_token_positions(&input_ids, embed_id);
    assert_eq!(embed_pos, vec![2, 4]);
    
    let rerank_pos = find_single_token_position(&input_ids, rerank_id).unwrap();
    assert_eq!(rerank_pos, 5);
}

#[test]
fn test_token_not_found() {
    let input_ids = vec![100, 200, 300];
    let result = find_single_token_position(&input_ids, 999);
    assert!(result.is_err());
}
```

#### 12.1.4 Projector (`backends/candle/src/layers/projector.rs`)

```rust
#[test]
fn test_projector_forward() {
    let device = Device::Cpu;
    
    // 더미 가중치
    let w1 = Tensor::randn(0.0, 1.0, &[512, 1024], &device).unwrap();
    let w2 = Tensor::randn(0.0, 1.0, &[512, 512], &device).unwrap();
    
    let projector = Projector::new(
        Linear::new(w1, None),
        Linear::new(w2, None),
    );
    
    // 입력
    let h = Tensor::randn(0.0, 1.0, &[1, 1024], &device).unwrap();
    
    // Forward
    let z = projector.forward(&h).unwrap();
    
    assert_eq!(z.dims(), &[1, 512]);
}

#[test]
fn test_projector_no_bias() {
    // bias 있는 가중치로 로드 시도 시 실패해야 함
    // (구현 생략)
}
```

#### 12.1.5 블록 가중치 계산

```rust
#[test]
fn test_block_weight_calculation() {
    let scores = vec![0.5, 0.8, 0.3];
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let weight = (1.0 + max_score) / 2.0;
    
    assert_eq!(max_score, 0.8);
    assert!((weight - 0.9).abs() < 1e-6);
}

#[test]
fn test_combine_query_embeddings() {
    let device = Device::Cpu;
    
    let q1 = Tensor::ones(&[1, 512], DType::F32, &device).unwrap();
    let q2 = Tensor::full(2.0, &[1, 512], &device).unwrap();
    
    let query_embeds = vec![q1, q2];
    let weights = vec![0.5, 1.5];  // sum = 2.0
    
    let combined = combine_query_embeddings(query_embeds, weights).unwrap();
    
    // 기대값: (0.5*1 + 1.5*2) / 2.0 = 1.75
    let expected = combined.get(&[0, 0]).unwrap().to_scalar::<f32>().unwrap();
    assert!((expected - 1.75).abs() < 1e-6);
}
```

### 12.2 통합 테스트

#### 12.2.1 기본 재랭킹 (`integration_tests/tests/rerank_listwise.rs`)

```rust
#[tokio::test]
async fn test_listwise_rerank_basic() {
    let client = setup_test_client("jinaai/jina-reranker-v3").await;
    
    let req = RerankRequest {
        query: "What is machine learning?".to_string(),
        texts: vec![
            "Machine learning is a subset of AI...".to_string(),
            "Cooking pasta requires boiling water...".to_string(),
            "Neural networks are inspired by the brain...".to_string(),
        ],
    };
    
    let resp = client.post("/rerank")
        .json(&req)
        .send()
        .await
        .unwrap();
    
    assert_eq!(resp.status(), 200);
    
    let body: RerankResponse = resp.json().await.unwrap();
    
    // 결과 개수 확인
    assert_eq!(body.results.len(), 3);
    
    // 점수 순 정렬 확인
    for i in 1..body.results.len() {
        assert!(body.results[i-1].score >= body.results[i].score);
    }
    
    // 인덱스 유효성 확인
    for result in &body.results {
        assert!(result.index < 3);
    }
    
    // 관련성 확인 (ML 관련 문서가 상위)
    assert!(body.results[0].index == 0 || body.results[0].index == 2);
}
```

#### 12.2.2 청킹 동작 검증

```rust
#[tokio::test]
async fn test_listwise_chunking() {
    let client = setup_test_client_with_flags(
        "jinaai/jina-reranker-v3",
        &["--max-listwise-docs-per-pass", "4"],
    ).await;
    
    // 10개 문서 (4+4+2 블록으로 나뉨)
    let docs: Vec<String> = (0..10)
        .map(|i| format!("Document number {}", i))
        .collect();
    
    let req = RerankRequest {
        query: "test".to_string(),
        texts: docs,
    };
    
    let resp = client.post("/rerank").json(&req).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    
    let body: RerankResponse = resp.json().await.unwrap();
    assert_eq!(body.results.len(), 10);
    
    // 메트릭 확인 (3개 블록 처리되었는지)
    let metrics = client.get("/metrics").send().await.unwrap().text().await.unwrap();
    assert!(metrics.contains("tei_lbnl_groups_per_req"));
}
```

#### 12.2.3 순서 제어

```rust
#[tokio::test]
async fn test_ordering_input() {
    let client = setup_test_client_with_flags(
        "jinaai/jina-reranker-v3",
        &["--rerank-ordering", "input"],
    ).await;
    
    let docs = vec!["A", "B", "C"];
    let req = RerankRequest {
        query: "test".to_string(),
        texts: docs.iter().map(|s| s.to_string()).collect(),
    };
    
    // 같은 요청을 여러 번 보내서 일관성 확인
    let resp1 = client.post("/rerank").json(&req).send().await.unwrap();
    let resp2 = client.post("/rerank").json(&req).send().await.unwrap();
    
    let body1: RerankResponse = resp1.json().await.unwrap();
    let body2: RerankResponse = resp2.json().await.unwrap();
    
    // 순서가 동일해야 함
    assert_eq!(body1.results, body2.results);
}

#[tokio::test]
async fn test_ordering_random() {
    let client = setup_test_client_with_flags(
        "jinaai/jina-reranker-v3",
        &["--rerank-ordering", "random"],
    ).await;
    
    // (랜덤 모드는 통계적 검증 필요, 생략)
}
```

#### 12.2.4 에러 처리

```rust
#[tokio::test]
async fn test_payload_too_large() {
    let client = setup_test_client_with_flags(
        "jinaai/jina-reranker-v3",
        &["--listwise-payload-limit-bytes", "1000"],
    ).await;
    
    let huge_doc = "x".repeat(2000);
    let req = RerankRequest {
        query: "test".to_string(),
        texts: vec![huge_doc],
    };
    
    let resp = client.post("/rerank").json(&req).send().await.unwrap();
    assert_eq!(resp.status(), 413);
}

#[tokio::test]
async fn test_missing_special_tokens() {
    // 특수 토큰 없는 모델로 listwise 강제 시도
    let result = setup_test_client_with_flags(
        "some/regular-model",
        &["--reranker-mode", "listwise"],
    ).await;
    
    // 로드 실패해야 함
    assert!(result.is_err());
}
```

### 12.3 성능 회귀 테스트

#### 12.3.1 Latency 벤치마크

```rust
#[bench]
fn bench_listwise_small(b: &mut Bencher) {
    let client = setup_bench_client();
    let req = create_bench_request(query_len=50, num_docs=8, doc_len=200);
    
    b.iter(|| {
        client.post("/rerank").json(&req).send().wait()
    });
}

#[bench]
fn bench_listwise_medium(b: &mut Bencher) {
    let client = setup_bench_client();
    let req = create_bench_request(query_len=100, num_docs=32, doc_len=500);
    
    b.iter(|| {
        client.post("/rerank").json(&req).send().wait()
    });
}

#[bench]
fn bench_listwise_large(b: &mut Bencher) {
    let client = setup_bench_client();
    let req = create_bench_request(query_len=200, num_docs=125, doc_len=1000);
    
    b.iter(|| {
        client.post("/rerank").json(&req).send().wait()
    });
}
```

#### 12.3.2 메모리 프로파일링

```bash
# VRAM 사용량 측정
pytest tests/performance/test_memory.py --gpu-monitor

# Expected:
# - Small (8 docs): < 2GB VRAM
# - Medium (32 docs): < 4GB VRAM
# - Large (125 docs): < 8GB VRAM
```

### 12.4 호환성 테스트

#### 12.4.1 기존 Pairwise 동작 유지

```rust
#[tokio::test]
async fn test_pairwise_unchanged() {
    let client = setup_test_client("cross-encoder/ms-marco-MiniLM-L-6-v2").await;
    
    let req = RerankRequest {
        query: "test".to_string(),
        texts: vec!["doc1".to_string(), "doc2".to_string()],
    };
    
    let resp = client.post("/rerank").json(&req).send().await.unwrap();
    let body: RerankResponse = resp.json().await.unwrap();
    
    // 기존 방식대로 동작
    assert_eq!(body.results.len(), 2);
}
```

---

## 13. 문서화

### 13.1 README 업데이트

`README.md`의 "Using Re-rankers models" 섹션에 추가:

````markdown
#### Listwise (LBNL) Reranker

TEI는 **Listwise reranking** 방식을 지원합니다. 이 방식은 여러 문서를 단일 컨텍스트에서 함께 인코딩하여, 문서 간 상호작용을 통한 더 정확한 재랭킹을 제공합니다.

**지원 모델**:
- `jinaai/jina-reranker-v3` (Qwen3 기반, 최대 8192 토큰)

**특징**:
- 문서 간 early interaction을 활용한 높은 정확도
- 긴 컨텍스트 지원 (최대 길이는 모델/토크나이저의 `model_max_length`에 따름)
- 자동 블록 청킹으로 대량 문서 처리

**사용 예시**:

```bash
model=jinaai/jina-reranker-v3
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id $model \
  --reranker-mode auto
```

```bash
curl 127.0.0.1:8080/rerank \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is Deep Learning?",
    "texts": [
      "Deep learning is a subset of machine learning...",
      "Neural networks consist of layers of nodes...",
      "Backpropagation is the algorithm used to train..."
    ]
  }'
```

**고급 옵션**:

```bash
--reranker-mode listwise                  # 강제로 listwise 모드 사용
--max-listwise-docs-per-pass 100          # 블록당 최대 문서 수
--rerank-ordering input                   # 문서 순서 유지 (기본값)
--listwise-payload-limit-bytes 5000000    # 페이로드 제한 (5MB)
```

**주의사항**:
- `--rerank-ordering random`은 입력 순서를 셔플하므로 **요청마다 점수/순위가 달라지는 것이 정상**입니다 (프로덕션 비권장).
- Listwise 방식은 문서 리스트 구성과 순서에 영향을 받을 수 있습니다
- 매우 많은 문서(100개 이상)의 경우 pairwise 방식이 더 빠를 수 있습니다
- 권장 설정: 쿼리 512토큰 이하, 문서 각각 2048토큰 이하
````

### 13.2 API 문서 업데이트

OpenAPI 스키마 (`openapi.yaml` 또는 코드 내 utoipa 어노테이션):

```yaml
paths:
  /rerank:
    post:
      summary: Rerank documents
      description: |
        재랭킹 API는 쿼리와 문서 리스트를 받아 관련성 점수를 반환합니다.
        
        **처리 방식**:
        - 모델이 listwise를 지원하면 자동으로 listwise 모드 사용
        - 그렇지 않으면 기존 pairwise 방식 사용
        - CLI 플래그로 강제 지정 가능
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - query
                - texts
              properties:
                query:
                  type: string
                  description: 검색 쿼리
                  example: "What is machine learning?"
                texts:
                  type: array
                  items:
                    type: string
                  description: 재랭킹할 문서 리스트
                  example: ["Document 1", "Document 2"]
                  minItems: 1
                  maxItems: 500
      responses:
        '200':
          description: 성공
          content:
            application/json:
              schema:
                type: object
                properties:
                  results:
                    type: array
                    items:
                      type: object
                      properties:
                        index:
                          type: integer
                          description: 원본 texts 배열의 인덱스
                        score:
                          type: number
                          format: float
                          description: 관련성 점수 (높을수록 관련성 높음)
        '413':
          description: 페이로드 크기 초과
        '422':
          description: 검증 실패 (예: 특수 토큰 미발견, `<|rerank_token|>`이 1회가 아닌 경우 등)
          content:
            application/json:
              example:
                error: "invalid_rerank_prompt: rerank_token not found or duplicated"
                error_type: "invalid_input"
```

### 13.3 CLI 도움말

`--help` 출력에 추가 설명:

```
RERANKING OPTIONS:
    --reranker-mode <MODE>
        Reranking strategy [default: auto] [possible values: auto, pairwise, listwise]
        
        auto      : Automatically detect based on model metadata
        pairwise  : Force pairwise (document-by-document) reranking
        listwise  : Force listwise (batch) reranking (requires compatible model)
    
    --max-listwise-docs-per-pass <INTEGER>
        Maximum documents per listwise forward pass [default: 125]
        
        Controls how many documents are processed together in one batch.
        Higher values may improve accuracy but increase memory usage.
    
    --rerank-ordering <MODE>
        Document ordering [default: input] [possible values: input, random]
        
        input   : Maintain original order (deterministic, recommended)
        random  : Shuffle before processing (for testing only)
    
    --listwise-payload-limit-bytes <INTEGER>
        Maximum request payload size for listwise reranking [default: 2000000]
        
        Increase this if you need to process very long documents.

EXAMPLES:
    # Auto-detect listwise support
    text-embeddings-inference --model-id jinaai/jina-reranker-v3
    
    # Force listwise with custom settings
    text-embeddings-inference \
        --model-id jinaai/jina-reranker-v3 \
        --reranker-mode listwise \
        --max-listwise-docs-per-pass 100
```

### 13.4 마이그레이션 가이드

새 문서 `docs/migration/listwise-reranker.md`:

````markdown
# Listwise Reranker 마이그레이션 가이드

## 개요

TEI 1.8부터 Listwise reranking을 지원합니다. 기존 pairwise 방식과 병존하며, 하위 호환성을 유지합니다.

## 변경 사항

### 1. 자동 감지

모델이 listwise를 지원하면 자동으로 활성화됩니다 (`--reranker-mode auto` 기본값).

### 2. API 변경 없음

HTTP `/rerank` 엔드포인트는 동일합니다. 요청/응답 형식 변경 없음.

### 3. 새 CLI 플래그

선택적으로 다음 플래그를 사용할 수 있습니다:

- `--max-listwise-docs-per-pass`: 블록 크기 조정
- `--rerank-ordering`: 순서 제어 (프로덕션에서는 `input` 권장)

## 마이그레이션 체크리스트

1. **모델 호환성 확인**
   ```bash
   # 현재 모델이 listwise를 지원하는지 확인
   curl http://localhost:8080/info
   ```

2. **페이로드 크기 검토**
   - 긴 문서를 많이 보내는 경우 `--listwise-payload-limit-bytes` 증가 고려

3. **성능 테스트**
   - 기존 워크로드로 latency/throughput 측정
   - 메트릭 `/metrics` 엔드포인트 확인

4. **에러 핸들링 검토**
   - 413 (페이로드 초과) 처리 추가
   - 422 (모델 불일치) 처리 추가

## 롤백

문제 발생 시 pairwise로 되돌리기:

```bash
--reranker-mode pairwise
```

또는 이전 버전으로:

```bash
docker pull ghcr.io/huggingface/text-embeddings-inference:1.7
```
````

---

## 14. 보안 및 에러 처리

### 14.1 입력 검증

#### 14.1.1 길이 제한

```rust
fn validate_rerank_request(req: &RerankRequest, config: &Config) -> Result<()> {
    // 쿼리 길이
    if req.query.len() > config.max_query_bytes {
        return Err(ApiError::InvalidInput(
            format!("Query exceeds maximum length: {} > {}", 
                    req.query.len(), config.max_query_bytes)
        ));
    }
    
    // 문서 개수
    if req.texts.is_empty() {
        return Err(ApiError::InvalidInput("No documents provided".to_string()));
    }
    
    if req.texts.len() > config.max_documents {
        return Err(ApiError::InvalidInput(
            format!("Too many documents: {} > {}", 
                    req.texts.len(), config.max_documents)
        ));
    }
    
    // 개별 문서 길이
    for (i, doc) in req.texts.iter().enumerate() {
        if doc.len() > config.max_document_bytes {
            return Err(ApiError::InvalidInput(
                format!("Document {} exceeds maximum length: {} > {}", 
                        i, doc.len(), config.max_document_bytes)
            ));
        }
    }
    
    Ok(())
}
```

#### 14.1.2 페이로드 크기

```rust
// Axum 미들웨어
async fn check_payload_size(
    req: Request<Body>,
    next: Next<Body>,
) -> Result<Response, StatusCode> {
    let content_length = req.headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    
    if content_length > state.listwise_payload_limit_bytes {
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }
    
    Ok(next.run(req).await)
}
```

### 14.2 특수 토큰 인젝션 방지

```rust
fn sanitize_input(text: &str) -> String {
    const SPECIAL_TOKENS: &[&str] = &[
        "<|embed_token|>",
        "<|rerank_token|>",
    ];
    
    let mut result = text.to_string();
    for token in SPECIAL_TOKENS {
        result = result.replace(token, "");
    }
    result
}
```

> **Note**: 원본 `modeling.py`와 동일하게 두 특수 토큰(`<|embed_token|>`, `<|rerank_token|>`)만 제거합니다. `<query>`, `<passage>` 등 일반 태그 문자열은 사용자 입력으로 남겨 둡니다.

**적용 위치**:
- 프롬프트 구성 전 모든 입력 (query, docs)
- 에러 메시지 내 사용자 입력 출력 시

### 14.3 토큰 오버플로우 방지

```rust
fn estimate_prompt_tokens(
    query: &str,
    docs: &[&str],
    tokenizer: &Tokenizer,
) -> Result<usize> {
    let query_tokens = count_tokens(tokenizer, query);
    let doc_tokens: usize = docs.iter()
        .map(|d| count_tokens(tokenizer, d))
        .sum();
    
    const OVERHEAD: usize = 200;  // 템플릿 오버헤드
    
    let total = query_tokens * 2 + doc_tokens + OVERHEAD;
    
    Ok(total)
}

fn check_token_budget(
    estimated: usize,
    max_seq_len: usize,
) -> Result<()> {
    let budget = (max_seq_len as f32 * 0.9) as usize;
    
    if estimated > budget {
        return Err(ApiError::TokenLimitExceeded(
            format!("Estimated {} tokens exceeds budget {}", estimated, budget)
        ));
    }

    Ok(())
}
```

> **Note**: 이 가드는 요청을 선제적으로 거부하기 위한 용도이며, 실제 청킹 루프는 `modeling.py`와 동일한 휴리스틱(`block_size=125`, `length_capacity = max_seq_len - 2 * query_len`, `length_capacity <= max_doc_len`)으로 동작합니다.

### 14.4 에러 응답 형식

```rust
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    error_type: String,
    details: Option<serde_json::Value>,
}

impl From<ApiError> for ErrorResponse {
    fn from(err: ApiError) -> Self {
        match err {
            ApiError::InvalidInput(msg) => ErrorResponse {
                error: msg,
                error_type: "invalid_input".to_string(),
                details: None,
            },
            ApiError::TokenLimitExceeded(msg) => ErrorResponse {
                error: msg,
                error_type: "token_limit_exceeded".to_string(),
                details: Some(json!({
                    "suggestion": "Try reducing document count or length"
                })),
            },
            ApiError::PayloadTooLarge => ErrorResponse {
                error: "Request payload exceeds maximum size".to_string(),
                error_type: "payload_too_large".to_string(),
                details: Some(json!({
                    "suggestion": "Increase --listwise-payload-limit-bytes"
                })),
            },
            ApiError::ModelNotSupported(msg) => ErrorResponse {
                error: msg,
                error_type: "model_not_supported".to_string(),
                details: Some(json!({
                    "suggestion": "Use --reranker-mode auto or pairwise"
                })),
            },
            // ...
        }
    }
}
```

### 14.5 모델 로드 실패 처리

```rust
async fn load_listwise_model(
    model_path: &Path,
    cli_flags: &CliFlags,
) -> Result<LbnlRerankerCandle> {
    // 토크나이저 체크
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|e| LoadError::TokenizerNotFound(e.to_string()))?;
    
    if !tokenizer.is_fast() {
        return Err(LoadError::FastTokenizerRequired);
    }
    
    // 특수 토큰 체크
    let embed_token_id = tokenizer.token_to_id("<|embed_token|>")
        .ok_or(LoadError::SpecialTokenMissing("<|embed_token|>"))?;
    
    let rerank_token_id = tokenizer.token_to_id("<|rerank_token|>")
        .ok_or(LoadError::SpecialTokenMissing("<|rerank_token|>"))?;
    
    // Projector 가중치 체크
    if !repo.has_weight("projector.0.weight") || 
       !repo.has_weight("projector.2.weight") {
        return Err(LoadError::ProjectorWeightsNotFound);
    }
    
    if repo.has_weight("projector.0.bias") || 
       repo.has_weight("projector.2.bias") {
        return Err(LoadError::InvalidProjectorStructure(
            "Projector should not have bias parameters"
        ));
    }
    
    // 모델 로딩
    let model = Qwen3CausalLm::load(model_path, &device)?;
    let projector = Projector::load(&repo, &device)?;
    
    Ok(LbnlRerankerCandle {
        model,
        tokenizer: Arc::new(tokenizer),
        projector,
        // ...
    })
}
```

**에러 타입**:

```rust
#[derive(Debug, thiserror::Error)]
enum LoadError {
    #[error("Fast tokenizer is required for TEI")]
    FastTokenizerRequired,
    
    #[error("Special token not found: {0}")]
    SpecialTokenMissing(&'static str),
    
    #[error("Projector weights not found (projector.0.weight, projector.2.weight required)")]
    ProjectorWeightsNotFound,
    
    #[error("Invalid projector structure: {0}")]
    InvalidProjectorStructure(&'static str),
    
    #[error("Tokenizer not found: {0}")]
    TokenizerNotFound(String),
}
```
