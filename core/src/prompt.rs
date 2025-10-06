//! Jina v3 listwise reranker를 위한 프롬프트 빌딩
//!
//! 이 모듈은 Python 참조 구현의 정확한 템플릿을 따르는
//! 프롬프트 구성을 제공합니다.

/// 프롬프트 주입을 일으킬 수 있는 특수 토큰 제거하여 입력 텍스트 sanitize
///
/// hidden state 추출을 방해할 수 있는 두 개의 임베딩 관련 토큰만 제거합니다.
/// 채팅 형식 토큰(<|im_start|>, <|im_end|>)은 정상적인 사용자 콘텐츠의
/// 일부일 수 있으므로 그대로 유지합니다.
pub fn sanitize_input(text: &str) -> String {
    text.replace("<|embed_token|>", "")
        .replace("<|rerank_token|>", "")
}

/// Python 참조 템플릿을 정확히 따르는 Jina v3 LBNL 프롬프트 빌드
///
/// 템플릿 구조:
/// 1. System 메시지 (역할 정의)
/// 2. User 메시지:
///    - 문서 개수가 포함된 작업 설명
///    - 선택적 instruction 블록
///    - <|embed_token|> 마커가 있는 Passages
///    - <|rerank_token|> 마커가 있는 Query 블록
/// 3. Thinking placeholder가 있는 Assistant 메시지
///
/// # 인자
/// * `query` - 검색 쿼리 문자열 (sanitize됨)
/// * `docs` - 순위를 매길 문서 문자열들 (sanitize됨)
/// * `instruction` - 선택적 추가 instruction
///
/// # 반환
/// 토크나이제이션 준비가 된 완전한 프롬프트 문자열
pub fn build_jina_v3_prompt(query: &str, docs: &[&str], instruction: Option<&str>) -> String {
    // 모든 입력 sanitize
    let query_clean = sanitize_input(query);
    let docs_clean: Vec<String> = docs.iter().map(|d| sanitize_input(d)).collect();
    let k = docs.len();

    let mut prompt = String::with_capacity(
        1024 + query_clean.len() * 2 + docs_clean.iter().map(|d| d.len()).sum::<usize>(),
    );

    // System 메시지 (TECHSPEC §7.1.1 및 modeling.py와 정확히 일치)
    // 경고: 이 문자열을 수정하면 모델 호환성이 깨집니다 - 재포맷하지 마세요
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str("You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. If an instruction is provided, you should follow the instruction when determining the ranking.\n");
    prompt.push_str("<|im_end|>\n");

    // User 메시지 헤더
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(&format!(
        "I will provide you with {} passages, each indicated by a numerical identifier. \
         Rank the passages based on their relevance to query: {}\n",
        k, query_clean
    ));

    // 선택적 instruction 블록
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

    // Query 블록 (샌드위치 패턴 - 쿼리가 두 번 나타남)
    prompt.push_str("<query>\n");
    prompt.push_str(&query_clean);
    prompt.push_str("<|rerank_token|>\n</query>\n");

    // Thinking placeholder가 있는 Assistant 메시지
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>assistant\n");
    prompt.push_str("<think>\n\n</think>\n\n");

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_removes_special_tokens() {
        let input = "Hello <|embed_token|> world <|rerank_token|> test";
        let result = sanitize_input(input);
        assert_eq!(result, "Hello  world  test");
    }

    #[test]
    fn test_build_prompt_structure() {
        let query = "What is Rust?";
        let docs = vec!["Rust is a systems programming language.", "Python is easy."];
        let prompt = build_jina_v3_prompt(query, &docs, None);

        // 주요 컴포넌트 확인
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("You are a search relevance expert"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("I will provide you with 2 passages"));
        assert!(prompt.contains("<passage id=\"0\">"));
        assert!(prompt.contains("<passage id=\"1\">"));
        assert!(prompt.contains("<|embed_token|>"));
        assert!(prompt.contains("<|rerank_token|>"));
        assert!(prompt.contains("<query>"));
        assert!(prompt.contains("<|im_start|>assistant"));
        assert!(prompt.contains("<think>"));
    }

    #[test]
    fn test_build_prompt_with_instruction() {
        let query = "test query";
        let docs = vec!["doc1"];
        let prompt = build_jina_v3_prompt(query, &docs, Some("Focus on technical accuracy."));

        assert!(prompt.contains("<instruct>"));
        assert!(prompt.contains("Focus on technical accuracy."));
        assert!(prompt.contains("</instruct>"));
    }
}
