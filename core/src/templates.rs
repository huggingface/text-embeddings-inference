/// Chat template system for Qwen3-Reranker models
///
/// Qwen3-Reranker uses a ChatML-style template for reranking:
/// ```
/// <|im_start|>system
/// Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
/// <|im_start|>user
/// <Instruct>: {instruction}
/// <Query>: {query}
/// <Document>: {document}<|im_end|>
/// <|im_start|>assistant
/// ```

/// Default instruction for Qwen3-Reranker
pub const DEFAULT_RERANKER_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// System prompt for Qwen3-Reranker
const SYSTEM_PROMPT: &str = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".";

/// Apply Qwen3-Reranker template to format query and document
///
/// # Arguments
/// * `query` - The search query
/// * `document` - The document to evaluate
/// * `instruction` - Optional custom instruction (uses default if None)
///
/// # Returns
/// Formatted template string ready for tokenization
pub fn apply_qwen3_reranker_template(
    query: &str,
    document: &str,
    instruction: Option<&str>,
) -> String {
    let instruction = instruction.unwrap_or(DEFAULT_RERANKER_INSTRUCTION);

    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n<Instruct>: {}\n<Query>: {}\n<Document>: {}<|im_end|>\n<|im_start|>assistant\n",
        SYSTEM_PROMPT,
        instruction,
        query,
        document
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_qwen3_reranker_template_default() {
        let query = "What is deep learning?";
        let document = "Deep learning is a subset of machine learning.";

        let result = apply_qwen3_reranker_template(query, document, None);

        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains(SYSTEM_PROMPT));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains(DEFAULT_RERANKER_INSTRUCTION));
        assert!(result.contains(query));
        assert!(result.contains(document));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_apply_qwen3_reranker_template_custom_instruction() {
        let query = "Find code examples";
        let document = "Here is a Python example.";
        let instruction = "Find code snippets that demonstrate the query";

        let result = apply_qwen3_reranker_template(query, document, Some(instruction));

        assert!(result.contains(instruction));
        assert!(!result.contains(DEFAULT_RERANKER_INSTRUCTION));
    }
}
