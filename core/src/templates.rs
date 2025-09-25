use std::fmt::Write;

/// Template formatter for models that require structured prompts
pub trait TemplateFormatter {
    /// Format a query-document pair for reranking
    fn format_rerank(
        &self,
        query: &str,
        document: &str,
        instruction: Option<&str>,
    ) -> String;
}

/// Qwen3 reranker template formatter
pub struct Qwen3RerankerTemplate {
    default_instruction: String,
}

impl Qwen3RerankerTemplate {
    pub fn new() -> Self {
        Self {
            default_instruction: "Select only the Documents that are semantically similar to the Query.".to_string(),
        }
    }
}

impl TemplateFormatter for Qwen3RerankerTemplate {
    fn format_rerank(
        &self,
        query: &str,
        document: &str,
        instruction: Option<&str>,
    ) -> String {
        let instruction = instruction.unwrap_or(&self.default_instruction);
        
        let mut result = String::with_capacity(512);
        
        // System prompt
        result.push_str("<|im_start|>system\n");
        result.push_str("Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n");
        
        // User prompt with instruction, query, and document
        result.push_str("<|im_start|>user\n");
        write!(&mut result, "<Instruct>: {}\n", instruction).unwrap();
        write!(&mut result, "<Query>: {}\n", query).unwrap();
        write!(&mut result, "<Document>: {}", document).unwrap();
        result.push_str("<|im_end|>\n");
        
        // Assistant prompt to trigger reasoning
        result.push_str("<|im_start|>assistant\n");
        result.push_str("<think>\n\n</think>\n\n");
        
        result
    }
}

/// Check if a model requires template formatting
pub fn requires_template(model_name: &str) -> bool {
    // Check if this is a Qwen3 sequence classification model
    model_name.contains("Qwen3") && model_name.contains("seq-cls")
}

/// Get the appropriate template formatter for a model
pub fn get_template_formatter(model_name: &str) -> Option<Box<dyn TemplateFormatter + Send + Sync>> {
    if requires_template(model_name) {
        Some(Box::new(Qwen3RerankerTemplate::new()))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_template() {
        let template = Qwen3RerankerTemplate::new();
        let formatted = template.format_rerank(
            "What is Deep Learning?",
            "Deep Learning is a branch of machine learning",
            None,
        );
        
        assert!(formatted.contains("<|im_start|>system"));
        assert!(formatted.contains("<Query>: What is Deep Learning?"));
        assert!(formatted.contains("<Document>: Deep Learning is a branch of machine learning"));
        assert!(formatted.contains("<think>"));
    }
    
    #[test]
    fn test_custom_instruction() {
        let template = Qwen3RerankerTemplate::new();
        let formatted = template.format_rerank(
            "test query",
            "test doc",
            Some("Custom instruction"),
        );
        
        assert!(formatted.contains("<Instruct>: Custom instruction"));
    }
    
    #[test]
    fn test_requires_template() {
        assert!(requires_template("tomaarsen/Qwen3-Reranker-0.6B-seq-cls"));
        assert!(requires_template("Qwen3-Something-seq-cls"));
        assert!(!requires_template("BAAI/bge-reranker"));
        assert!(!requires_template("Qwen3-Embed"));
    }
}