#!/usr/bin/env python3
"""Test template formatting for Qwen3 reranker"""

from text_embeddings_inference.core.templates import Qwen3RerankerTemplate, requires_template

def test_template_formatting():
    # Test model names that should require templates
    models_requiring_template = [
        "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        "Qwen3-Something-seq-cls",
        "qwen3-reranker-model",
        "QWEN3-Reranker-1B"
    ]
    
    models_not_requiring_template = [
        "BAAI/bge-reranker",
        "Qwen3-Embed",
        "bert-base-uncased"
    ]
    
    print("Testing requires_template function:")
    for model in models_requiring_template:
        result = requires_template(model)
        print(f"  {model}: {result} (expected: True)")
        assert result == True
    
    for model in models_not_requiring_template:
        result = requires_template(model)
        print(f"  {model}: {result} (expected: False)")
        assert result == False
    
    print("\nTesting template formatting:")
    template = Qwen3RerankerTemplate()
    
    # Test Korean example
    query = "한국 음식"
    documents = ["부산", "뉴욕", "김치"]
    
    for i, doc in enumerate(documents):
        formatted = template.format_rerank(query, doc, None)
        print(f"\nDocument {i+1} ({doc}):")
        print("-" * 40)
        print(formatted)
        print("-" * 40)
        
        # Verify template structure
        assert "<|im_start|>system" in formatted
        assert "<Query>: 한국 음식" in formatted
        assert f"<Document>: {doc}" in formatted
        assert "<|im_start|>assistant" in formatted
    
    # Test with custom instruction
    custom_instruction = "음식과 관련된 문서만 선택하세요"
    formatted = template.format_rerank("한국 음식", "김치", custom_instruction)
    print("\nWith custom instruction:")
    print("-" * 40)
    print(formatted)
    print("-" * 40)
    assert f"<Instruct>: {custom_instruction}" in formatted
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_template_formatting()