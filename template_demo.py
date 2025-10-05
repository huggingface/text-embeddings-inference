#!/usr/bin/env python3
"""Demonstrate template formatting for Qwen3 reranker"""

def format_qwen3_rerank(query, document, instruction=None):
    """Format a query-document pair for Qwen3 reranking"""
    if instruction is None:
        instruction = "Select only the Documents that are semantically similar to the Query."
    
    template = f"""<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
"""
    return template

def main():
    # Korean food example
    query = "한국 음식"
    documents = ["부산", "뉴욕", "김치"]
    
    print("Template Formatting for Qwen3 Reranker")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Documents: {documents}")
    print("=" * 60)
    
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}: {doc}")
        print("-" * 60)
        formatted = format_qwen3_rerank(query, doc)
        print(formatted)
        print("-" * 60)
    
    # Expected scoring:
    # "김치" (Kimchi) - highest score (directly related to Korean food)
    # "부산" (Busan) - medium score (Korean city, indirectly related)
    # "뉴욕" (New York) - lowest score (not related to Korean food)
    
    print("\nExpected ranking:")
    print("1. 김치 (Kimchi) - Directly related to Korean food")
    print("2. 부산 (Busan) - Korean city, indirectly related")
    print("3. 뉴욕 (New York) - Not related to Korean food")
    
    # Custom instruction example
    print("\n" + "=" * 60)
    print("With custom instruction:")
    custom_instruction = "음식 이름인 문서만 선택하세요" # Select only documents that are food names
    formatted = format_qwen3_rerank(query, "김치", custom_instruction)
    print(formatted)

if __name__ == "__main__":
    main()