#!/usr/bin/env python3
"""Test script for Qwen3 reranker model"""

import requests
import json

def test_reranker(query, documents, expected_order=None):
    """Test the reranker with given query and documents"""
    url = "http://localhost:8080/rerank"
    payload = {
        "query": query,
        "texts": documents
    }
    
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"Documents: {documents}")
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        results = response.json()
        print(f"\nResults:")
        for item in results:
            idx = item['index']
            score = item['score']
            doc = documents[idx]
            print(f"  [{idx}] '{doc}' -> score: {score:.6f}")
        
        # Sort by score to show ranking
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        print(f"\nRanking (highest to lowest):")
        for i, item in enumerate(sorted_results):
            idx = item['index']
            score = item['score']
            doc = documents[idx]
            print(f"  {i+1}. '{doc}' (score: {score:.6f})")
            
        if expected_order:
            actual_order = [documents[item['index']] for item in sorted_results]
            if actual_order == expected_order:
                print(f"\n✓ Ranking matches expected order")
            else:
                print(f"\n✗ Ranking does NOT match expected order")
                print(f"  Expected: {expected_order}")
                print(f"  Actual:   {actual_order}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

# Test 1: Korean food
test_reranker(
    "한국 음식",
    ["김치", "뉴욕", "부산"],
    expected_order=["김치", "부산", "뉴욕"]
)

# Test 2: English - capital cities
test_reranker(
    "What is the capital of France?",
    ["Paris is the capital of France", "Tokyo is the capital of Japan", "Pizza is a food from Italy"],
    expected_order=["Paris is the capital of France", "Tokyo is the capital of Japan", "Pizza is a food from Italy"]
)

# Test 3: Simple similarity
test_reranker(
    "food",
    ["pizza", "car", "food", "hamburger", "bicycle"],
    expected_order=["food", "pizza", "hamburger", "car", "bicycle"]
)

# Test 4: Technical query
test_reranker(
    "machine learning",
    ["neural networks", "cooking recipes", "deep learning", "car engines", "artificial intelligence"],
    expected_order=["deep learning", "neural networks", "artificial intelligence", "car engines", "cooking recipes"]
)

# Test 5: Mixed languages
test_reranker(
    "Python programming",
    ["파이썬 프로그래밍", "Python is a programming language", "자바스크립트", "How to cook pasta"],
    expected_order=["Python is a programming language", "파이썬 프로그래밍", "자바스크립트", "How to cook pasta"]
)

# Test 6: Identical documents
test_reranker(
    "test",
    ["test", "test", "different"],
    expected_order=["test", "test", "different"]
)

print(f"\n{'='*60}")
print("Test completed!")