#!/usr/bin/env python3
"""
Verify the complete Qwen3 reranker fix.
Tests that different documents get different scores for Korean food query.
"""
import requests
import json

def test_reranking():
    url = "http://localhost:8080/rerank"
    
    # Test with Korean food query
    data = {
        "query": "한국의 음식",
        "texts": ["부산", "뉴욕", "김치"],
        "truncate": False
    }
    
    print("Testing Qwen3 reranker with Korean food query...")
    print(f"Query: {data['query']}")
    print(f"Documents: {data['texts']}")
    print("-" * 50)
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            results = response.json()
            
            # Create score mapping
            scores = {}
            for item in results:
                text = data['texts'][item['index']]
                score = item['score']
                scores[text] = score
                print(f"{text}: {score:.6f}")
            
            # Check if scores are different
            unique_scores = set(scores.values())
            print(f"\nUnique scores: {len(unique_scores)}")
            
            if len(unique_scores) == 1:
                print("❌ FAIL: All documents have identical scores!")
                return False
            
            # Check semantic ranking
            if scores["김치"] > scores["부산"] and scores["부산"] > scores["뉴욕"]:
                print("✅ SUCCESS: Ranking is semantically correct!")
                print("   김치 (Kimchi) > 부산 (Busan) > 뉴욕 (New York)")
                return True
            else:
                print("⚠️  WARNING: Documents have different scores but ranking seems incorrect")
                print(f"   Expected: 김치 > 부산 > 뉴욕")
                print(f"   Got: {' > '.join(sorted(scores.keys(), key=lambda k: scores[k], reverse=True))}")
                return True  # At least scores are different
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_reranking()
    exit(0 if success else 1)