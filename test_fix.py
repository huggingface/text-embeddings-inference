#!/usr/bin/env python3
"""
Quick test to verify the Qwen3 reranking fix is working.
"""

import requests
import json

def test_rerank():
    url = "http://localhost:8080/rerank"
    
    # Test with the Korean query from the original problem
    data = {
        "query": "털사 대학교에서 2003년부터 2006년까지 감독을 맡았던 사람이 누구야?",
        "texts": [
            "스티븐 존 크래고퍼는 2003년부터 2006년까지 툴사에서 헤드 코치로 역임했습니다.",  # Should score high
            "존 맥널티는 NFL 코치입니다.",  # Should score low
            "2005년 툴사 팀의 감독은 스티브 크래고였습니다."  # Should score high
        ],
        "truncate": False
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("Testing Qwen3 reranking fix...")
    print(f"Query: {data['query']}")
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        results = response.json()
        scores = [item['score'] for item in results]
        
        print("\nResults:")
        for i, (text, score) in enumerate(zip(data['texts'], scores)):
            print(f"{i}: {score:.6f} - {text[:60]}...")
        
        # Check if scores are different
        unique_scores = set(scores)
        if len(unique_scores) == 1:
            print(f"\n❌ BUG STILL PRESENT: All scores are {scores[0]}")
            return False
        else:
            print(f"\n✅ BUG FIXED: Scores vary from {min(scores):.6f} to {max(scores):.6f}")
            
            # Check if relevant texts score higher
            if scores[0] > scores[1] and scores[2] > scores[1]:
                print("✅ CORRECT: Relevant texts score higher!")
                return True
            else:
                print("⚠️  WARNING: Scoring order may be incorrect")
                return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the service is running with the Qwen3 reranker model")
        return False

if __name__ == "__main__":
    test_rerank()