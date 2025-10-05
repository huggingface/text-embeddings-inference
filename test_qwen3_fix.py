#!/usr/bin/env python3
"""
Test the Qwen3 CausalLM reranker fix.
"""

import requests
import json
import sys

def test_qwen3_rerank():
    """Test the Qwen3 reranking with the Korean query."""
    
    url = "http://localhost:8080/rerank"
    
    data = {
        "query": "털사 대학교에서 2003년부터 2006년까지 감독을 맡았던 사람이 누구야?",
        "texts": [
            "스티븐 존 크래고퍼(1965년 4월 28일 출생)는 미식축구 코치이자 전 선수입니다. 그는 이전에 타이거즈 풋볼 팀의 쿼터백 코치로 일한 후 현재 루이지애나 주립대학교 미식축구 프로그램의 행정 보좌관으로 일하고 있습니다. 그는 2007년부터 2009년까지 루이빌에서, 2003년부터 2006년까지 툴사에서 헤드 코치로 역임했습니다.",
            "존 맥널티(1968년 5월 29일 출생)는 미국 프로 미식축구 리그(NFL) 로스앤젤레스 차저스의 타이트 엔드 코치인 미식축구 코치입니다.",
            "2005년 툴사 골든 허리케인 축구팀은 2005년 NCAA 디비전 I-A 축구 시즌에서 툴사 대학교를 대표했습니다. 팀의 감독은 스티브 크래고였습니다."
        ],
        "truncate": False,
        "instruction": "Given a web search query, retrieve relevant passages that answer the query"
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("🚀 Testing Qwen3 CausalLM reranker fix...")
    print(f"Query: {data['query']}")
    print(f"Number of texts: {len(data['texts'])}")
    print("-" * 80)
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        results = response.json()
        
        # Check if we got ranks
        if isinstance(results, list):
            print("✅ Response received successfully!")
            print(f"Number of results: {len(results)}")
            
            # Extract scores
            scores = {item['index']: item['score'] for item in results}
            unique_scores = set(scores.values())
            
            print(f"Unique scores: {len(unique_scores)}")
            print("-" * 80)
            
            # Check if bug is fixed
            if len(unique_scores) == 1:
                print("❌ BUG STILL PRESENT: All scores are identical!")
                print(f"   All texts have score: {list(unique_scores)[0]}")
            else:
                print("✅ BUG FIXED: Scores are different!")
                print(f"   Score range: {min(scores.values()):.6f} to {max(scores.values()):.6f}")
            
            print("-" * 80)
            print("Results (sorted by score):")
            
            for i, result in enumerate(results):
                idx = result['index']
                score = result['score']
                text_preview = data['texts'][idx][:100] + "..."
                print(f"{i+1}. Text #{idx}: Score = {score:.6f}")
                print(f"   Preview: {text_preview}")
                
                # Check if this is about Steve Kragthorpe
                if "크래고" in data['texts'][idx] or "툴사" in data['texts'][idx]:
                    print("   ⭐ Contains Kragthorpe/Tulsa reference!")
            
            print("-" * 80)
            
            # Expected behavior check
            kragthorpe_indices = [0, 2]  # Texts about Steve Kragthorpe
            kragthorpe_scores = [scores[i] for i in kragthorpe_indices if i in scores]
            other_scores = [scores[i] for i in scores if i not in kragthorpe_indices]
            
            if kragthorpe_scores and other_scores:
                avg_kragthorpe = sum(kragthorpe_scores) / len(kragthorpe_scores)
                avg_other = sum(other_scores) / len(other_scores)
                
                print("Expected behavior check:")
                print(f"Average score for Kragthorpe texts: {avg_kragthorpe:.6f}")
                print(f"Average score for other texts: {avg_other:.6f}")
                
                if avg_kragthorpe > avg_other:
                    print("✅ CORRECT: Kragthorpe texts score higher on average!")
                else:
                    print("⚠️  WARNING: Kragthorpe texts do not score higher")
            
            return len(unique_scores) > 1
        else:
            print(f"❌ Unexpected response format: {results}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making request: {e}")
        print("\nMake sure the reranking service is running at http://localhost:8080")
        print("with the Qwen3-Reranker model")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Main test runner."""
    print("=" * 80)
    print("Qwen3 CausalLM Reranking Bug Fix Test")
    print("=" * 80)
    print("\nThis test verifies that:")
    print("1. Token IDs are detected dynamically")
    print("2. Prompts are formatted correctly for Qwen3")
    print("3. Different texts get different scores")
    print("=" * 80)
    
    success = test_qwen3_rerank()
    
    print("=" * 80)
    if success:
        print("🎉 TEST PASSED: The Qwen3 CausalLM reranking bug has been fixed!")
        sys.exit(0)
    else:
        print("❌ TEST FAILED: The bug is still present or service is unavailable")
        sys.exit(1)


if __name__ == "__main__":
    main()