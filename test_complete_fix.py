#!/usr/bin/env python3
"""
Test the complete Qwen3 reranker fix with both tensor shape and template formatting.
"""

import requests
import json
import sys

def test_korean_food_rerank():
    """Test the Qwen3 reranking with Korean food query."""
    
    url = "http://localhost:8080/rerank"
    
    data = {
        "query": "한국의 음식",
        "texts": [
            "부산",
            "뉴욕", 
            "김치"
        ],
        "truncate": False
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("🚀 Testing Qwen3 reranker with Korean food query...")
    print(f"Query: {data['query']} (Korean food)")
    print(f"Texts: {data['texts']}")
    print("-" * 80)
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        results = response.json()
        
        if isinstance(results, list) and len(results) == 3:
            print("✅ Response received successfully!")
            print(f"Number of results: {len(results)}")
            print("-" * 80)
            
            # Sort by score to see ranking
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            print("Results (sorted by score):")
            for i, result in enumerate(sorted_results):
                idx = result['index']
                score = result['score']
                text = data['texts'][idx]
                print(f"{i+1}. '{text}': Score = {score:.6f}")
                
                # Check if this is kimchi
                if text == "김치":
                    kimchi_rank = i + 1
                    kimchi_score = score
            
            print("-" * 80)
            
            # Check if kimchi is ranked first
            if sorted_results[0]['index'] == 2:  # Index 2 is 김치
                print("✅ SUCCESS: 김치 (Kimchi) is ranked first!")
                print("   This confirms both fixes are working:")
                print("   1. Tensor shape issue is resolved")
                print("   2. Template formatting is applied correctly")
                return True
            else:
                print("❌ ISSUE: 김치 (Kimchi) is not ranked first")
                print(f"   김치 is ranked #{kimchi_rank} with score {kimchi_score:.6f}")
                print("   Expected: 김치 should have the highest score for '한국의 음식' query")
                return False
                
        else:
            print(f"❌ Unexpected response format: {results}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making request: {e}")
        print("\nMake sure the reranking service is running at http://localhost:8080")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_larger_rerank():
    """Test with the original larger dataset."""
    
    url = "http://localhost:8080/rerank"
    
    data = {
        "query": "털사 대학교에서 2003년부터 2006년까지 감독을 맡았던 사람이 누구야?",
        "texts": [
            "존 맥널티(1968년 5월 29일 출생)는 미국 프로 미식축구 리그(NFL) 로스앤젤레스 차저스의 타이트 엔드 코치인 미식축구 코치입니다.",
            "브라이언 쇼튼하이머(1973년 10월 16일 출생)는 미국 프로 미식축구 리그(NFL) 인디애나폴리스 콜츠의 쿼터백 코치인 미국 미식축구 코치입니다.",
            "랜디 샌더스(1965년 9월 22일 출생)는 현재 플로리다 주립대학교의 쿼터백 코치 겸 공동 공격 코디네이터로 활동하고 있는 미국의 미식축구 코치입니다.",
            "메이저 리 애플화이트(1978년 7월 26일 출생)는 미국의 미식축구 코치이자 전 선수입니다.",
            "모리스 왓츠(1936년생)는 은퇴한 미국 미식축구 코치이자 전 선수입니다.",
            "스티븐 존 크래고퍼(1965년 4월 28일 출생)는 미식축구 코치이자 전 선수입니다. 그는 2007년부터 2009년까지 루이빌에서, 2003년부터 2006년까지 툴사에서 헤드 코치로 역임했습니다.",
            "11월 1일생(1974년)인 스콧 로펠러는 미식축구 코치이자 전 선수입니다.",
            "밥 샌더스는 현재 내셔널 풋볼 리그(NFL)의 클리블랜드 브라운스에서 공격 보조 코치로 활동하고 있습니다.",
            "마이크 오케인(1954년 7월 20일 출생)은 미국의 미식축구 코치이자 전 선수입니다.",
            "2005년 툴사 골든 허리케인 축구팀은 2005년 NCAA 디비전 I-A 축구 시즌에서 툴사 대학교를 대표했습니다. 팀의 감독은 스티브 크래고였습니다."
        ],
        "truncate": False
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("\n\n🚀 Testing Qwen3 reranker with Tulsa University query...")
    print(f"Query: {data['query'][:50]}...")
    print(f"Number of texts: {len(data['texts'])}")
    print("-" * 80)
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        results = response.json()
        
        if isinstance(results, list):
            print("✅ Response received successfully!")
            
            # Sort by score
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            print("\nTop 3 results:")
            for i in range(min(3, len(sorted_results))):
                idx = sorted_results[i]['index']
                score = sorted_results[i]['score']
                text_preview = data['texts'][idx][:100] + "..."
                print(f"{i+1}. Score = {score:.6f}")
                print(f"   Text: {text_preview}")
                
                # Check if this mentions Tulsa and the years
                if "툴사" in data['texts'][idx] and ("2003" in data['texts'][idx] or "2006" in data['texts'][idx]):
                    print("   ⭐ Contains Tulsa + relevant years!")
            
            return True
            
        else:
            print(f"❌ Unexpected response format")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main test runner."""
    print("=" * 80)
    print("Qwen3 Complete Fix Test")
    print("=" * 80)
    print("\nThis test verifies that:")
    print("1. Tensor shape issue is fixed (no rank mismatch error)")
    print("2. Template formatting is applied (correct semantic ranking)")
    print("=" * 80)
    
    # Test 1: Korean food query
    success1 = test_korean_food_rerank()
    
    # Test 2: Larger dataset
    success2 = test_larger_rerank()
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED: Both fixes are working correctly!")
        sys.exit(0)
    else:
        print("❌ TESTS FAILED: Check the results above")
        sys.exit(1)


if __name__ == "__main__":
    main()