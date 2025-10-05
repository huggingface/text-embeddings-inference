#!/usr/bin/env python3
"""Verify Qwen3 reranker fix"""
import requests

url = "http://localhost:8080/rerank"
data = {
    "query": "한국의 음식",
    "texts": ["부산", "뉴욕", "김치"],
    "truncate": False
}

response = requests.post(url, json=data)
if response.status_code == 200:
    results = response.json()
    print("Results:")
    for r in results:
        text = data["texts"][r["index"]]
        print(f"  {text}: {r['score']:.4f}")
    
    # Check semantic correctness
    scores = {data["texts"][r["index"]]: r["score"] for r in results}
    if scores["김치"] > scores["부산"] > scores["뉴욕"]:
        print("\n✓ Ranking is semantically correct!")
    else:
        print("\n✗ Ranking issue still exists")
else:
    print(f"Error: {response.text}")