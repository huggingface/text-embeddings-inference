import requests
import json

# Base URL
BASE_URL = "http://localhost:3000"

# Test data
test_texts = [
    "Apple is looking at buying U.K. startup for $1 billion",
    "John works at Google in Mountain View, California",
    "The Eiffel Tower is in Paris, France",
]

print("=" * 80)
print("Testing BERT Sequence Classification")
print("=" * 80)

# Test 1: Single sequence classification
print("\n1. Single sequence classification (/predict)")
print("-" * 80)
response = requests.post(f"{BASE_URL}/predict", json={"inputs": [test_texts[0]]})
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 2: Batch sequence classification
print("\n2. Batch sequence classification (/predict)")
print("-" * 80)
response = requests.post(
    f"{BASE_URL}/predict", json={"inputs": [[text] for text in test_texts]}
)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 3: Single token classification
print("\n3. Single token classification (/predict_tokens)")
print("-" * 80)
response = requests.post(
    f"{BASE_URL}/predict_tokens", json={"inputs": [[test_texts[0]]]}
)
print(f"Status: {response.status_code}")
result = response.json()
if isinstance(result, list) and len(result) > 0:
    # Single response
    print(f"Number of tokens: {len(result[0])}")
    print(f"First 5 tokens:")
    for i, token_pred in enumerate(result[0][:5]):
        print(f"  Token {i}: '{token_pred['token']}' (id={token_pred['token_id']})")
        print(f"    Start: {token_pred['start']}, End: {token_pred['end']}")
        # Get the best prediction (highest score)
        if "results" in token_pred:
            best_label = max(token_pred["results"].items(), key=lambda x: x[1])
            print(f"    Best prediction: {best_label[0]} ({best_label[1]:.4f})")
            print(
                f"    All predictions: {dict(sorted(token_pred['results'].items(), key=lambda x: x[1], reverse=True))}"
            )
        else:
            print(
                f"    Prediction: {token_pred.get('label', 'N/A')} ({token_pred.get('score', 'N/A'):.4f})"
            )
else:
    print(f"Response: {json.dumps(result, indent=2)}")

# Test 4: Batch token classification
print("\n4. Batch token classification (/predict_tokens)")
print("-" * 80)
response = requests.post(
    f"{BASE_URL}/predict_tokens", json={"inputs": [[text] for text in test_texts]}
)
print(f"Status: {response.status_code}")
result = response.json()
if isinstance(result, list):
    # Batch response
    print(f"Number of sequences: {len(result)}")
    for seq_idx, token_preds in enumerate(result):
        print(f"\n  Sequence {seq_idx}: '{test_texts[seq_idx]}'")
        print(f"  Number of tokens: {len(token_preds)}")
        print(f"  First 3 tokens:")
        for i, token_pred in enumerate(token_preds[:3]):
            print(
                f"    Token {i}: '{token_pred['token']}' (id={token_pred['token_id']})"
            )
            if "results" in token_pred:
                best_label = max(token_pred["results"].items(), key=lambda x: x[1])
                print(f"      Best prediction: {best_label[0]} ({best_label[1]:.4f})")
            else:
                print(
                    f"      Prediction: {token_pred.get('label', 'N/A')} ({token_pred.get('score', 'N/A'):.4f})"
                )
else:
    print(f"Response: {json.dumps(result, indent=2)}")

# Test 5: Token classification with raw scores
print("\n5. Token classification with raw scores (/predict_tokens)")
print("-" * 80)
response = requests.post(
    f"{BASE_URL}/predict_tokens", json={"inputs": [[test_texts[0]]], "raw_scores": True}
)
print(f"Status: {response.status_code}")
result = response.json()
if isinstance(result, list) and len(result) > 0:
    print(f"Number of tokens: {len(result[0])}")
    print(f"First token:")
    token_pred = result[0][0]
    print(f"  Token: '{token_pred['token']}' (id={token_pred['token_id']})")
    if "results" in token_pred:
        best_label = max(token_pred["results"].items(), key=lambda x: x[1])
        print(f"  Best prediction: {best_label[0]} ({best_label[1]:.4f})")
        print(
            f"  All raw scores: {dict(sorted(token_pred['results'].items(), key=lambda x: x[1], reverse=True))}"
        )
    else:
        print(
            f"  Prediction: {token_pred.get('label', 'N/A')} ({token_pred.get('score', 'N/A'):.4f})"
        )
else:
    print(f"Response: {json.dumps(result, indent=2)}")

print("\n" + "=" * 80)
print("Tests completed!")
print("=" * 80)
