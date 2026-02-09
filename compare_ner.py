"""
Compare BEI NER predictions with HuggingFace Transformers
"""

import requests
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

# Configuration
TEI_URL = "http://localhost:3000"
MODEL_PATH = "./bert-base-NER-uncased"

# Test texts
test_texts = [
    "Apple is looking at buying U.K. startup for $1 billion",
    "John works at Google in Mountain View, California",
    "The Eiffel Tower is in Paris, France",
]

print("=" * 80)
print("Comparing BEI vs Transformers NER Predictions")
print("=" * 80)

# Load transformers model
print("\nLoading Transformers model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()
label_list = model.config.id2label

print(f"Model labels: {label_list}")
print(f"Number of labels: {len(label_list)}")

# Compare predictions for each text
for text_idx, text in enumerate(test_texts):
    print(f"\n{'=' * 80}")
    print(f"Text {text_idx + 1}: '{text}'")
    print(f"{'=' * 80}")

    # Get BEI predictions
    print("\n--- BEI Predictions ---")
    bei_response = requests.post(
        f"{TEI_URL}/predict_tokens", json={"inputs": [[text]], "raw_scores": False}
    )
    bei_data = bei_response.json()

    # Get Transformers predictions
    print("\n--- Transformers Predictions ---")
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)

    # Get tokens and labels from transformers
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tf_predictions = [label_list[p.item()] for p in predictions[0]]
    tf_scores = torch.softmax(logits, dim=2)[0]

    # Compare token by token
    print(
        f"\n{'Token':<20} {'BEI Label':<15} {'TF Label':<15} {'BEI Score':<10} {'TF Score':<10} {'Match':<10}"
    )
    print("-" * 90)

    # BEI returns predictions for all tokens (including special tokens)
    # Transformers also returns predictions for all tokens
    # We need to align them

    # Get BEI tokens and predictions
    bei_tokens = []
    bei_labels = []
    bei_scores = []

    if isinstance(bei_data, list) and len(bei_data) > 0:
        # bei_data is a list of token predictions (one per token)
        for token_pred in bei_data[0]:  # bei_data[0] contains the list of tokens
            if "results" in token_pred:
                # New format: one prediction per token with results dictionary
                bei_tokens.append(token_pred["token"])
                # Find the label with highest score
                best_label = max(token_pred["results"].items(), key=lambda x: x[1])
                bei_labels.append(best_label[0])
                bei_scores.append(best_label[1])
            elif "label" in token_pred and "score" in token_pred:
                # Old format: multiple predictions per token
                bei_tokens.append(token_pred["token"])
                bei_labels.append(token_pred["label"])
                bei_scores.append(token_pred["score"])
    else:
        print(f"Unexpected BEI response format: {type(bei_data)}")
        print(f"Response: {json.dumps(bei_data, indent=2)}")
        raise ValueError(f"Unexpected BEI response format: {type(bei_data)}")

    # Compare
    matches = 0
    total = 0

    for i, (bei_token, bei_label, bei_score) in enumerate(
        zip(bei_tokens, bei_labels, bei_scores)
    ):
        if i >= len(tokens):
            break

        tf_token = tokens[i]
        tf_label = tf_predictions[i]
        tf_score = tf_scores[i][predictions[0][i]].item()

        # Check if labels match
        match = "✓" if bei_label == tf_label else "✗"
        if bei_label == tf_label:
            matches += 1
        total += 1

        print(
            f"{bei_token:<20} {bei_label:<15} {tf_label:<15} {bei_score:<10.4f} {tf_score:<10.4f} {match:<10}"
        )
    assert len(bei_tokens) == len(tokens), (
        "Token length mismatch between BEI and Transformers"
    )

    # Print summary
    accuracy = matches / total if total > 0 else 0
    print(f"\nSummary: {matches}/{total} tokens match ({accuracy:.2%} accuracy)")

# Detailed comparison for first text
print(f"\n{'=' * 80}")
print("Detailed Score Comparison (First Text)")
print(f"{'=' * 80}")

text = test_texts[0]

# Get BEI predictions with raw scores
bei_response = requests.post(
    f"{TEI_URL}/predict_tokens", json={"inputs": [[text]], "raw_scores": False}
)
bei_data = bei_response.json()

# Get Transformers predictions
with torch.no_grad():
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    tf_scores = torch.softmax(logits, dim=2)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Compare scores for each token
if isinstance(bei_data, list) and len(bei_data) > 0:
    for i, token_pred in enumerate(
        bei_data[0]
    ):  # bei_data[0] contains the list of tokens
        if i >= len(tokens):
            break

        if "results" not in token_pred:
            continue

        bei_token = token_pred["token"]
        tf_token = tokens[i]

        print(f"\nToken {i}: '{bei_token}' (TF: '{tf_token}')")
        print(f"{'Label':<20} {'BEI Score':<15} {'TF Score':<15} {'Diff':<15}")
        print("-" * 65)

        # Get all label scores from BEI
        bei_label_scores = token_pred["results"]

        # Get all label scores from TF
        tf_token_scores = tf_scores[i]

        # Compare for each label
        for label_idx, label in label_list.items():
            bei_score = bei_label_scores.get(label, 0.0)
            tf_score = tf_token_scores[label_idx].item()
            diff = abs(bei_score - tf_score)

            print(f"{label:<20} {bei_score:<15.6f} {tf_score:<15.6f} {diff:<15.6f}")

print(f"\n{'=' * 80}")
print("Comparison completed!")
print(f"{'=' * 80}")
