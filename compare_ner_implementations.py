import requests
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import numpy as np
from collections import defaultdict

# Base URL for our implementation
BASE_URL = "http://localhost:8000"

# Model name
MODEL_NAME = "./bert-base-NER-uncased"

print("=" * 80)
print("Comparing TEI NER Implementation vs Hugging Face Transformers")
print("=" * 80)
print(f"Model: {MODEL_NAME}")

# Load Hugging Face pipeline for comparison
print("\nLoading Hugging Face pipeline...")
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
hf_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, device=device, aggregation_strategy="max"
)

print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Representative test dataset covering various NER challenges
test_sentences = [
    # Basic person names
    "John Doe works at Microsoft.",
    "Mary Smith lives in New York City.",
    # Organizations
    "Apple Inc. is headquartered in Cupertino, California.",
    "Google LLC acquired YouTube for $1.65 billion.",
    # Locations
    "The Eiffel Tower is in Paris, France.",
    "Mount Everest is on the border of Nepal and Tibet.",
    # Mixed entities
    "Tim Cook, CEO of Apple, announced the new iPhone at the Steve Jobs Theater in Cupertino.",
    "Satya Nadella from Microsoft visited the White House in Washington, D.C.",
    # Complex names and titles
    "Dr. Martin Luther King Jr. gave his famous speech in Washington, D.C.",
    "Professor Albert Einstein taught at Princeton University in New Jersey.",
    # Financial and numerical entities
    "Microsoft bought GitHub for $7.5 billion in 2018.",
    "Amazon's market cap exceeded $1 trillion in 2020.",
    # Ambiguous cases
    "Washington was the first president of the United States.",
    "Paris Hilton is a celebrity, not the city in France.",
    # Multi-word entities
    "The United Nations is headquartered in New York City.",
    "San Francisco International Airport serves the Bay Area.",
    # Edge cases with punctuation
    "J. K. Rowling wrote Harry Potter in Edinburgh, Scotland.",
    "J. D. Vance authored 'Hillbilly Elegy' about Ohio.",
]


def get_tei_predictions(sentence, strategy="max"):
    """Get predictions from TEI implementation"""
    try:
        response = requests.post(
            f"{BASE_URL}/predict_tokens",
            json={"inputs": [sentence], "aggregation_strategy": strategy},
        )
        if response.status_code == 200:
            return response.json()[0]
        else:
            print(f"TEI Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"TEI Request Error: {e}")
        return None


def get_hf_predictions(sentence):
    """Get predictions from Hugging Face pipeline"""
    try:
        return hf_pipeline(sentence)
    except Exception as e:
        print(f"HF Pipeline Error: {e}")
        return None


def normalize_entities(entities):
    """Normalize entity format for comparison"""
    normalized = []
    for entity in entities:
        if isinstance(entity, dict):
            # Handle TEI format
            if "token" in entity and "results" in entity:
                # TEI format
                best_label = max(entity["results"].items(), key=lambda x: x[1])
                normalized.append(
                    {
                        "word": entity["token"],
                        "entity_group": best_label[0],
                        "score": best_label[1],
                        "start": entity.get("start"),
                        "end": entity.get("end"),
                    }
                )
            elif "word" in entity and "entity_group" in entity:
                # HF format
                normalized.append(entity)
    return normalized


def compare_entities(tei_entities, hf_entities, sentence):
    """Compare entities between TEI and HF using position-based comparison"""

    # Create position-based mappings
    tei_by_position = {}
    hf_by_position = {}

    for entity in tei_entities:
        start = entity.get("start")
        end = entity.get("end")
        if start is not None and end is not None:
            key = (start, end)
            tei_by_position[key] = entity

    for entity in hf_entities:
        start = entity.get("start")
        end = entity.get("end")
        if start is not None and end is not None:
            key = (start, end)
            hf_by_position[key] = entity

    # Compare by position
    exact_matches = 0
    partial_matches = 0
    matches = []

    all_positions = set(tei_by_position.keys()) | set(hf_by_position.keys())

    for pos in all_positions:
        tei_entity = tei_by_position.get(pos)
        hf_entity = hf_by_position.get(pos)

        if tei_entity and hf_entity:
            # Both have this position
            tei_label = get_entity_label(tei_entity)
            hf_label = get_entity_label(hf_entity)

            tei_score = get_entity_score(tei_entity)
            hf_score = get_entity_score(hf_entity)

            # Get the actual text from the sentence
            start, end = pos
            actual_text = sentence[start:end]

            label_match = tei_label == hf_label
            score_diff = abs(tei_score - hf_score)

            if label_match and score_diff < 0.1:
                exact_matches += 1
                matches.append(
                    f"✓ [{start},{end}): '{actual_text}' - {tei_label} (diff: {score_diff:.3f})"
                )
            elif label_match:
                partial_matches += 1
                matches.append(
                    f"~ [{start},{end}): '{actual_text}' - {tei_label} (diff: {score_diff:.3f})"
                )
            else:
                matches.append(
                    f"✗ [{start},{end}): '{actual_text}' - TEI={tei_label} vs HF={hf_label}"
                )
        elif tei_entity:
            start, end = pos
            actual_text = sentence[start:end]
            tei_label = get_entity_label(tei_entity)
            matches.append(
                f"- [{start},{end}): '{actual_text}' - TEI only ({tei_label})"
            )
        elif hf_entity:
            start, end = pos
            actual_text = sentence[start:end]
            hf_label = get_entity_label(hf_entity)
            matches.append(f"+ [{start},{end}): '{actual_text}' - HF only ({hf_label})")

    return {
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "total_te": len(tei_by_position),
        "total_hf": len(hf_by_position),
        "matches": matches,
    }


def get_entity_label(entity):
    """Extract entity label from different formats"""
    if "results" in entity:
        # TEI format
        best_label = max(entity["results"].items(), key=lambda x: x[1])
        return best_label[0]
    elif "entity_group" in entity:
        # HF format
        return entity["entity_group"]
    return "UNKNOWN"


def get_entity_score(entity):
    """Extract entity score from different formats"""
    if "results" in entity:
        # TEI format
        best_label = max(entity["results"].items(), key=lambda x: x[1])
        return best_label[1]
    elif "score" in entity:
        # HF format
        return entity["score"]
    return 0.0


# Test each strategy
strategies = ["simple", "first", "max", "average"]
results = defaultdict(dict)

for strategy in strategies:
    print(f"\n{'=' * 80}")
    print(f"Testing Strategy: {strategy.upper()}")
    print(f"{'=' * 80}")

    strategy_results = {
        "exact_matches": 0,
        "partial_matches": 0,
        "total_te": 0,
        "total_hf": 0,
        "sentence_results": [],
    }

    for i, sentence in enumerate(test_sentences):
        print(f"\n{i + 1}. Testing: '{sentence}'")
        print("-" * 60)

        # Get predictions
        tei_pred = get_tei_predictions(sentence, strategy)
        hf_pred = get_hf_predictions(sentence)

        if tei_pred and hf_pred:
            comparison = compare_entities(tei_pred, hf_pred, sentence)

            print(
                f"TEI entities: {comparison['total_te']}, HF entities: {comparison['total_hf']}"
            )
            print(
                f"Exact matches: {comparison['exact_matches']}, Partial: {comparison['partial_matches']}"
            )

            # Show detailed comparison for first few sentences
            if i < 3:
                print("\nDetailed comparison:")
                for match in comparison["matches"]:
                    print(f"  {match}")

            # Accumulate results
            strategy_results["exact_matches"] += comparison["exact_matches"]
            strategy_results["partial_matches"] += comparison["partial_matches"]
            strategy_results["total_te"] += comparison["total_te"]
            strategy_results["total_hf"] += comparison["total_hf"]
            strategy_results["sentence_results"].append(comparison)
        else:
            print(f"Failed to get predictions for sentence {i + 1}")

    results[strategy] = strategy_results

# Summary report
print(f"\n{'=' * 80}")
print("SUMMARY REPORT")
print(f"{'=' * 80}")

print(f"\nTested {len(test_sentences)} sentences across {len(strategies)} strategies")
print(f"Model: {MODEL_NAME}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

print(
    f"\n{'Strategy':<12} {'TEI Entities':<12} {'HF Entities':<12} {'Exact':<8} {'Partial':<8} {'Match Rate':<12}"
)
print("-" * 70)

for strategy in strategies:
    result = results[strategy]
    total_possible = result["total_te"] + result["total_hf"] - result["exact_matches"]
    match_rate = (result["exact_matches"] / max(total_possible, 1)) * 100

    print(
        f"{strategy:<12} {result['total_te']:<12} {result['total_hf']:<12} "
        f"{result['exact_matches']:<8} {result['partial_matches']:<8} {match_rate:<11.1f}%"
    )

# Find best strategy
best_strategy = max(strategies, key=lambda s: results[s]["exact_matches"])
print(f"\nBest performing strategy: {best_strategy.upper()}")
print(f"Exact matches: {results[best_strategy]['exact_matches']}")

print(f"\n{'=' * 80}")
print("Analysis complete!")
print(f"{'=' * 80}")
