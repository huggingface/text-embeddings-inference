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


def compare_entities(tei_entities, hf_entities):
    """Compare entities between TEI and HF"""
    tei_norm = normalize_entities(tei_entities)
    hf_norm = normalize_entities(hf_entities)

    # Calculate metrics
    exact_matches = 0
    partial_matches = 0

    # Create word-entity mappings for comparison
    tei_word_to_entity = {e["word"]: e for e in tei_norm}
    hf_word_to_entity = {e["word"]: e for e in hf_norm}

    # Check matches
    all_words = set(tei_word_to_entity.keys()) | set(hf_word_to_entity.keys())

    matches = []
    for word in all_words:
        tei_entity = tei_word_to_entity.get(word)
        hf_entity = hf_word_to_entity.get(word)

        if tei_entity and hf_entity:
            # Both have this word
            entity_match = tei_entity["entity_group"] == hf_entity["entity_group"]
            score_diff = abs(tei_entity["score"] - hf_entity["score"])

            if entity_match and score_diff < 0.1:
                exact_matches += 1
                matches.append(
                    f"✓ {word}: {tei_entity['entity_group']} (diff: {score_diff:.3f})"
                )
            elif entity_match:
                partial_matches += 1
                matches.append(
                    f"~ {word}: {tei_entity['entity_group']} (diff: {score_diff:.3f})"
                )
            else:
                matches.append(
                    f"✗ {word}: TEI={tei_entity['entity_group']} vs HF={hf_entity['entity_group']}"
                )
        elif tei_entity:
            matches.append(f"- {word}: TEI only ({tei_entity['entity_group']})")
        elif hf_entity:
            matches.append(f"+ {word}: HF only ({hf_entity['entity_group']})")

    return {
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "total_te": len(tei_norm),
        "total_hf": len(hf_norm),
        "matches": matches,
    }


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
            comparison = compare_entities(tei_pred, hf_pred)

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
