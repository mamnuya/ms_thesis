'''
Takes tokenized/processed bias terms, and 
computes levenshtein distance across original, simple debiased, and complex debiased outputs.
'''
import json
import os
import numpy as np
from Levenshtein import distance as levenshtein_distance
from collections import defaultdict

# Directory containing input JSON files
input_folder = "../../../data/levenshtein_analysis/word_level/biased_terms/"

# List of languages
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

# Function to compute Levenshtein distance at the word level (comparing tokenized lists)
def compute_word_level_levenshtein(words1, words2):
    # Convert word lists back to space-separated strings for Levenshtein calculation
    # Sort words alphabetically to remove order dependency
    str1 = ' '.join(sorted(words1))
    str2 = ' '.join(sorted(words2))
    
    return levenshtein_distance(str1, str2)

# Function to compute average word-level Levenshtein distance per identity within a language
def compute_avg_distance_per_identity(language_data):
    identity_distances = defaultdict(lambda: {"complex": [], "simple": []})

    for entry in language_data:
        identity = entry["identity"]

        # Get the tokenized and lemmatized word lists for the generated and debiased outputs
        original_words = entry["biased_terms_original"]
        complex_words = entry["biased_terms_complex"]
        simple_words = entry["biased_terms_simple"]

        # Compute word-level Levenshtein distances
        complex_dist = compute_word_level_levenshtein(original_words, complex_words)
        simple_dist = compute_word_level_levenshtein(original_words, simple_words)

        # Store distances
        identity_distances[identity]["complex"].append(complex_dist)
        identity_distances[identity]["simple"].append(simple_dist)

    # Compute averages for each identity
    avg_identity_distances = {
        identity: {
            "complex_avg_change_from_original": np.mean(distances["complex"]),
            "simple_avg_change_from_original": np.mean(distances["simple"])
        }
        for identity, distances in identity_distances.items()
    }

    return avg_identity_distances

# Function to compute overall average word-level Levenshtein distance for a language
def compute_avg_distance_across_identities(identity_avg_distances):
    complex_distances = [values["complex_avg_change_from_original"] for values in identity_avg_distances.values()]
    simple_distances = [values["simple_avg_change_from_original"] for values in identity_avg_distances.values()]

    return {
        "complex_avg_change_from_original": np.mean(complex_distances),
        "simple_avg_change_from_original": np.mean(simple_distances)
    }

# Store results
identity_level_results = {}
language_level_results = {}

for lang in languages:
    input_file = os.path.join(input_folder, f"bias_term_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json")

    # Load data
    with open(input_file, "r", encoding="utf-8") as f:
        language_data = json.load(f)

    # Compute average word-level Levenshtein distance per identity
    avg_distance_per_identity = compute_avg_distance_per_identity(language_data)

    # Compute overall averages across all identities
    avg_distance_across_identities = compute_avg_distance_across_identities(avg_distance_per_identity)

    # Store results in required format
    identity_level_results[lang] = avg_distance_per_identity
    language_level_results[lang] = avg_distance_across_identities

# Save results
output_identity_file = "../../../data/levenshtein_analysis/word_level/avg_bias_change_by_debiasing_method/average_word_level_levenshtein_distance_by_identity.json"
output_language_file = "../../../data/levenshtein_analysis/word_level/avg_bias_change_by_debiasing_method/average_word_level_levenshtein_distance_by_language.json"

with open(output_identity_file, "w", encoding="utf-8") as f:
    json.dump(identity_level_results, f, indent=4, ensure_ascii=False)

with open(output_language_file, "w", encoding="utf-8") as f:
    json.dump(language_level_results, f, indent=4, ensure_ascii=False)

print(f"Identity-level results saved to {output_identity_file}")
print(f"Language-level results saved to {output_language_file}")

# Function to compute the average word-level complex/simple Levenshtein distance per identity across all languages
def compute_avg_distance_across_languages(identity_level_results):
    identity_aggregated = defaultdict(lambda: {"complex": [], "simple": []})

    # Aggregate distances for each identity across all languages
    for lang, identities in identity_level_results.items():
        for identity, scores in identities.items():
            identity_aggregated[identity]["complex"].append(scores["complex_avg_change_from_original"])
            identity_aggregated[identity]["simple"].append(scores["simple_avg_change_from_original"])

    # Compute average per identity across languages
    avg_across_languages = {
        identity: {
            "complex_avg_change_from_original": np.mean(scores["complex"]),
            "simple_avg_change_from_original": np.mean(scores["simple"])
        }
        for identity, scores in identity_aggregated.items()
    }

    return avg_across_languages

# Compute averages across all languages and save the result
avg_identity_across_languages = compute_avg_distance_across_languages(identity_level_results)
'''
output_identity_across_languages = "../../../data/levenshtein_analysis/word_level/avg_bias_change_by_debiasing_method/word_level_levenshtein_distance_by_identity_ALL_languages.json"

with open(output_identity_across_languages, "w", encoding="utf-8") as f:
    json.dump(avg_identity_across_languages, f, indent=4, ensure_ascii=False)

print(f"Identity-level averages across languages saved to {output_identity_across_languages}")
'''

def compute_method_avg_distance(language_level_results):
    """Compute the method average of word-level Levenshtein distances across all languages."""
    complex_distances = []
    simple_distances = []

    for lang, values in language_level_results.items():
        complex_distances.append(values["complex_avg_change_from_original"])
        simple_distances.append(values["simple_avg_change_from_original"])

    method_avg = {
        "method_complex_avg_change_from_original": np.mean(complex_distances),
        "method_simple_avg_change_from_original": np.mean(simple_distances)
    }

    return method_avg

# Load per-language average results
language_avg_file = "../../../data/levenshtein_analysis/word_level/avg_bias_change_by_debiasing_method/average_word_level_levenshtein_distance_by_language.json"

with open(language_avg_file, "r", encoding="utf-8") as f:
    language_level_results = json.load(f)

# Compute method average
method_avg_distance = compute_method_avg_distance(language_level_results)

# Save the method average results
method_output_file = "../../../data/levenshtein_analysis/word_level/avg_bias_change_by_debiasing_method/average_word_level_levenshtein_distance_by_method.json"

with open(method_output_file, "w", encoding="utf-8") as f:
    json.dump(method_avg_distance, f, indent=4, ensure_ascii=False)

print(f"Method average word-level Levenshtein distance saved to {method_output_file}")