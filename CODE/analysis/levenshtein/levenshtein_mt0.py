'''
pip install levenshtein

Levenshtein distance: a method to determine the minimum number of single-character edits 
(insertions, deletions, or substitutions) needed to transform one string into another
'''

import json
import os
import numpy as np
from Levenshtein import distance as levenshtein_distance
from collections import defaultdict

# Directory containing input JSON files
input_folder = "../../../data/complex_and_simple_debiaspromptsQs/cleaned_tokenized_lemmatized/"

# List of languages
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

# Function to compute Levenshtein distance
def compute_levenshtein(text1, text2):
    return levenshtein_distance(text1, text2)

# Function to compute average Levenshtein distance per identity within a language
def compute_avg_distance_per_identity(language_data):
    identity_distances = defaultdict(lambda: {"complex": [], "simple": []})

    for entry in language_data:
        identity = entry["identity"]

        # Get the cleaned generated output and debiased outputs
        gen_text = entry["cleaned_translated_generated_output"]
        complex_text = entry["complex_cleaned_translated_debiased_output"]
        simple_text = entry["simple_cleaned_translated_debiased_output"]

        # Compute Levenshtein distances
        complex_dist = compute_levenshtein(gen_text, complex_text)
        simple_dist = compute_levenshtein(gen_text, simple_text)

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

# Function to compute overall average Levenshtein distance for a language
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
    input_file = os.path.join(input_folder, f"generated_data_{lang}_10k_mt0xxl_with_complex_and_simple_debiasing.json")

    # Load data
    with open(input_file, "r", encoding="utf-8") as f:
        language_data = json.load(f)

    # Compute average Levenshtein distance per identity
    avg_distance_per_identity = compute_avg_distance_per_identity(language_data)

    # Compute overall averages across all identities
    avg_distance_across_identities = compute_avg_distance_across_identities(avg_distance_per_identity)

    # Store results in required format
    identity_level_results[lang] = avg_distance_per_identity
    language_level_results[lang] = avg_distance_across_identities

# Save results
output_identity_file = "../../../data/levenshtein_analysis/avg_bias_change_by_debiasing_method/average_levenshtein_distance_by_identity.json"
output_language_file = "../../../data/levenshtein_analysis/avg_bias_change_by_debiasing_method/average_levenshtein_distance_by_method_ALL_languages.json"

with open(output_identity_file, "w", encoding="utf-8") as f:
    json.dump(identity_level_results, f, indent=4, ensure_ascii=False)

with open(output_language_file, "w", encoding="utf-8") as f:
    json.dump(language_level_results, f, indent=4, ensure_ascii=False)

print(f"Identity-level results saved to {output_identity_file}")
print(f"Language-level results saved to {output_language_file}")

# Function to compute the average complex/simple Levenshtein distance per identity across all languages
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
output_identity_across_languages = "../../../data/levenshtein_analysis/avg_bias_change_by_debiasing_method/levenshtein_distance_by_identity_ALL_languages.json"

with open(output_identity_across_languages, "w", encoding="utf-8") as f:
    json.dump(avg_identity_across_languages, f, indent=4, ensure_ascii=False)

print(f"Identity-level averages across languages saved to {output_identity_across_languages}")