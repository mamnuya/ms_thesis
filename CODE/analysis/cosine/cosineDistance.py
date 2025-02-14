import json
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Directory containing input JSON files
input_folder = "../../../data/complex_and_simple_debiaspromptsQs/cleaned_tokenized_lemmatized/"
tfidf_folder = "../../../data/lexicon_analysis/tfidf/tfidf_scores/"
output_folder = "../../../data/cosine_analysis/avg_bias_change_by_debiasing_method/"

# List of languages
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

# Function to compute cosine distance using TF-IDF scores
def compute_cosine_distance_tfidf(tfidf_original, tfidf_variant):
    """Compute cosine distance between two TF-IDF vectors."""
    terms = set(tfidf_original.keys()).union(set(tfidf_variant.keys()))
    
    # Convert term dictionaries into aligned vectors
    vec1 = np.array([tfidf_original.get(term, 0) for term in terms])
    vec2 = np.array([tfidf_variant.get(term, 0) for term in terms])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    
    # Cosine distance is 1 - cosine similarity
    return 1 - cosine_sim

# Function to compute average cosine distance per identity within a language using TF-IDF
def compute_avg_cosine_distance_per_identity(language_data, tfidf_scores):
    identity_distances = defaultdict(lambda: {"complex": [], "simple": []})

    for entry in language_data:
        identity = entry["identity"]

        # Retrieve TF-IDF scores for original and debiased versions
        tfidf_original = tfidf_scores.get(identity, {}).get("original", {})
        tfidf_complex = tfidf_scores.get(identity, {}).get("complex", {})
        tfidf_simple = tfidf_scores.get(identity, {}).get("simple", {})

        # Compute cosine distances using TF-IDF vectors
        complex_dist = compute_cosine_distance_tfidf(tfidf_original, tfidf_complex)
        simple_dist = compute_cosine_distance_tfidf(tfidf_original, tfidf_simple)

        # Store distances
        identity_distances[identity]["complex"].append(complex_dist)
        identity_distances[identity]["simple"].append(simple_dist)

    # Compute average distances per identity
    avg_identity_distances = {
        identity: {
            "complex_avg_change_from_original": np.mean(distances["complex"]),
            "simple_avg_change_from_original": np.mean(distances["simple"])
        }
        for identity, distances in identity_distances.items()
    }

    return avg_identity_distances

# Function to compute overall average cosine distance for a language
def compute_avg_cosine_distance_across_identities(identity_avg_distances):
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
    tfidf_file = os.path.join(tfidf_folder, f"tfidf_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json")

    # Load dataset
    with open(input_file, "r", encoding="utf-8") as f:
        language_data = json.load(f)

    # Load TF-IDF scores
    with open(tfidf_file, "r", encoding="utf-8") as f:
        tfidf_scores = json.load(f)

    # Compute average cosine distance per identity
    avg_distance_per_identity = compute_avg_cosine_distance_per_identity(language_data, tfidf_scores)

    # Compute overall averages across all identities
    avg_distance_across_identities = compute_avg_cosine_distance_across_identities(avg_distance_per_identity)

    # Store results
    identity_level_results[lang] = avg_distance_per_identity
    language_level_results[lang] = avg_distance_across_identities

# Save results
output_identity_file = os.path.join(output_folder, "average_cosine_distance_by_identity.json")
output_language_file = os.path.join(output_folder, "average_cosine_distance_by_method_ALL_languages.json")

with open(output_identity_file, "w", encoding="utf-8") as f:
    json.dump(identity_level_results, f, indent=4, ensure_ascii=False)

with open(output_language_file, "w", encoding="utf-8") as f:
    json.dump(language_level_results, f, indent=4, ensure_ascii=False)

print(f"Identity-level results saved to {output_identity_file}")
print(f"Language-level results saved to {output_language_file}")

# Function to compute the average cosine distance per identity across all languages
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
output_identity_across_languages = os.path.join(output_folder, "cosine_distance_by_identity_ALL_languages.json")

with open(output_identity_across_languages, "w", encoding="utf-8") as f:
    json.dump(avg_identity_across_languages, f, indent=4, ensure_ascii=False)

print(f"Identity-level averages across languages saved to {output_identity_across_languages}")