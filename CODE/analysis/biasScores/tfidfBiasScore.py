import json
import os

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Define input file paths
identity_avg_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_identity.json"
language_avg_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_language.json"
method_avg_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_method.json"

# Load data
identity_avg_change = load_json(identity_avg_file)
language_avg_change = load_json(language_avg_file)
method_avg_change = load_json(method_avg_file)

# Compute Identity-Wise TF-IDF Bias Scores
identity_bias_scores = {}

for lang, identities in identity_avg_change.items():
    language_avg_complex = language_avg_change[lang]["complex_avg_change_from_original"]
    language_avg_simple = language_avg_change[lang]["simple_avg_change_from_original"]
    
    identity_bias_scores[lang] = {}
    
    for identity, values in identities.items():
        identity_complex = values["complex_avg_change"]
        identity_simple = values["simple_avg_change"]
        
        # Compute bias scores
        identity_complex_bias_score = identity_complex / language_avg_complex if language_avg_complex else 0
        identity_simple_bias_score = identity_simple / language_avg_simple if language_avg_simple else 0

        identity_bias_scores[lang][identity] = {
            "complex_bias_score": identity_complex_bias_score,
            "simple_bias_score": identity_simple_bias_score
        }

# Save identity-wise bias scores
identity_bias_output = "../../../data/bias_scores_analysis/tfidf_identity_bias_scores.json"

#print(identity_bias_scores)

with open(identity_bias_output, "w", encoding="utf-8") as f:
    json.dump(identity_bias_scores, f, indent=4, ensure_ascii=False)
print(f"Identity-wise TF-IDF bias scores saved to {identity_bias_output}")

# Compute Language-Wise TF-IDF Bias Scores
method_avg_complex = method_avg_change["method_complex_avg_change_from_original"]
method_avg_simple = method_avg_change["method_simple_avg_change_from_original"]

language_bias_scores = {}

for lang, values in language_avg_change.items():
    language_complex = values["complex_avg_change_from_original"]
    language_simple = values["simple_avg_change_from_original"]
    
    # Compute bias scores
    language_complex_bias_score = language_complex / method_avg_complex if method_avg_complex else 0
    language_simple_bias_score = language_simple / method_avg_simple if method_avg_simple else 0

    language_bias_scores[lang] = {
        "complex_bias_score": language_complex_bias_score,
        "simple_bias_score": language_simple_bias_score
    }

# Save language-wise bias scores
language_bias_output = "../../../data/bias_scores_analysis/tfidf_language_bias_scores.json"

#print(language_bias_scores)

with open(language_bias_output, "w", encoding="utf-8") as f:
    json.dump(language_bias_scores, f, indent=4, ensure_ascii=False)
print(f"Language-wise TF-IDF bias scores saved to {language_bias_output}")
