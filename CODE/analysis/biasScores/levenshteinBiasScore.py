import json
import os

# File paths for input data
identity_avg_file = "../../../data/levenshtein_analysis/word_level/avg_bias_change_by_debiasing_method/average_word_level_levenshtein_distance_by_identity.json"
language_avg_file = "../../../data/levenshtein_analysis/word_level/avg_bias_change_by_debiasing_method/average_word_level_levenshtein_distance_by_language.json"
method_avg_file = "../../../data/levenshtein_analysis/word_level/avg_bias_change_by_debiasing_method/average_word_level_levenshtein_distance_by_method.json"

# Output file paths
identity_bias_score_file = "../../../data/bias_scores_analysis/identity_wise_levenshtein_bias_score.json"
language_bias_score_file = "../../../data/bias_scores_analysis/language_wise_levenshtein_bias_score.json"

# Load data
with open(identity_avg_file, "r", encoding="utf-8") as f:
    identity_avg_distances = json.load(f)

with open(language_avg_file, "r", encoding="utf-8") as f:
    language_avg_distances = json.load(f)

with open(method_avg_file, "r", encoding="utf-8") as f:
    method_avg_distance = json.load(f)

# Compute Identity-Wise Levenshtein Bias Score
identity_bias_scores = {}

for lang, identities in identity_avg_distances.items():
    identity_bias_scores[lang] = {}

    # Get the language-wise average distances
    lang_complex_avg = language_avg_distances[lang]["complex_avg_change_from_original"]
    lang_simple_avg = language_avg_distances[lang]["simple_avg_change_from_original"]

    for identity, scores in identities.items():
        identity_bias_scores[lang][identity] = {
            "complex_bias_score": scores["complex_avg_change_from_original"] / lang_complex_avg if lang_complex_avg else 0,
            "simple_bias_score": scores["simple_avg_change_from_original"] / lang_simple_avg if lang_simple_avg else 0
        }

# Save Identity-Wise Bias Scores
with open(identity_bias_score_file, "w", encoding="utf-8") as f:
    json.dump(identity_bias_scores, f, indent=4, ensure_ascii=False)

print(f"Identity-wise Levenshtein bias scores saved to {identity_bias_score_file}")




# PRINT the identity with the highest and lowest absolute bias scores for each language
for lang, identities in identity_bias_scores.items():
    # Get the identity with the highest absolute complex bias score
    top_complex_identity = max(identities.items(), key=lambda x: abs(x[1]["complex_bias_score"]))
    top_complex_name, top_complex_values = top_complex_identity

    # Get the identity with the lowest absolute complex bias score
    lowest_complex_identity = min(identities.items(), key=lambda x: abs(x[1]["complex_bias_score"]))
    lowest_complex_name, lowest_complex_values = lowest_complex_identity

    # Get the identity with the highest absolute simple bias score
    top_simple_identity = max(identities.items(), key=lambda x: abs(x[1]["simple_bias_score"]))
    top_simple_name, top_simple_values = top_simple_identity

    # Get the identity with the lowest absolute simple bias score
    lowest_simple_identity = min(identities.items(), key=lambda x: abs(x[1]["simple_bias_score"]))
    lowest_simple_name, lowest_simple_values = lowest_simple_identity

    print(f" - Levenshtein Bias Score - ")
    print(f"Language: {lang}")
    print(f"  Top Complex Bias Score: {top_complex_name} -> {top_complex_values['complex_bias_score']:.6f}")
    print(f"  Top Simple Bias Score: {top_simple_name} -> {top_simple_values['simple_bias_score']:.6f}")
    print(f"  Lowest Complex Bias Score: {lowest_complex_name} -> {lowest_complex_values['complex_bias_score']:.6f}")
    print(f"  Lowest Simple Bias Score: {lowest_simple_name} -> {lowest_simple_values['simple_bias_score']:.6f}")
    print("-" * 50)


# Compute Language-Wise Levenshtein Bias Score
language_bias_scores = {}

# Get overall method-wise averages
method_complex_avg = method_avg_distance["method_complex_avg_change_from_original"]
method_simple_avg = method_avg_distance["method_simple_avg_change_from_original"]

for lang, values in language_avg_distances.items():
    language_bias_scores[lang] = {
        "complex_bias_score": values["complex_avg_change_from_original"] / method_complex_avg if method_complex_avg else 0,
        "simple_bias_score": values["simple_avg_change_from_original"] / method_simple_avg if method_simple_avg else 0
    }

#print (language_bias_scores)

# Save Language-Wise Bias Scores
with open(language_bias_score_file, "w", encoding="utf-8") as f:
    json.dump(language_bias_scores, f, indent=4, ensure_ascii=False)
print(f"Language-wise Levenshtein bias scores saved to {language_bias_score_file}")
