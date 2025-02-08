'''
compute averages from biasReduction percentages which compared original versus simple/complex debiasing per identity group.
'''

import json
import pandas as pd
import os

# Directory where bias reduction files are stored
bias_reduction_dir = "../../data/lexicon_analysis/bias_reduction/bias_reduction_by_debiasing_method/"

# Aggregate bias reduction across languages
language_comparisons = []

# List of languages
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

for lang in languages:
    file_path = os.path.join(bias_reduction_dir, f"bias_reduction_{lang}.json")

    # Load bias reduction results for the current language
    with open(file_path, "r", encoding="utf-8") as f:
        bias_reduction_results = json.load(f)

    for identity, reductions in bias_reduction_results.items():
        avg_complex_reduction_from_original = sum(reductions["complex_reduction_from_original"].values()) / len(reductions["complex_reduction_from_original"])
        avg_simple_reduction_from_original = sum(reductions["simple_reduction_from_original"].values()) / len(reductions["simple_reduction_from_original"])

        language_comparisons.append({
            "language": lang,
            "identity": identity,
            "avg_complex_reduction_from_original": avg_complex_reduction_from_original,
            "avg_simple_reduction_from_original": avg_simple_reduction_from_original
        })

# Convert to DataFrame for easy analysis
df = pd.DataFrame(language_comparisons)

# Compute overall effectiveness of debiasing methods
avg_complex_per_lang = df.groupby("language")["avg_complex_reduction_from_original"].mean()
avg_simple_per_lang = df.groupby("language")["avg_simple_reduction_from_original"].mean()

# Save results to CSV
df.to_csv("../../data/lexicon_analysis/bias_reduction/avg_bias_reduction_by_debiasing_method_per_language/cross_language_bias_reduction.csv", index=False)
print("Cross-language bias reduction comparison saved as CSV.")
