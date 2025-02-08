'''
Calculate change in bias as a percentage to compare original output versus complex/simple debiasing outputs
'''

import json
import os

def compute_bias_reduction(bias_frequencies):
    """Compute percentage reduction of bias terms for complex and simple debiasing."""
    reduction_results = {}

    for identity, freqs in bias_frequencies.items():
        original = freqs.get("original", {})
        complex_debiased = freqs.get("complex", {})
        simple_debiased = freqs.get("simple", {})

        # original.get(term, 1e-8) prevents division by zero by using a tiny number if the term isn't found.
        reduction_results[identity] = {
            "complex_reduction_from_original": {
                term: ((original.get(term, 0) - complex_debiased.get(term, 0)) / original.get(term, 1e-8)) * 100
                for term in original
            },
            "simple_reduction_from_original": {
                term: ((original.get(term, 0) - simple_debiased.get(term, 0)) / original.get(term, 1e-8)) * 100
                for term in original
            }
        }

    return reduction_results

# Process bias reduction for each language
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

for lang in languages:
    # Load bias frequencies for the current language
    file_path = f"../../data/lexicon_analysis/normalized_frequency/frequency_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json"
    
    with open(file_path, "r", encoding="utf-8") as f:
        bias_frequencies = json.load(f)
    
    # Compute the bias reduction
    bias_reduction_results = compute_bias_reduction(bias_frequencies)

    # Save bias reduction results for the current language
    save_path = f"../../data/lexicon_analysis/bias_reduction/bias_reduction_by_debiasing_method/bias_reduction_{lang}.json"
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(bias_reduction_results, f, indent=4, ensure_ascii=False)

    print(f"Bias reduction results for {lang} saved successfully.")
