'''
Calculate change for each identity and debiasing method

Bias Change % =
((debiasing method frequency - original frequency) / original frequency) * 100

Negative values will indicate a percentage reduction in bias 
(i.e., the debiased method results in fewer occurrences of bias terms compared to the original).

Positive values will indicate a percentage increase in bias 
(i.e., the debiased method results in more occurrences of bias terms compared to the original).

'''

import json
from collections import defaultdict

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_bias_change(normalized_bias_freq):
    """
    Compute the change in bias frequency for each identity and each text type
    (original, complex, and simple debiasing).
    """
    change_results = {}

    for identity, freqs in normalized_bias_freq.items():
        original = freqs.get("original", {})
        complex = freqs.get("complex", {})
        simple = freqs.get("simple", {})

        change_results[identity] = {
            "complex_change_from_original": {
                term: ((complex.get(term, 0) - original.get(term, 0)) / original.get(term, 1e-8)) * 100
                for term in original
            },
            "simple_change_from_original": {
                term: ((simple.get(term, 0) - original.get(term, 0)) / original.get(term, 1e-8)) * 100
                for term in original
            }
        }

    return change_results

def save_json(data, filepath):
    """Save JSON data to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# Load normalized bias frequencies
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

for lang in languages:
    # Load normalized bias frequencies for the current language
    file_path = f"../../data/lexicon_analysis/normalized_frequency/frequency_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json"
    
    with open(file_path, "r", encoding="utf-8") as f:
        normalized_bias_freq = json.load(f)
    
    # Compute the bias change
    bias_change_results = compute_bias_change(normalized_bias_freq)

    # Save bias change results for the current language
    save_path = f"../../data/lexicon_analysis/bias_change/bias_change_by_debiasing_method/bias_change_{lang}.json"
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(bias_change_results, f, indent=4, ensure_ascii=False)

    print(f"Bias change results for {lang} saved successfully.")