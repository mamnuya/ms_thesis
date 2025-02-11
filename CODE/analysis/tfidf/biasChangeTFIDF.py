'''
Calculate change for each identity and debiasing method

Bias Change % =
((debiasing method frequency - original frequency) / original frequency) * 100

Negative values will indicate a percentage decrease in bias 
(i.e., the debiased method results in fewer occurrences of bias terms compared to the original).

Positive values will indicate a percentage increase in bias 
(i.e., the debiased method results in more occurrences of bias terms compared to the original).

How to Interpret the TF-IDF Bias Change (%)
The percentage tells you how much the presence of a term increased or decreased in the complex or 
simple debiased outputs relative to the original output.

Bias Change %	Interpretation
Negative (-X%)	Term appears less often after debiasing (Bias decreased)
Positive (+X%)	Term appears more often after debiasing (Bias increased)
Zero (0%)	No change in the term's presence

Example:
complex bias change for the word "household": -20%
The word "household" appears 20% less in the complex debiased text, suggesting debiasing reduced its emphasis.

'''

### Step 1: Compare within each language
import json
from collections import defaultdict

import json
import os

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# calculate bias change within a language
def compare_tfidf_within_language(language, input_folder, output_folder):
    """
    Compare TF-IDF scores for original, complex, and simple debiasing within a language.
    Compute percentage change from original for complex and simple debiased outputs.
    """
    input_file = os.path.join(input_folder, f"tfidf_analysis_{language}_mt0xxl_with_complex_and_simple_debiasing.json")
    output_file = os.path.join(output_folder, f"tfidf_comparison_within_{language}.json")

    tfidf_data = load_json(input_file)
    comparison_results = {}

    for identity, scores in tfidf_data.items():
        comparison_results[identity] = {"complex_change_from_original": {}, "simple_change_from_original": {}}

        for term, original_tfidf in scores["original"].items():
            complex_tfidf = scores["complex"].get(term, 0)
            simple_tfidf = scores["simple"].get(term, 0)

            # Compute percentage change relative to original TF-IDF
            complex_change = (((complex_tfidf - original_tfidf) / original_tfidf) * 100) if original_tfidf > 0 else 0
            simple_change = (((simple_tfidf - original_tfidf) / original_tfidf) * 100) if original_tfidf > 0 else 0


            comparison_results[identity]["complex_change_from_original"][term] = complex_change
            comparison_results[identity]["simple_change_from_original"][term] = simple_change

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=4, ensure_ascii=False)

    print(f"TF-IDF comparison within {language} saved to {output_file}")

# Run for all languages
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]
input_folder = "../../../data/lexicon_analysis/tfidf/tfidf_scores/"
output_folder = "../../../data/lexicon_analysis/tfidf/bias_change/bias_change_by_debiasing_method/"

for lang in languages:
    compare_tfidf_within_language(lang, input_folder, output_folder)


'''
### Step 2: Compute averages by identities within a language, then these averages across languages
'''
import numpy as np
def calculate_avg_change(language_data):
    comparison_results = {}

    # Iterate over identities
    for identity, scores in language_data.items():
        comparison_results[identity] = {"complex_avg_change": 0, "simple_avg_change": 0}

        complex_changes = []
        simple_changes = []

        # Collect bias changes for each term within the identity
        complex_changes.extend(scores["complex_change_from_original"].values())
        simple_changes.extend(scores["simple_change_from_original"].values())

        # Calculate average change for complex and simple
        comparison_results[identity]["complex_avg_change"] = np.mean(complex_changes)
        comparison_results[identity]["simple_avg_change"] = np.mean(simple_changes)

    return comparison_results

# Load and calculate average change for each language
language_results = {}

for lang in languages:
    input_file = os.path.join("../../../data/lexicon_analysis/tfidf/bias_change/bias_change_by_debiasing_method", f"tfidf_comparison_within_{lang}.json")
    language_data = load_json(input_file)
    
    # Calculate averages for the current language
    language_results[lang] = calculate_avg_change(language_data)

# Save individual language results (complex and simple average change per identity)
output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_identity.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(language_results, f, indent=4, ensure_ascii=False)

print(f"Results saved to {output_file}")


def calculate_avg_change_by_identity_across_languages(language_results):
    comparison_across_languages = {}

    # For each identity, calculate the average change across all languages
    for identity in language_results[languages[0]].keys():
        comparison_across_languages[identity] = {
            "complex_avg_change_from_original": 0,
            "simple_avg_change_from_original": 0
        }

        complex_changes = []
        simple_changes = []

        # Collect average change data across languages
        for lang in languages:
            complex_changes.append(language_results[lang][identity]["complex_avg_change"])
            simple_changes.append(language_results[lang][identity]["simple_avg_change"])

        # Calculate average change across languages
        comparison_across_languages[identity]["complex_avg_change_from_original"] = np.mean(complex_changes)
        comparison_across_languages[identity]["simple_avg_change_from_original"] = np.mean(simple_changes)

    return comparison_across_languages

# Calculate and save the result
comparison_across_languages = calculate_avg_change_by_identity_across_languages(language_results)

# Save final comparison across languages
final_output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_identity_ALL_languages.json"
with open(final_output_file, "w", encoding="utf-8") as f:
    json.dump(comparison_across_languages, f, indent=4, ensure_ascii=False)

print(f"Final comparison across languages saved to {final_output_file}")


def calculate_avg_change_by_method(language_results):
    """Compute overall average for complex and simple methods across all languages."""
    complex_changes = []
    simple_changes = []

    for lang in languages:
        lang_complex_avg = np.mean([language_results[lang][identity]["complex_avg_change"] for identity in language_results[lang]])
        lang_simple_avg = np.mean([language_results[lang][identity]["simple_avg_change"] for identity in language_results[lang]])

        complex_changes.append(lang_complex_avg)
        simple_changes.append(lang_simple_avg)

    # Store results
    avg_change_by_method = {}
    for i, lang in enumerate(languages):
        avg_change_by_method[lang] = {
            "complex_avg_change_from_original": complex_changes[i],
            "simple_avg_change_from_original": simple_changes[i]
        }

    return avg_change_by_method

# Compute and save overall averages by debiasing method
avg_change_by_method = calculate_avg_change_by_method(language_results)
method_output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_method_ALL_languages.json"

with open(method_output_file, "w", encoding="utf-8") as f:
    json.dump(avg_change_by_method, f, indent=4, ensure_ascii=False)

print(f"Overall average bias change across languages saved to {method_output_file}")
