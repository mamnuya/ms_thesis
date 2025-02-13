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

# Step 1: Calculate bias change within a language
def compare_tfidf_within_language(language, input_folder, output_folder):
    """
    Compare TF-IDF scores for original, complex, and simple debiasing within a language.
    Compute percentage change from original for complex and simple debiased outputs.
    Store original, complex, and simple TF-IDF values.
    """
    input_file = os.path.join(input_folder, f"tfidf_analysis_{language}_mt0xxl_with_complex_and_simple_debiasing.json")
    output_file = os.path.join(output_folder, f"tfidf_comparison_within_{language}.json")

    tfidf_data = load_json(input_file)
    comparison_results = {}

    for identity, scores in tfidf_data.items():
        comparison_results[identity] = {
            "complex_change_from_original": {},
            "simple_change_from_original": {},
            "original_tfidf": {},
            "complex_tfidf": {},
            "simple_tfidf": {}
        }

        for term, original_tfidf in scores["original"].items():
            complex_tfidf = scores["complex"].get(term, 0)
            simple_tfidf = scores["simple"].get(term, 0)

            # Compute percentage change relative to original TF-IDF
            complex_change = (((complex_tfidf - original_tfidf) / original_tfidf) * 100) if original_tfidf > 0 else 0
            simple_change = (((simple_tfidf - original_tfidf) / original_tfidf) * 100) if original_tfidf > 0 else 0

            # Store values
            comparison_results[identity]["complex_change_from_original"][term] = complex_change
            comparison_results[identity]["simple_change_from_original"][term] = simple_change
            comparison_results[identity]["original_tfidf"][term] = original_tfidf
            comparison_results[identity]["complex_tfidf"][term] = complex_tfidf
            comparison_results[identity]["simple_tfidf"][term] = simple_tfidf

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
    """Compute average TF-IDF change and store original, complex, and simple averages."""
    comparison_results = {}

    for identity, scores in language_data.items():
        complex_changes = list(scores["complex_change_from_original"].values())
        simple_changes = list(scores["simple_change_from_original"].values())

        original_values = list(scores["original_tfidf"].values())
        complex_values = list(scores["complex_tfidf"].values())
        simple_values = list(scores["simple_tfidf"].values())

        # Store averages
        comparison_results[identity] = {
            "complex_avg_change": np.mean(complex_changes),
            "simple_avg_change": np.mean(simple_changes),
            "original_avg_tfidf": np.mean(original_values),
            "complex_avg_tfidf": np.mean(complex_values),
            "simple_avg_tfidf": np.mean(simple_values)
        }

    return comparison_results

# Load and calculate average change for each language
language_results = {}

for lang in languages:
    input_file = os.path.join(output_folder, f"tfidf_comparison_within_{lang}.json")
    language_data = load_json(input_file)
    
    # Calculate averages for the current language
    language_results[lang] = calculate_avg_change(language_data)

# Save individual language results
output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_identity.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(language_results, f, indent=4, ensure_ascii=False)

print(f"Results saved to {output_file}")


# Step 3: Compute averages across languages
def calculate_avg_change_by_identity_across_languages(language_results):
    """Compute the average TF-IDF change across all languages for each identity."""
    comparison_across_languages = {}

    for identity in language_results[languages[0]].keys():
        complex_changes = []
        simple_changes = []
        original_values = []
        complex_values = []
        simple_values = []

        # Collect average change data across languages
        for lang in languages:
            complex_changes.append(language_results[lang][identity]["complex_avg_change"])
            simple_changes.append(language_results[lang][identity]["simple_avg_change"])
            original_values.append(language_results[lang][identity]["original_avg_tfidf"])
            complex_values.append(language_results[lang][identity]["complex_avg_tfidf"])
            simple_values.append(language_results[lang][identity]["simple_avg_tfidf"])

        # Compute and store averages
        comparison_across_languages[identity] = {
            "complex_avg_change_from_original": np.mean(complex_changes),
            "simple_avg_change_from_original": np.mean(simple_changes),
            "original_avg_tfidf": np.mean(original_values),
            "complex_avg_tfidf": np.mean(complex_values),
            "simple_avg_tfidf": np.mean(simple_values)
        }

    return comparison_across_languages

# Compute and save final comparison across languages
comparison_across_languages = calculate_avg_change_by_identity_across_languages(language_results)
final_output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_identity_ALL_languages.json"

with open(final_output_file, "w", encoding="utf-8") as f:
    json.dump(comparison_across_languages, f, indent=4, ensure_ascii=False)

print(f"Final comparison across languages saved to {final_output_file}")

# Step 4: Compute overall averages by method across languages
def calculate_avg_change_by_method(language_results):
    """Compute overall average TF-IDF change and values for complex and simple methods."""
    complex_changes = []
    simple_changes = []
    original_values = []
    complex_values = []
    simple_values = []

    for lang in languages:
        for identity in language_results[lang]:
            complex_changes.append(language_results[lang][identity]["complex_avg_change"])
            simple_changes.append(language_results[lang][identity]["simple_avg_change"])
            original_values.append(language_results[lang][identity]["original_avg_tfidf"])
            complex_values.append(language_results[lang][identity]["complex_avg_tfidf"])
            simple_values.append(language_results[lang][identity]["simple_avg_tfidf"])

    # Store overall averages
    avg_change_by_method = {
        "complex_avg_change_from_original": np.mean(complex_changes),
        "simple_avg_change_from_original": np.mean(simple_changes),
        "original_avg_tfidf": np.mean(original_values),
        "complex_avg_tfidf": np.mean(complex_values),
        "simple_avg_tfidf": np.mean(simple_values)
    }

    return avg_change_by_method

# Compute and save overall averages
avg_change_by_method = calculate_avg_change_by_method(language_results)
method_output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_method_ALL_languages.json"

with open(method_output_file, "w", encoding="utf-8") as f:
    json.dump(avg_change_by_method, f, indent=4, ensure_ascii=False)

print(f"Overall average bias change across languages saved to {method_output_file}")