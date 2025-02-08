'''
Negative values: This still indicates a reduction in bias (debiasing method results in fewer occurrences of bias terms than the original).

Positive values: This indicates an increase in bias (debiasing method results in more occurrences of bias terms than the original).
'''


import json
from collections import defaultdict

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_average_bias_reduction(bias_change_results):
    """Compute the average bias reduction for each identity and debiasing method."""
    identity_avg_reduction = {}

    # Loop through each identity in the bias change results
    for identity, changes in bias_change_results.items():
        complex_changes = changes.get("complex_change_from_original", {})
        simple_changes = changes.get("simple_change_from_original", {})

        avg_complex_reduction = sum(complex_changes.values()) / len(complex_changes) if complex_changes else 0
        avg_simple_reduction = sum(simple_changes.values()) / len(simple_changes) if simple_changes else 0

        identity_avg_reduction[identity] = {
            "complex_avg_reduction": avg_complex_reduction,
            "simple_avg_reduction": avg_simple_reduction
        }

    return identity_avg_reduction

def save_json(data, filepath):
    """Save JSON data to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Languages to process
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

# Dictionary to store average reductions for each language and identity
language_avg_reductions = {}

# Process each language
for lang in languages:
    file_path = f"../../data/lexicon_analysis/bias_change/bias_change_by_debiasing_method/bias_change_{lang}.json"
    
    # Load bias change results for the current language
    with open(file_path, "r", encoding="utf-8") as f:
        bias_change_results = json.load(f)

    # Compute the average bias reduction for each identity (per debiasing method)
    identity_avg_reduction = compute_average_bias_reduction(bias_change_results)

    # Store the results for the current language, broken down by identity
    language_avg_reductions[lang] = identity_avg_reduction

    print(f"Average bias reduction for {lang} - identity breakdown saved.")

# Save the results to a JSON file
save_path = f"../../data/lexicon_analysis/bias_change/avg_bias_change_by_debiasing_method/average_bias_reduction_by_identity.json"
save_json(language_avg_reductions, save_path)

print("Average bias reduction for each identity group, broken down by debiasing method, saved successfully.")

#---

def compute_average_by_language(bias_reduction_by_identity):
    """Compute the average bias reduction (complex and simple) for each language."""
    language_avg = {}

    for language, identity_data in bias_reduction_by_identity.items():
        avg_complex = 0
        avg_simple = 0
        total_identities = len(identity_data)

        # Compute averages for complex and simple reductions across all identities
        for identity, reductions in identity_data.items():
            avg_complex += reductions.get("complex_avg_reduction", 0)
            avg_simple += reductions.get("simple_avg_reduction", 0)

        # Calculate average for complex and simple
        if total_identities > 0:
            avg_complex /= total_identities
            avg_simple /= total_identities

        # Store average reduction for the language
        language_avg[language] = {
            "complex_avg_reduction_from_original": avg_complex,
            "simple_avg_reduction_from_original": avg_simple
        }

    return language_avg

def save_json(data, filepath):
    """Save JSON data to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def calculate_and_save_language_averages(input_file, output_file):
    """Load the bias change results by identity, compute the average per language, and save the result."""
    # Load the bias reduction results by identity from the provided input file
    bias_reduction_by_identity = load_json(input_file)

    # Compute the average by language for complex and simple bias reductions
    language_avg = compute_average_by_language(bias_reduction_by_identity)

    # Save the calculated averages to the output file
    save_json(language_avg, output_file)
    print(f"Average bias reductions by language saved to {output_file}")

# Example of calling the function
input_file = "../../data/lexicon_analysis/bias_change/avg_bias_change_by_debiasing_method/average_bias_reduction_by_identity.json"
output_file = "../../data/lexicon_analysis/bias_change/avg_bias_change_by_debiasing_method/average_bias_reduction_by_identity_language.json"
calculate_and_save_language_averages(input_file, output_file)