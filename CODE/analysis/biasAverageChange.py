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
