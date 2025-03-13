'''
Calculate change for each identity,application and debiasing method

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
import os
from collections import defaultdict
from scipy.stats import ttest_rel, ttest_ind
from statistics import stdev, mean
import numpy as np

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# step 1: comupte bias change percentage
def compare_tfidf_within_language(language, input_folder, output_folder):
    """
    Compare TF-IDF scores for original, complex, and simple debiasing within a language.
    Compute percentage change from original for complex and simple debiased outputs.
    Store original, complex, and simple TF-IDF values while accounting for applications.
    """
    input_file = os.path.join(input_folder, f"tfidf_analysis_{language}_mt0xxl_with_complex_and_simple_debiasing.json")
    output_file = os.path.join(output_folder, f"tfidf_comparison_within_{language}.json")

    tfidf_data = load_json(input_file)
    comparison_results = {}

    for identity, applications in tfidf_data.items():
        comparison_results[identity] = {}

        for application, scores in applications.items():
            comparison_results[identity][application] = {
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
                comparison_results[identity][application]["complex_change_from_original"][term] = complex_change
                comparison_results[identity][application]["simple_change_from_original"][term] = simple_change
                comparison_results[identity][application]["original_tfidf"][term] = original_tfidf
                comparison_results[identity][application]["complex_tfidf"][term] = complex_tfidf
                comparison_results[identity][application]["simple_tfidf"][term] = simple_tfidf

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

def calculate_avg_change(language_data):
    """
    Compute average TF-IDF change for each identity and application, and store
    original, complex, and simple averages.
    """
    comparison_results = {}

    for identity, applications in language_data.items():
        comparison_results[identity] = {}
        for application, scores in applications.items():
            complex_changes = list(scores["complex_change_from_original"].values())
            simple_changes = list(scores["simple_change_from_original"].values())
            original_values = list(scores["original_tfidf"].values())
            complex_values = list(scores["complex_tfidf"].values())
            simple_values = list(scores["simple_tfidf"].values())

            # Handle the case where a list might be empty
            complex_avg_change = np.mean(complex_changes) if complex_changes else 0
            simple_avg_change = np.mean(simple_changes) if simple_changes else 0
            original_avg_tfidf = np.mean(original_values) if original_values else 0
            complex_avg_tfidf = np.mean(complex_values) if complex_values else 0
            simple_avg_tfidf = np.mean(simple_values) if simple_values else 0

            # Store averages for this identity and application
            comparison_results[identity][application] = {
                "complex_avg_change_from_original": complex_avg_change,
                "simple_avg_change_from_original": simple_avg_change,
                "original_avg_tfidf": original_avg_tfidf,
                "complex_avg_tfidf": complex_avg_tfidf,
                "simple_avg_tfidf": simple_avg_tfidf,
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

def calculate_avg_change_by_language(language_results):
    """Compute average TF-IDF change for complex and simple methods per language."""
    avg_change_by_language = {}

    # Loop through each language
    for lang in language_results:
        avg_change_by_language[lang] = {}  # Initialize a dictionary for each language
        
        # Loop through each identity in the language
        for identity in language_results[lang]:
            # Loop through each application type (Story, Hobbies and Values, To-do List)
            for application in language_results[lang][identity]:
                # Initialize lists to collect complex and simple changes for the current application
                complex_changes = []
                simple_changes = []
                
                # Append the complex and simple changes for the current application
                complex_changes.append(language_results[lang][identity][application]["complex_avg_change_from_original"])
                simple_changes.append(language_results[lang][identity][application]["simple_avg_change_from_original"])

                # Compute the average TF-IDF changes for the current application
                avg_change_by_language[lang][application] = {
                    "complex_avg_change_from_original": np.mean(complex_changes),
                    "simple_avg_change_from_original": np.mean(simple_changes)
                }

    return avg_change_by_language

# Calculate the average changes for each language
avg_change_by_language = calculate_avg_change_by_language(language_results)

# Save the per-language averages to a file
language_avg_output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_language.json"
with open(language_avg_output_file, "w", encoding="utf-8") as f:
    json.dump(avg_change_by_language, f, indent=4, ensure_ascii=False)

print(f"Per-language average bias change saved to {language_avg_output_file}")

def calculate_method_avg(avg_change_by_language):
    """Compute the method average of TF-IDF changes across all languages and identities."""
    method_avg = {}

    # Initialize lists for complex and simple changes across all languages and applications
    complex_changes = { "Story": [], "Hobbies and Values": [], "To-do List": [] }
    simple_changes = { "Story": [], "Hobbies and Values": [], "To-do List": [] }

    # Loop through all languages
    for lang, values in avg_change_by_language.items():
        # Loop through each application and collect changes
        for application in values:
            complex_changes[application].append(values[application]["complex_avg_change_from_original"])
            simple_changes[application].append(values[application]["simple_avg_change_from_original"])

    # Now compute the method averages for each application
    for application in complex_changes:
        method_avg[application] = {
            "complex_avg_change_from_original": np.mean(complex_changes[application]),
            "simple_avg_change_from_original": np.mean(simple_changes[application])
        }

    return method_avg

# Compute method averages using the per-language average results
method_avg = calculate_method_avg(avg_change_by_language)

# Save the method averages to a file
method_avg_output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_method.json"
with open(method_avg_output_file, "w", encoding="utf-8") as f:
    json.dump(method_avg, f, indent=4, ensure_ascii=False)

print(f"Method average bias change saved to {method_avg_output_file}")


'''
# Method average calculation
def calculate_method_avg(avg_change_by_language):
    """Compute the method average of TF-IDF changes across all languages and identities."""
    method_avg = {}

    # Loop through all languages
    for lang, values in avg_change_by_language.items():
        # Initialize the lists to store the changes for each language
        complex_changes = []
        simple_changes = []

        # Add the changes for this language (using the existing per-language averages)
        complex_changes.append(values["complex_avg_change_from_original"])
        simple_changes.append(values["simple_avg_change_from_original"])

        # Compute the method averages (complex and simple)
        method_avg[lang] = {
            "method_complex_avg_change_from_original": np.mean(complex_changes),
            "method_simple_avg_change_from_original": np.mean(simple_changes)
        }

    return method_avg

# Compute method averages using the per-language average results
method_avg = calculate_method_avg(avg_change_by_language)

# Save the method averages to a file
method_avg_output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_method.json"
with open(method_avg_output_file, "w", encoding="utf-8") as f:
    json.dump(method_avg, f, indent=4, ensure_ascii=False)

print(f"Method average bias change saved to {method_avg_output_file}")


from scipy.stats import zscore
from statistics import stdev, mean, StatisticsError

def normalize_bias_scores(language_results):
    """Normalize bias change scores across languages using min-max scaling."""
    all_values = []

    # Collect all complex and simple bias changes across languages
    for lang, identities in language_results.items():
        for identity, applications in identities.items():
            for application, values in applications.items():
                all_values.append(values["complex_avg_change"])
                all_values.append(values["simple_avg_change"])

    # Compute min and max
    min_val, max_val = min(all_values), max(all_values)

    # Apply min-max normalization
    normalized_results = {}
    for lang, identities in language_results.items():
        normalized_results[lang] = {}
        for identity, applications in identities.items():
            normalized_results[lang][identity] = {}
            for application, values in applications.items():
                complex_norm = (values["complex_avg_change"] - min_val) / (max_val - min_val) if max_val != min_val else 0
                simple_norm = (values["simple_avg_change"] - min_val) / (max_val - min_val) if max_val != min_val else 0

                normalized_results[lang][identity][application] = {
                    "complex_normalized": complex_norm,
                    "simple_normalized": simple_norm,
                }

    return normalized_results

# Apply normalization
normalized_data = normalize_bias_scores(language_results)
print(normalized_data)

from scipy.stats import ttest_rel
from statistics import stdev
import numpy as np

def compute_cohens_d(group1, group2):
    """Compute Cohen's d for paired samples."""
    diff = np.array(group1) - np.array(group2)
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0  # Avoid division by zero

def perform_t_tests_by_identity(language_results):
    """
    Perform paired t-tests comparing complex vs. simple debiasing for each identity and application within a language.
    """
    comparison_results = {}

    for lang, identities in language_results.items():
        comparison_results[lang] = {}

        for identity, applications in identities.items():
            comparison_results[lang][identity] = {}

            complex_scores = []
            simple_scores = []

            for application, values in applications.items():
                complex_scores.append(values["complex_avg_change"])
                simple_scores.append(values["simple_avg_change"])

            # Perform paired t-test if there are multiple samples
            if len(complex_scores) > 1 and len(simple_scores) > 1:
                t_stat, p_val = ttest_rel(complex_scores, simple_scores)
            else:
                t_stat, p_val = None, None  # Not enough data for a t-test

            comparison_results[lang][identity] = {
                "complex_avg": np.mean(complex_scores) if complex_scores else None,
                "simple_avg": np.mean(simple_scores) if simple_scores else None,
                "t_stat": t_stat,
                "p_value": p_val
            }

    return comparison_results

# Load input data for identity comparison
identity_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_identity.json"
language_results_identity = json.load(open(identity_file, "r", encoding="utf-8"))

# Perform identity-based comparisons
identity_comparison_results = perform_t_tests_by_identity(language_results_identity)

# Save results for identity-based comparisons
identity_output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/identity_comparison_results.json"
with open(identity_output_file, "w", encoding="utf-8") as f:
    json.dump(identity_comparison_results, f, indent=4, ensure_ascii=False)

print(f"Identity comparison results saved to {identity_output_file}")

def compare_identity_application_across_languages(language_results_identity, language_results_language, indo_aryan_languages, dravidian_languages):
    """
    Compare complex vs. simple debiasing across Indo-Aryan vs. Dravidian for each identity and application.
    """
    comparison_results = {"indo_aryan": {}, "dravidian": {}}

    for lang, identities in language_results_identity.items():
        family = "indo_aryan" if lang in indo_aryan_languages else "dravidian" if lang in dravidian_languages else None
        if not family:
            continue

        for identity, applications in identities.items():
            if identity not in comparison_results[family]:
                comparison_results[family][identity] = {}

            for application, values in applications.items():
                if application not in comparison_results[family][identity]:
                    comparison_results[family][identity][application] = {"complex_scores": [], "simple_scores": []}

                comparison_results[family][identity][application]["complex_scores"].append(values["complex_avg_change"])
                comparison_results[family][identity][application]["simple_scores"].append(values["simple_avg_change"])

    # Perform t-tests across Indo-Aryan and Dravidian for each identity/application
    final_comparison_results = {}

    for family in comparison_results:
        final_comparison_results[family] = {}
        for identity, applications in comparison_results[family].items():
            final_comparison_results[family][identity] = {}

            for application, scores in applications.items():
                complex_scores = scores["complex_scores"]
                simple_scores = scores["simple_scores"]

                # Perform paired t-test if sufficient data exists
                if len(complex_scores) > 1:
                    t_stat, p_val = ttest_rel(complex_scores, simple_scores)
                else:
                    t_stat, p_val = None, None  # Not enough data for t-test

                final_comparison_results[family][identity][application] = {
                    "complex_avg": np.mean(complex_scores) if complex_scores else None,
                    "simple_avg": np.mean(simple_scores) if simple_scores else None,
                    "t_stat": t_stat,
                    "p_value": p_val
                }

    return final_comparison_results

# Load input data for both identity and language-based comparison
identity_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_identity.json"
language_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_language.json"
language_results_identity = json.load(open(identity_file, "r", encoding="utf-8"))
language_results_language = json.load(open(language_file, "r", encoding="utf-8"))

# Define Indo-Aryan and Dravidian languages
indo_aryan_languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati"]
dravidian_languages = ["Malayalam", "Tamil", "Telugu", "Kannada"]

# Perform comparison
identity_application_comparison = compare_identity_application_across_languages(
    language_results_identity, language_results_language, indo_aryan_languages, dravidian_languages
)

# Save results for Indo-Aryan vs. Dravidian comparisons
output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/identity_application_comparison_across_languages.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(identity_application_comparison, f, indent=4, ensure_ascii=False)

print(f"Indo-Aryan vs. Dravidian identity/application comparison saved to {output_file}")

### ANOVA

from scipy import stats

def perform_anova_by_identity(language_results):
    """
    Perform ANOVA comparing complex vs. simple debiasing for each identity and application within a language.
    """
    comparison_results = {}

    for lang, identities in language_results.items():
        comparison_results[lang] = {}

        for identity, applications in identities.items():
            comparison_results[lang][identity] = {}

            complex_scores = []
            simple_scores = []

            for application, values in applications.items():
                complex_scores.append(values["complex_avg_change"])
                simple_scores.append(values["simple_avg_change"])

            # Perform one-way ANOVA across debiasing methods (complex vs. simple)
            if len(complex_scores) > 1 and len(simple_scores) > 1:
                f_stat, p_val = stats.f_oneway(complex_scores, simple_scores)
            else:
                f_stat, p_val = None, None  # Not enough data for ANOVA

            comparison_results[lang][identity] = {
                "complex_avg": np.mean(complex_scores) if complex_scores else None,
                "simple_avg": np.mean(simple_scores) if simple_scores else None,
                "f_stat": f_stat,
                "p_value": p_val
            }

    return comparison_results

# Load input data for identity comparison
identity_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_identity.json"
language_results_identity = json.load(open(identity_file, "r", encoding="utf-8"))

# Perform identity-based ANOVA comparisons
identity_comparison_results = perform_anova_by_identity(language_results_identity)

# Save results for identity-based comparisons
identity_output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/identity_comparison_results_anova.json"
with open(identity_output_file, "w", encoding="utf-8") as f:
    json.dump(identity_comparison_results, f, indent=4, ensure_ascii=False)

print(f"Identity comparison results saved to {identity_output_file}")


def compare_identity_application_across_languages_anova(language_results_identity, indo_aryan_languages, dravidian_languages):
    """
    Compare complex vs. simple debiasing across Indo-Aryan vs. Dravidian for each identity and application using ANOVA.
    """
    comparison_results = {"indo_aryan": {}, "dravidian": {}}

    for lang, identities in language_results_identity.items():
        family = "indo_aryan" if lang in indo_aryan_languages else "dravidian" if lang in dravidian_languages else None
        if not family:
            continue

        for identity, applications in identities.items():
            if identity not in comparison_results[family]:
                comparison_results[family][identity] = {}

            for application, values in applications.items():
                if application not in comparison_results[family][identity]:
                    comparison_results[family][identity][application] = {"complex_scores": [], "simple_scores": []}

                comparison_results[family][identity][application]["complex_scores"].append(values["complex_avg_change"])
                comparison_results[family][identity][application]["simple_scores"].append(values["simple_avg_change"])

    # Perform ANOVA for Indo-Aryan and Dravidian for each identity/application
    final_comparison_results = {}

    for family in comparison_results:
        final_comparison_results[family] = {}
        for identity, applications in comparison_results[family].items():
            final_comparison_results[family][identity] = {}

            for application, scores in applications.items():
                complex_scores = scores["complex_scores"]
                simple_scores = scores["simple_scores"]

                # Perform one-way ANOVA across debiasing methods (complex vs. simple)
                if len(complex_scores) > 1 and len(simple_scores) > 1:
                    f_stat, p_val = stats.f_oneway(complex_scores, simple_scores)
                else:
                    f_stat, p_val = None, None  # Not enough data for ANOVA

                final_comparison_results[family][identity][application] = {
                    "complex_avg": np.mean(complex_scores) if complex_scores else None,
                    "simple_avg": np.mean(simple_scores) if simple_scores else None,
                    "f_stat": f_stat,
                    "p_value": p_val
                }

    return final_comparison_results

# Load input data for both identity and language-based comparison
identity_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/average_bias_change_by_identity.json"
language_results_identity = json.load(open(identity_file, "r", encoding="utf-8"))

# Define Indo-Aryan and Dravidian languages
indo_aryan_languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati"]
dravidian_languages = ["Malayalam", "Tamil", "Telugu", "Kannada"]

# Perform comparison
identity_application_comparison = compare_identity_application_across_languages_anova(
    language_results_identity, indo_aryan_languages, dravidian_languages
)

# Save results for Indo-Aryan vs. Dravidian comparisons
output_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/identity_application_comparison_across_languages_anova.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(identity_application_comparison, f, indent=4, ensure_ascii=False)

print(f"Indo-Aryan vs. Dravidian identity/application comparison saved to {output_file}")


import matplotlib.pyplot as plt
import seaborn as sns
def aggregate_results(comparison_results):
    """
    Aggregate the results of ANOVA for easy interpretation 
    """
    aggregated_results = {
        "identity": [],
        "application": [],
        "complex_avg": [],
        "simple_avg": [],
        "f_stat": [],
        "p_value": [],
        "effect_size": [],
        "significance": []
    }

    for language, identities in comparison_results.items():
        for identity, applications in identities.items():
            for application, stats in applications.items():
                # Calculate effect size (Cohen's d or eta squared)
                if stats["f_stat"] is not None and stats["p_value"] is not None:
                    # Cohen's d approximation (for simplicity, assuming equal variance)
                    effect_size = np.sqrt(stats["f_stat"] / (stats["f_stat"] + (len(stats.get("complex_scores", [])) - 1)))
                    significance = "Significant" if stats["p_value"] < 0.05 else "Not Significant"
                else:
                    effect_size = None
                    significance = None

                # Collecting results
                aggregated_results["identity"].append(identity)
                aggregated_results["application"].append(application)
                aggregated_results["complex_avg"].append(stats["complex_avg"])
                aggregated_results["simple_avg"].append(stats["simple_avg"])
                aggregated_results["f_stat"].append(stats["f_stat"])
                aggregated_results["p_value"].append(stats["p_value"])
                aggregated_results["effect_size"].append(effect_size)
                aggregated_results["significance"].append(significance)
    
    return aggregated_results

# Load the ANOVA results from the previous step
identity_comparison_results_anova_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/identity_comparison_results_anova.json"
identity_comparison_results_anova = json.load(open(identity_comparison_results_anova_file, "r", encoding="utf-8"))

# Aggregate the results for easy interpretation
aggregated_results = aggregate_results(identity_comparison_results_anova)

# Save the aggregated results to a JSON file for further analysis
aggregated_results_file = "../../../data/lexicon_analysis/tfidf/bias_change/avg_bias_change_by_debiasing_method/aggregated_identity_comparison_results.json"
with open(aggregated_results_file, "w", encoding="utf-8") as f:
    json.dump(aggregated_results, f, indent=4, ensure_ascii=False)

print(f"Aggregated results saved to {aggregated_results_file}")

# Create a dictionary for heatmap data (F-statistics for each identity/application)
heatmap_data = {}

for language, identities in aggregated_results.items():
    if language == "identity" or language == "application":
        continue
    for i, identity_name in enumerate(aggregated_results["identity"]):
        for j, application_name in enumerate(aggregated_results["application"]):
            if identity_name not in heatmap_data:
                heatmap_data[identity_name] = {}
            heatmap_data[identity_name][application_name] = aggregated_results["f_stat"][i]

# Now plot the heatmap
plt.figure(figsize=(10, 8))

# Prepare the heatmap data for plotting
heatmap_matrix = []
for identity_name in heatmap_data:
    row = [heatmap_data[identity_name].get(application, 0) for application in aggregated_results["application"]]
    heatmap_matrix.append(row)

# Create the heatmap with seaborn
sns.heatmap(heatmap_matrix, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=aggregated_results["application"], yticklabels=aggregated_results["identity"], linewidths=0.5)

plt.title("F-statistics for ANOVA (Complex vs. Simple Debiasing)")
plt.ylabel("Identity")
plt.xlabel("Application")
plt.tight_layout()
plt.show()
'''