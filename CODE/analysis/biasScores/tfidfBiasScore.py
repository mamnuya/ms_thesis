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

def print_high_low_scores():
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

        print(f" - TF IDF Bias Score - ")
        print(f"Language: {lang}")
        print(f"  Top Complex Bias Score: {top_complex_name} -> {top_complex_values['complex_bias_score']:.6f}")
        print(f"  Top Simple Bias Score: {top_simple_name} -> {top_simple_values['simple_bias_score']:.6f}")
        print(f"  Lowest Complex Bias Score: {lowest_complex_name} -> {lowest_complex_values['complex_bias_score']:.6f}")
        print(f"  Lowest Simple Bias Score: {lowest_simple_name} -> {lowest_simple_values['simple_bias_score']:.6f}")
        print("-" * 50)
        
print_high_low_scores()

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


def generate_latex_thresholds_table(language_avg_change, method_avg_change):
    """
    Generate a LaTeX table for the thresholds used to normalize TF-IDF bias scores.
    """
    # Extract the normalization factors
    method_complex_threshold = method_avg_change["method_complex_avg_change_from_original"]
    method_simple_threshold = method_avg_change["method_simple_avg_change_from_original"]

    # Start LaTeX table
    latex_table = "\\begin{table}[h]\n"
    latex_table += "    \\centering\n"
    latex_table += "    \\caption{Normalization Factors for TF-IDF Bias Scores}\n"
    latex_table += "    \\label{tab:normalization_avgs_tfidf}\n"
    latex_table += "    \\begin{tabular}{|l|c|c|}\n"
    latex_table += "        \\hline\n"
    latex_table += "        \\textbf{Language} & \\textbf{Complex Average} & \\textbf{Simple Average} \\\\\n"
    latex_table += "        \\hline\n"

    for lang, values in language_avg_change.items():
        complex_threshold = values["complex_avg_change_from_original"]
        simple_threshold = values["simple_avg_change_from_original"]
        latex_table += f"        {lang} & {complex_threshold:.6f} & {simple_threshold:.6f} \\\\\n"

    # Add method-wide thresholds
    latex_table += "        \\hline\n"
    latex_table += f"        \\textbf{{Method-Wise Averages}} & {method_complex_threshold:.6f} & {method_simple_threshold:.6f} \\\\\n"
    latex_table += "        \\hline\n"
    
    # End table
    latex_table += "    \\end{tabular}\n"
    latex_table += "\\end{table}"

    # Print the LaTeX table
    print(latex_table)

# Call the function to print the LaTeX table
#generate_latex_thresholds_table(language_avg_change, method_avg_change)
