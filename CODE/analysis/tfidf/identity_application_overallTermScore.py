"""
This script loads precomputed TF-IDF values for terms extracted from language model outputs 
across multiple South Asian languages. 

It identifies the highest TF-IDF scoring term for each 
application (e.g., 'Story', 'Hobbies and Values', 'To-do List') and identity group 
(e.g., 'Muslim Male who is Married with No children') under the 'original' prompting method 
(i.e., before debiasing). 

Generates LaTeX-formatted tables summarizing these top terms 
to visually represent dominant associations and potential cultural or gendered biases in model outputs.

`get_top_tfidf_per_application_identity(...)`:
   For each language and identity group, extracts the term with the highest TF-IDF score 
   (from the 'original' version) for each application.

`generate_latex_tables_by_application(...)`:
   Generates a series of LaTeX tables showing top TF-IDF terms organized by:
   - Religion & Gender
   - Marital Status
   - Number of Children
   Each cell is color-coded by z-score thresholding:
     - Red: High TF-IDF (above mean + 1 SD)
     - Yellow: Medium TF-IDF (within 1 SD of mean)
     - Green: Low TF-IDF (below mean - 1 SD)


This script supports qualitative evaluation of how identity-related biases manifest in generated 
text and how prominent those associations are before any debiasing is applied. The LaTeX tables 
can be used in publications or appendices to present language-specific findings in a structured, 
interpretable format.

"""

from collections import defaultdict
import numpy as np
import json

def save_json(data, filepath):
    """Save JSON data to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)





def get_top_tfidf_per_application_identity(tf_idf_scores_all_languages_by_application, identities_to_analyze):
    """
    Extracts the highest TF-IDF term per application first, then per identity group, ensuring 
    that within each application, only the most prominent term per identity is kept.
    """
    results = {}

    for lang, tf_idf_scores in tf_idf_scores_all_languages_by_application.items():
        application_top_terms = {}

        for identity, applications in tf_idf_scores.items():
            for base_identity in identities_to_analyze:
                if base_identity in identity:  # Match only relevant identities
                    for application, methods in applications.items():
                        if application not in application_top_terms:
                            application_top_terms[application] = {}

                        if "original" in methods:
                            sorted_terms = sorted(methods["original"].items(), key=lambda x: x[1], reverse=True)
                            if sorted_terms:
                                top_term, top_value = sorted_terms[0]

                                # Store only the highest TF-IDF term per identity within each application
                                if identity not in application_top_terms[application] or top_value > application_top_terms[application][identity]["tfidf_value"]:
                                    application_top_terms[application][identity] = {
                                        "language": lang,
                                        "identity": identity,
                                        "application": application,
                                        "term": top_term,
                                        "tfidf_value": top_value
                                    }

        results[lang] = application_top_terms  # Store results per language

    return results


# Store term scores per language
language_term_scores = {}
tf_idf_scores_all_languages = {}  # Store all TF-IDF scores for top-term analysis

languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

for lang in languages:
    tfidf_path = f"../../../data/lexicon_analysis/tfidf/tfidf_values/allTerms/tfidf_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json"
    
    # Load precomputed TF-IDF scores
    tf_idf_scores = load_json(tfidf_path)

    tf_idf_scores_all_languages[lang] = tf_idf_scores  # Store TF-IDF for top-term analysis

    print(f"Term scores saved successfully for {lang}.")

# Define the base identities we want to analyze
identities_to_analyze = ["Muslim Male", "Muslim Female", "Hindu Male", "Hindu Female"]

# Run the function
top_tfidf_per_identity_group_and_application = get_top_tfidf_per_application_identity(tf_idf_scores_all_languages, identities_to_analyze)
def generate_latex_tables_by_application(top_tfidf_per_identity_group_and_application):
    """
    Generates LaTeX tables for the highest TF-IDF terms per application, then per identity.
    Each application gets a separate table.
    """

    application_order = ["Story", "Hobbies and Values", "To-do List"]  # Enforce order

    for lang, application_data in top_tfidf_per_identity_group_and_application.items():
        if lang != "NONE":  # Only process certain lang
            
            # Iterate over applications in the enforced order
            for application in application_order:
                if application not in application_data:
                    continue  # Skip applications that don't exist in the data
                
                identity_data = application_data[application]
                
                print(f"\n\\section{{Top Overall Terms for {lang} - {application}}}")

                # Calculate mean and standard deviation for TF-IDF values
                tfidf_values = [entry["tfidf_value"] for entry in identity_data.values()]
                mean_tfidf = sum(tfidf_values) / len(tfidf_values)
                std_dev_tfidf = (sum((x - mean_tfidf) ** 2 for x in tfidf_values) / len(tfidf_values)) ** 0.5

                # Thresholds for color coding
                lower_threshold = mean_tfidf - std_dev_tfidf
                upper_threshold = mean_tfidf + std_dev_tfidf

                print(f"\\newpage")
                print(f"\\begin{{table}}[h!]")
                print(f"\\centering")
                print(f"\\caption{{Highest Overall TF-IDF terms and values for all identities under the application \\textbf{{{application}}} in {lang} outputs under the original prompting method.}}")
                print(f"\\scriptsize")
                print(f"\\label{{tab:tfidf_comparison_identity_overall_{lang}_{application.replace(' ', '_')}}}")

                # Adjusted table structure with 'Application' moved to the end
                print(f"\\begin{{tabular}}{{|p{{1.9cm}}|p{{1.3cm}}|l|l|p{{0.7cm}}|}}")
                print(f"\\hline")
                print(f"\\textbf{{Religion \\& Gender}} & \\textbf{{Marital Status}} & \\textbf{{Children}} & \\textbf{{Term}} & \\textbf{{Over-all TF-IDF}} \\\\")
                print(f"\\hline")

                # Dictionary to track multirow formatting
                identity_grouped_entries = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

                for entry in identity_data.values():
                    identity = entry["identity"]
                    term = entry["term"]
                    tfidf_value = entry["tfidf_value"]

                    # Determine color based on threshold
                    if tfidf_value > upper_threshold:
                        color = "\\cellcolor{red!30}"
                    elif tfidf_value < lower_threshold:
                        color = "\\cellcolor{green!30}"
                    else:
                        color = "\\cellcolor{yellow!30}"

                    # Extract key identity components
                    parts = identity.split("who is")  # Split identity into base (religion & gender) and marital/children details
                    main_identity = parts[0].strip().replace("A ","")  # "A Hindu Male"
                    details = parts[1].strip() if len(parts) > 1 else ""  # "Married with No children"
                    
                    # Extract marital status and children count
                    marital_status = "Single"  # Default
                    if "Married" in details:
                        marital_status = "Married"
                    elif "Divorced" in details:
                        marital_status = "Divorced"
                    elif "Widowed" in details:
                        marital_status = "Widowed"

                    children = "No children"  # Default
                    if "One child" in details:
                        children = "One child"
                    elif "Many children" in details:
                        children = "Many children"

                    # Store in dictionary for hierarchical LaTeX formatting
                    identity_grouped_entries[main_identity][marital_status][children].append((term, tfidf_value, color))

                # Sort identities alphabetically to ensure consistent order
                sorted_main_identities = sorted(identity_grouped_entries.keys())

                # Define consistent sorting order for marital status and children
                marital_status_order = ["Single", "Married", "Divorced", "Widowed"]
                children_order = ["No children", "One child", "Many children"]

                for main_identity in sorted_main_identities:
                    marital_groups = identity_grouped_entries[main_identity]

                    # Sort marital status and children groups to ensure consistent order
                    sorted_marital_statuses = sorted(marital_groups.keys(), key=lambda x: marital_status_order.index(x))
                    
                    num_rows_identity = sum(len(children_list) for children_list in marital_groups.values())  # Total rows for religion & gender
                    first_identity = True  # Track first row for \multirow

                    print(f"\\hline")  # Horizontal line before a new Religion & Gender group

                    for marital_status in sorted_marital_statuses:
                        children_list = marital_groups[marital_status]
                        num_rows_marital = len(children_list)  # Rows for this marital status
                        first_marital = True  # Track first row for \multirow in marital status

                        # Sort children and entries to ensure consistent order
                        sorted_children_list = sorted(children_list.items(), key=lambda x: children_order.index(x[0]))  # Sort by children category

                        for idx, (children, entries) in enumerate(sorted_children_list):
                            if first_identity:
                                print(f"\\multirow{{{num_rows_identity}}}{{*}}{{{main_identity}}} ", end="")
                                first_identity = False
                            else:
                                print(" ", end="")  # No multirow, just align

                            if first_marital:
                                print(f"& \\multirow{{{num_rows_marital}}}{{*}}{{{marital_status}}} ", end="")
                                first_marital = False
                            else:
                                print("& ", end="")  # No multirow, just align

                            # Print the row for children, term, and tfidf value
                            for term, tfidf_value, color in entries:
                                print(f"& {children} & {term} & {color}{tfidf_value:.3f} \\\\")

                            # Add horizontal lines after each Children category
                            if idx < len(sorted_children_list) - 1:
                                print(f"\\cline{{3-5}}")  # Separates Children, Term, TF-IDF

                        print(f"\\cline{{2-5}}")  # Horizontal line after all Children under a Marital Status

                    print(f"\\hline")  # Horizontal line after all Marital Status under a Religion & Gender

                print(f"\\end{{tabular}}")
                print(f"\\end{{table}}")
                print(f"\\newpage")
                print()  
generate_latex_tables_by_application(top_tfidf_per_identity_group_and_application)

print("Overall term extraction and LaTeX formatting completed.")

