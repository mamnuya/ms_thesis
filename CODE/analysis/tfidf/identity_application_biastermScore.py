'''
Bias Scores per Identity, Application, and Language:---
Each (identity, application) in a language gets a bias score.
Higher scores indicate stronger bias-related term presence.
Debiasing effectiveness can be measured by comparing "original" vs. "complex" vs. "simple".

'''


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


def compute_bias_score(tf_idf_scores):
    """Compute the summation of tf_idfs to create bias scores for each (identity, application, method) pair."""
    bias_scores = defaultdict(lambda: defaultdict(lambda: {"original": 0, "complex": 0, "simple": 0}))

    for identity, applications in tf_idf_scores.items():
        for application, methods in applications.items():
            for method in ["original", "complex", "simple"]:
                term_values = list(methods[method].values())
                bias_scores[identity][application][method] = sum(term_values) # compute sum

    return bias_scores


import numpy as np




def get_top_tfidf_per_application_identity(tf_idf_scores_all_languages_by_application, identities_to_analyze):
    """
    Extracts the highest TF-IDF term in original outputs per application first, then per identity group, ensuring 
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

def calculate_averaged_scores_per_language(language_bias_scores, indo_aryan_languages, dravidian_languages):
    """Calculate averaged bias scores for each application within a language, then compute final aggregate averages per method,
    for Indo-Aryan, Dravidian, and All_Languages."""

    # Data structures to store scores before averaging
    indo_aryan_scores = {
        app: {
            "avg_religion": {"Hindu_original": [], "Muslim_original": [], "Hindu_simple": [], "Muslim_simple": [],
                             "Hindu_complex": [], "Muslim_complex": []},
            "avg_gender": {"Male_original": [], "Female_original": [], "Male_simple": [], "Female_simple": [],
                           "Male_complex": [], "Female_complex": []},
            "avg_marital_status": {"Single_original": [], "Married_original": [], "Divorced_original": [], "Widowed_original": [],
                                   "Single_simple": [], "Married_simple": [], "Divorced_simple": [], "Widowed_simple": [],
                                   "Single_complex": [], "Married_complex": [], "Divorced_complex": [], "Widowed_complex": []},
            "avg_child_count": {"No children_original": [], "One child_original": [], "Many children_original": [],
                                   "No children_simple": [], "One child_simple": [], "Many children_simple": [],
                                   "No children_complex": [], "One child_complex": [], "Many children_complex": []},
            "aggregate_original_application": [],
            "aggregate_simple_application": [],
            "aggregate_complex_application": []
        }
        for app in ["Story", "Hobbies and Values", "To-do List"]
    }

    dravidian_scores = {
        app: {
            "avg_religion": {"Hindu_original": [], "Muslim_original": [], "Hindu_simple": [], "Muslim_simple": [],
                             "Hindu_complex": [], "Muslim_complex": []},
            "avg_gender": {"Male_original": [], "Female_original": [], "Male_simple": [], "Female_simple": [],
                           "Male_complex": [], "Female_complex": []},
            "avg_marital_status": {"Single_original": [], "Married_original": [], "Divorced_original": [], "Widowed_original": [],
                                   "Single_simple": [], "Married_simple": [], "Divorced_simple": [], "Widowed_simple": [],
                                   "Single_complex": [], "Married_complex": [], "Divorced_complex": [], "Widowed_complex": []},
            "avg_child_count": {"No children_original": [], "One child_original": [], "Many children_original": [],
                                   "No children_simple": [], "One child_simple": [], "Many children_simple": [],
                                   "No children_complex": [], "One child_complex": [], "Many children_complex": []},
            "aggregate_original_application": [],
            "aggregate_simple_application": [],
            "aggregate_complex_application": []
        }
        for app in ["Story", "Hobbies and Values", "To-do List"]
    }

    all_languages_scores = {
        app: {
            "avg_religion": {"Hindu_original": [], "Muslim_original": [], "Hindu_simple": [], "Muslim_simple": [],
                             "Hindu_complex": [], "Muslim_complex": []},
            "avg_gender": {"Male_original": [], "Female_original": [], "Male_simple": [], "Female_simple": [],
                           "Male_complex": [], "Female_complex": []},
            "avg_marital_status": {"Single_original": [], "Married_original": [], "Divorced_original": [], "Widowed_original": [],
                                   "Single_simple": [], "Married_simple": [], "Divorced_simple": [], "Widowed_simple": [],
                                   "Single_complex": [], "Married_complex": [], "Divorced_complex": [], "Widowed_complex": []},
            "avg_child_count": {"No children_original": [], "One child_original": [], "Many children_original": [],
                                   "No children_simple": [], "One child_simple": [], "Many children_simple": [],
                                   "No children_complex": [], "One child_complex": [], "Many children_complex": []},
            "aggregate_original_application": [],
            "aggregate_simple_application": [],
            "aggregate_complex_application": []
        }
        for app in ["Story", "Hobbies and Values", "To-do List"]
    }

    for lang, bias_scores in language_bias_scores.items():
        for identity, applications in bias_scores.items():
            for application, methods in applications.items():

                # Religion scores
                if "Hindu" in identity:
                    all_languages_scores[application]["avg_religion"]["Hindu_original"].append(methods["original"])
                    all_languages_scores[application]["avg_religion"]["Hindu_simple"].append(methods["simple"])
                    all_languages_scores[application]["avg_religion"]["Hindu_complex"].append(methods["complex"])
                    if lang in indo_aryan_languages:
                        indo_aryan_scores[application]["avg_religion"]["Hindu_original"].append(methods["original"])
                        indo_aryan_scores[application]["avg_religion"]["Hindu_simple"].append(methods["simple"])
                        indo_aryan_scores[application]["avg_religion"]["Hindu_complex"].append(methods["complex"])
                    elif lang in dravidian_languages:
                        dravidian_scores[application]["avg_religion"]["Hindu_original"].append(methods["original"])
                        dravidian_scores[application]["avg_religion"]["Hindu_simple"].append(methods["simple"])
                        dravidian_scores[application]["avg_religion"]["Hindu_complex"].append(methods["complex"])

                elif "Muslim" in identity:
                    all_languages_scores[application]["avg_religion"]["Muslim_original"].append(methods["original"])
                    all_languages_scores[application]["avg_religion"]["Muslim_simple"].append(methods["simple"])
                    all_languages_scores[application]["avg_religion"]["Muslim_complex"].append(methods["complex"])
                    if lang in indo_aryan_languages:
                        indo_aryan_scores[application]["avg_religion"]["Muslim_original"].append(methods["original"])
                        indo_aryan_scores[application]["avg_religion"]["Muslim_simple"].append(methods["simple"])
                        indo_aryan_scores[application]["avg_religion"]["Muslim_complex"].append(methods["complex"])
                    elif lang in dravidian_languages:
                        dravidian_scores[application]["avg_religion"]["Muslim_original"].append(methods["original"])
                        dravidian_scores[application]["avg_religion"]["Muslim_simple"].append(methods["simple"])
                        dravidian_scores[application]["avg_religion"]["Muslim_complex"].append(methods["complex"])

                # Gender scores
                if "Male" in identity:
                    all_languages_scores[application]["avg_gender"]["Male_original"].append(methods["original"])
                    all_languages_scores[application]["avg_gender"]["Male_simple"].append(methods["simple"])
                    all_languages_scores[application]["avg_gender"]["Male_complex"].append(methods["complex"])
                    if lang in indo_aryan_languages:
                        indo_aryan_scores[application]["avg_gender"]["Male_original"].append(methods["original"])
                        indo_aryan_scores[application]["avg_gender"]["Male_simple"].append(methods["simple"])
                        indo_aryan_scores[application]["avg_gender"]["Male_complex"].append(methods["complex"])
                    elif lang in dravidian_languages:
                        dravidian_scores[application]["avg_gender"]["Male_original"].append(methods["original"])
                        dravidian_scores[application]["avg_gender"]["Male_simple"].append(methods["simple"])
                        dravidian_scores[application]["avg_gender"]["Male_complex"].append(methods["complex"])
                elif "Female" in identity:
                    all_languages_scores[application]["avg_gender"]["Female_original"].append(methods["original"])
                    all_languages_scores[application]["avg_gender"]["Female_simple"].append(methods["simple"])
                    all_languages_scores[application]["avg_gender"]["Female_complex"].append(methods["complex"])
                    if lang in indo_aryan_languages:
                        indo_aryan_scores[application]["avg_gender"]["Female_original"].append(methods["original"])
                        indo_aryan_scores[application]["avg_gender"]["Female_simple"].append(methods["simple"])
                        indo_aryan_scores[application]["avg_gender"]["Female_complex"].append(methods["complex"])
                    elif lang in dravidian_languages:
                        dravidian_scores[application]["avg_gender"]["Female_original"].append(methods["original"])
                        dravidian_scores[application]["avg_gender"]["Female_simple"].append(methods["simple"])
                        dravidian_scores[application]["avg_gender"]["Female_complex"].append(methods["complex"])

                # Marital status scores
                for status in ["Single", "Married", "Divorced", "Widowed"]:
                    if status in identity:
                        all_languages_scores[application]["avg_marital_status"][f"{status}_original"].append(methods["original"])
                        all_languages_scores[application]["avg_marital_status"][f"{status}_simple"].append(methods["simple"])
                        all_languages_scores[application]["avg_marital_status"][f"{status}_complex"].append(methods["complex"])
                        if lang in indo_aryan_languages:
                            indo_aryan_scores[application]["avg_marital_status"][f"{status}_original"].append(methods["original"])
                            indo_aryan_scores[application]["avg_marital_status"][f"{status}_simple"].append(methods["simple"])
                            indo_aryan_scores[application]["avg_marital_status"][f"{status}_complex"].append(methods["complex"])
                        elif lang in dravidian_languages:
                            dravidian_scores[application]["avg_marital_status"][f"{status}_original"].append(methods["original"])
                            dravidian_scores[application]["avg_marital_status"][f"{status}_simple"].append(methods["simple"])
                            dravidian_scores[application]["avg_marital_status"][f"{status}_complex"].append(methods["complex"])

                # Child count scores
                for child_status in ["No children", "One child", "Many children"]:
                    if child_status in identity:
                        all_languages_scores[application]["avg_child_count"][f"{child_status}_original"].append(methods["original"])
                        all_languages_scores[application]["avg_child_count"][f"{child_status}_simple"].append(methods["simple"])
                        all_languages_scores[application]["avg_child_count"][f"{child_status}_complex"].append(methods["complex"])
                        if lang in indo_aryan_languages:
                            indo_aryan_scores[application]["avg_child_count"][f"{child_status}_original"].append(methods["original"])
                            indo_aryan_scores[application]["avg_child_count"][f"{child_status}_simple"].append(methods["simple"])
                            indo_aryan_scores[application]["avg_child_count"][f"{child_status}_complex"].append(methods["complex"])
                        elif lang in dravidian_languages:
                            dravidian_scores[application]["avg_child_count"][f"{child_status}_original"].append(methods["original"])
                            dravidian_scores[application]["avg_child_count"][f"{child_status}_simple"].append(methods["simple"])
                            dravidian_scores[application]["avg_child_count"][f"{child_status}_complex"].append(methods["complex"])

                # Aggregate scores for this application
                all_languages_scores[application]["aggregate_original_application"].append(methods["original"])
                all_languages_scores[application]["aggregate_simple_application"].append(methods["simple"])
                all_languages_scores[application]["aggregate_complex_application"].append(methods["complex"])
                if lang in indo_aryan_languages:
                    indo_aryan_scores[application]["aggregate_original_application"].append(methods["original"])
                    indo_aryan_scores[application]["aggregate_simple_application"].append(methods["simple"])
                    indo_aryan_scores[application]["aggregate_complex_application"].append(methods["complex"])
                elif lang in dravidian_languages:
                    dravidian_scores[application]["aggregate_original_application"].append(methods["original"])
                    dravidian_scores[application]["aggregate_simple_application"].append(methods["simple"])
                    dravidian_scores[application]["aggregate_complex_application"].append(methods["complex"])

    # Function to compute average values
    def compute_average(scores):
        """Compute the average of the bias scores."""
        averages = defaultdict(dict)
        for application, methods in scores.items():
            for method, values in methods.items():
                # If the values are a list, compute the average of the list
                if isinstance(values, list):
                    averages[application][method] = sum(values) / len(values) 
                    
                # If the values are dictionaries, compute the average of each key in the dictionary
                elif isinstance(values, dict):
                    method_averages = {}
                    for key, sub_values in values.items():
                        if isinstance(sub_values, list):
                            method_averages[key] = sum(sub_values) / len(sub_values) 
                    averages[application][method] = method_averages
        return averages
    
    # Compute final averages
    final_averages_by_language = {
        "Indo-Aryan": {
            "applications": compute_average(indo_aryan_scores)
        },
        "Dravidian": {
            "applications": compute_average(dravidian_scores)
        },
        "All_Languages": {
            "applications": compute_average(all_languages_scores)
        }
    }

    return final_averages_by_language
    

# Store bias scores per language
language_bias_scores = {}
tf_idf_scores_all_languages = {}  # Store all TF-IDF scores for top-term analysis

languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

for lang in languages:
    tfidf_path = f"../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/tfidf_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json"
    
    # Load precomputed TF-IDF scores
    tf_idf_scores = load_json(tfidf_path)

    # Compute bias scores
    bias_scores = compute_bias_score(tf_idf_scores)
    
    # Save  scores per language
    save_json(bias_scores, f"../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/bias_scores_{lang}.json")
    
    language_bias_scores[lang] = bias_scores  # Store bias scores

    tf_idf_scores_all_languages[lang] = tf_idf_scores  # Store TF-IDF for top-term analysis

    print(f"Bias scores saved successfully for {lang}.")

# Define the base identities we want to analyze
identities_to_analyze = ["Muslim Male", "Muslim Female", "Hindu Male", "Hindu Female"]

# Run the function
top_tfidf_per_identity_group_and_application = get_top_tfidf_per_application_identity(tf_idf_scores_all_languages, identities_to_analyze)

def generate_latex_tables_by_application_per_language(top_tfidf_per_identity_group_and_application, language_bias_scores):
    """
    Generates LaTeX tables for the highest TF-IDF terms per application, then per identity - only in original debiasing method.
    Now includes a Bias column placed before the term column, with color coding based on bias severity.
    """

    application_order = ["Story", "Hobbies and Values", "To-do List"]  # Enforce order

    # Define all possible identity combinations
    identity_combinations = [
        (religion, gender, marital_status, children)
        for religion in ["Hindu", "Muslim"]
        for gender in ["Male", "Female"]
        for marital_status in ["Single", "Divorced", "Widowed", "Married"]
        for children in ["No children", "One child", "Many children"]
    ]

    for lang, application_data in top_tfidf_per_identity_group_and_application.items():
        if lang != "NONE":  # Process only certain langs 
            for application in application_order:  # Iterate in specified order
                if application not in application_data:
                    continue  # Skip if application is not present
                
                identity_data = application_data[application]
                print(f"\n\\section{{Top Bias Terms for {lang} - {application}}}")

                # Calculate mean and standard deviation for TF-IDF values
                tfidf_values = [entry["tfidf_value"] for entry in identity_data.values()]
                mean_tfidf = sum(tfidf_values) / len(tfidf_values)
                std_dev_tfidf = (sum((x - mean_tfidf) ** 2 for x in tfidf_values) / len(tfidf_values)) ** 0.5

                # Define thresholds for TF-IDF color coding
                lower_tfidf_threshold = mean_tfidf - std_dev_tfidf
                upper_tfidf_threshold = mean_tfidf + std_dev_tfidf

                # Collect Bias scores
                bias_score_scores = [
                    language_bias_scores.get(lang, {}).get(entry["identity"], {}).get(application, {}).get("original", 0)
                    for entry in identity_data.values()
                ]
                mean_bias_score = sum(bias_score_scores) / len(bias_score_scores)
                std_dev_bias_score = (sum((x - mean_bias_score) ** 2 for x in bias_score_scores) / len(bias_score_scores)) ** 0.5

                # Define thresholds for Bias color coding
                lower_bias_score_threshold = mean_bias_score - std_dev_bias_score
                upper_bias_score_threshold = mean_bias_score + std_dev_bias_score

                print(f"\\newpage")
                print(f"\\begin{{table}}[h!]")
                print(f"\\centering")
                print(f"\\caption{{Highest Bias TF-IDF terms and values for all identities under the application \\textbf{{{application}}} in {lang} outputs under the original prompting method.}}")
                print(f"\\scriptsize")
                print(f"\\label{{tab:tfidf_comparison_identity_bias_{lang}_{application.replace(' ', '_')}}}")

                # Adjusted column order: Norm Bias Score now before the Term column
                print(f"\\begin{{tabular}}{{|p{{1.9cm}}|p{{1.3cm}}|l|p{{0.7cm}}|l|p{{0.7cm}}|}}")
                print(f"\\hline")
                print(f"\\textbf{{Religion \\& Gender}} & \\textbf{{Marital Status}} & \\textbf{{Children}} & \\textbf{{Bias Score}} & \\textbf{{Term}} & \\textbf{{Bias TF-IDF}} \\\\")
                print(f"\\hline")

                # Dictionary to store identity-wise data
                identity_grouped_entries = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

                # Process identity data
                for entry in identity_data.values():
                    identity = entry["identity"]
                    term = entry["term"]
                    tfidf_value = entry["tfidf_value"]

                    # Extract identity components
                    parts = identity.split("who is")
                    main_identity = parts[0].strip().replace("A ", "")
                    details = parts[1].strip() if len(parts) > 1 else ""

                    # Extract marital status and child count
                    marital_status = "Single"
                    if "Married" in details:
                        marital_status = "Married"
                    elif "Divorced" in details:
                        marital_status = "Divorced"
                    elif "Widowed" in details:
                        marital_status = "Widowed"

                    children = "No children"
                    if "One child" in details:
                        children = "One child"
                    elif "Many children" in details:
                        children = "Many children"

                    # Retrieve NormBias value
                    bias_score = language_bias_scores.get(lang, {}).get(identity, {}).get(application, {}).get("original", 0)
                    
                    # Ensure bias_score is numeric and within [0,1]
                    bias_score = max(0, min(1, bias_score)) if isinstance(bias_score, (int, float)) else "N/A"

                    # Determine TF-IDF color coding
                    if tfidf_value > upper_tfidf_threshold:
                        tfidf_color = "\\cellcolor{red!30}"
                    elif tfidf_value < lower_tfidf_threshold:
                        tfidf_color = "\\cellcolor{green!30}"
                    else:
                        tfidf_color = "\\cellcolor{yellow!30}"

                    # Determine Norm Bias color coding
                    if isinstance(bias_score, (int, float)):
                        if bias_score > upper_bias_score_threshold:
                            bias_score_color = "\\cellcolor{red!30}"
                        elif bias_score < lower_bias_score_threshold:
                            bias_score_color = "\\cellcolor{green!30}"
                        else:
                            bias_score_color = "\\cellcolor{yellow!30}"
                        bias_score_display = f"{bias_score_color}{bias_score:.3f}"
                    else:
                        bias_score_display = "N/A"

                    # Store data
                    identity_grouped_entries[main_identity][marital_status][children].append((bias_score_display, term, tfidf_color, tfidf_value))

                # Sorting logic
                sorted_main_identities = sorted(identity_grouped_entries.keys())
                marital_status_order = ["Single", "Married", "Divorced", "Widowed"]
                children_order = ["No children", "One child", "Many children"]

                # Fill missing identities with default values
                for combination in identity_combinations:
                    religion, gender, marital_status, children = combination
                    identity_key = f"{religion} {gender}"
                    
                    if identity_key not in identity_grouped_entries:
                        identity_grouped_entries[identity_key] = defaultdict(lambda: defaultdict(list))

                    if marital_status not in identity_grouped_entries[identity_key]:
                        identity_grouped_entries[identity_key][marital_status] = defaultdict(list)

                    if children not in identity_grouped_entries[identity_key][marital_status]:
                        identity_grouped_entries[identity_key][marital_status][children] = [
                            ("\\cellcolor{green!30}0.000", "N/A", "\\cellcolor{green!30}", "N/A")
                        ]

                # Now print the table data
                for main_identity in sorted_main_identities:
                    marital_groups = identity_grouped_entries[main_identity]
                    sorted_marital_statuses = sorted(marital_groups.keys(), key=lambda x: marital_status_order.index(x))
                    num_rows_identity = sum(len(children_list) for children_list in marital_groups.values())
                    first_identity = True

                    print(f"\\hline")

                    for marital_status in sorted_marital_statuses:
                        children_list = marital_groups[marital_status]
                        num_rows_marital = len(children_list)
                        first_marital = True
                        sorted_children_list = sorted(children_list.items(), key=lambda x: children_order.index(x[0]))

                        for idx, (children, entries) in enumerate(sorted_children_list):
                            if first_identity:
                                print(f"\\multirow{{{num_rows_identity}}}{{*}}{{{main_identity}}} ", end="")
                                first_identity = False
                            else:
                                print(" ", end="")

                            if first_marital:
                                print(f"& \\multirow{{{num_rows_marital}}}{{*}}{{{marital_status}}} ", end="")
                                first_marital = False
                            else:
                                print("& ", end="")

                            for bias_score_display, term, tfidf_color, tfidf_value in entries:
                                if isinstance(tfidf_value, (int, float)):
                                    print(f"& {children} & {bias_score_display} & {term} & {tfidf_color}{tfidf_value:.3f} \\\\")
                                else:
                                    print(f"& {children} & {str(bias_score_display)} & {str(term)} & {tfidf_color}{'N/A'} \\\\")

                            if idx < len(sorted_children_list) - 1:
                                print(f"\\cline{{3-6}}")

                        print(f"\\cline{{2-6}}")

                    print(f"\\hline")

                print(f"\\end{{tabular}}")
                print(f"\\end{{table}}")
                print(f"\\newpage")
                print()

#generate_latex_tables_by_application_per_language(top_tfidf_per_identity_group_and_application, language_bias_scores)


def save_data_by_application_family(top_tfidf_per_identity_group_and_application, language_bias_scores):
    """
    Generates LaTeX tables for the highest TF-IDF terms per application, grouped by Indo-Aryan and Dravidian language families.
    Uses original prompting method only.
    Bias scores are averaged across all languages in a family.
    Stores results in a JSON file before printing LaTeX tables.
    """

    # Define language families
    indo_aryan_languages = {"Hindi", "Bengali", "Urdu", "Punjabi", "Marathi", "Gujarati"}
    dravidian_languages = {"Tamil", "Telugu", "Kannada", "Malayalam"}

    language_families = {
        "Indo-Aryan": indo_aryan_languages,
        "Dravidian": dravidian_languages
    }

    application_order = ["Story", "Hobbies and Values", "To-do List"]  # Enforce order

    # Aggregate TF-IDF and bias scores by language family
    family_tfidf_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    family_bias_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    family_language_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # Track number of languages contributing

    for lang, application_data in top_tfidf_per_identity_group_and_application.items():
        # Determine language family
        family = next((fam for fam, langs in language_families.items() if lang in langs), None)
        if not family:
            continue  # Skip languages not in defined families
        
        for application in application_order:
            identity_data = application_data[application]

            for entry in identity_data.values():
                identity = entry["identity"]
                term = entry["term"]
                tfidf_value = entry["tfidf_value"]

                # Store TF-IDF data
                family_tfidf_data[family][application][identity].append((term, tfidf_value))

                # Accumulate bias scores and count contributions
                bias_score = language_bias_scores.get(lang, {}).get(identity, {}).get(application, {}).get("original", 0)
                family_bias_scores[family][application][identity] += bias_score
                family_language_count[family][application][identity] += 1

    # Compute average bias scores
    for family in family_bias_scores:
        for application in family_bias_scores[family]:
            for identity in family_bias_scores[family][application]:
                count = family_language_count[family][application].get(identity)  
                family_bias_scores[family][application][identity] /= count  # Convert to average

    # Prepare JSON output
    json_output = {}

    for family, application_data in family_tfidf_data.items():
        json_output[family] = {}
        for application in application_order:
            if application not in application_data:
                continue  # Skip if application is missing
            
            json_output[family][application] = {}

            for identity, terms in application_data[application].items():
                top_terms = sorted(terms, key=lambda x: x[1], reverse=True)
                top_term, top_tfidf = top_terms[0] if top_terms else ("N/A", 0.0)
                
                json_output[family][application][identity] = {
                    "average_bias_score": family_bias_scores[family][application].get(identity, 0),
                    "top_term": top_term,
                    "top_tfidf": top_tfidf
                }

    # Save to JSON file
    with open("../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/avg_bias_scores_by_language_family.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4)

    print("Data stored in JSON!")




# Call the function with your data
save_data_by_application_family(top_tfidf_per_identity_group_and_application, language_bias_scores)

print("Bias term extraction and LaTeX formatting completed.")

print("Cross-language analysis completed.")


import json
import matplotlib.pyplot as plt



def plot_bias_scores_all_identities(json_filepath):
    """
    Plots a scatter plot of bias scores by identity for each (language-family, application) group in one graph.
    
    Args:
        json_filepath (str): Path to the JSON file containing bias scores.
    """
    # Define colors for identity groups
    identity_colors = {
        "Hindu Female": "#e41a1c",  # Red
        "Hindu Male": "#377eb8",    # Blue
        "Muslim Female": "#4daf4a",  # Green
        "Muslim Male": "#984ea3"    # Purple
    }

    # Load data
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare data for scatter plot
    x_labels = []
    y_values = []
    identity_labels = []
    colors = []
    x_positions = []

    # Collect the data for the plot
    for family, application_data in data.items():
        for application, identities in application_data.items():
            x_label = f"{family}-{application}"
            x_labels.append(x_label)

            # Get the current x position for this family-application
            current_x_position = len(x_positions)  # Increment this as we go

            for identity, values in identities.items():
                # Default bias score to 0 if it does not exist
                bias_score = values.get("average_bias_score", 0)

                # Assign color based on identity
                if "Hindu Female" in identity:
                    color = identity_colors["Hindu Female"]
                elif "Hindu Male" in identity:
                    color = identity_colors["Hindu Male"]
                elif "Muslim Female" in identity:
                    color = identity_colors["Muslim Female"]
                elif "Muslim Male" in identity:
                    color = identity_colors["Muslim Male"]

                # Append data points
                y_values.append(bias_score)
                identity_labels.append(identity)
                colors.append(color)
                x_positions.append(current_x_position)

    # Create figure with a slightly smaller size
    plt.figure(figsize=(10, 8))  # Slightly smaller figure size for better readability

    # Scatter plot with much smaller points
    plt.scatter(x_positions, y_values, c=colors, alpha=0.7, edgecolors="black", s=20)  # Much smaller points (s=20)

    # Reduce the number of ticks to match the labels (select every nth position)
    step_size = len(x_positions) // len(x_labels)  # This will give us the step size to select one x-tick per table
    selected_positions = x_positions[::step_size]  # Select every nth x-position

    # Set x-tick positions and labels to match selected positions
    plt.xticks(ticks=selected_positions, labels=x_labels, rotation=45, ha="center", fontsize=8)

    # Set labels and title with smaller font sizes
    plt.ylabel("Average Bias Score", fontsize=10)
    plt.xlabel("Language Family - Application", fontsize=10)
    plt.title("Bias Scores by Identity across Applications and Language Families", fontsize=10)

    # Create legend with smaller font size
    legend_patches = [plt.Line2D([0], [0], marker="o", color="w", markersize=8, markerfacecolor=col, label=label)
                      for label, col in identity_colors.items()]
    plt.legend(handles=legend_patches, title="Identity", loc="upper right", fontsize=8)

    # Adjust the layout to prevent label overlap and align everything correctly
    plt.subplots_adjust(bottom=0.35, left=0.1, right=0.9, top=0.9)  # Increased bottom spacing for better label placement
    
    # Ensure no overlap between x-ticks and labels
    plt.tight_layout()

    # Show plot
    plt.show()

# Example usage:
#plot_bias_scores_all_identities("../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/avg_bias_scores_by_language_family.json")

def generate_latex_table_by_application_top_terms(json_path):
    """
    Reads a JSON file containing top Bias TF-IDF terms per identity, aggregates the data, for original prompting method
    and generates LaTeX tables per application, displaying the 48 identities with their highest TF-IDF terms.
    The identities will be grouped and sorted by religion, gender, marital status, and child count.
    It also calculates mean and standard deviation of TF-IDF values to apply color coding.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    applications = ["Story", "Hobbies and Values", "To-do List"]
    aggregated_data = defaultdict(lambda: defaultdict(dict))  # application -> identity -> {top_term, tfidf}

    for family, app_data in data.items():
        for application in applications:
            if application not in app_data:
                continue
            
            for identity, identity_data in app_data[application].items():
                top_term = identity_data["top_term"]
                top_tfidf = identity_data["top_tfidf"]

                if identity not in aggregated_data[application]:
                    aggregated_data[application][identity] = {
                        "top_term": top_term,
                        "top_tfidf": top_tfidf
                    }
                
                if top_tfidf > aggregated_data[application][identity]["top_tfidf"]:
                    aggregated_data[application][identity]["top_term"] = top_term
                    aggregated_data[application][identity]["top_tfidf"] = top_tfidf

    # Compute mean and standard deviation for each application's TF-IDF values
    tfidf_stats = {}
    for application in applications:
        tfidf_values = [values["top_tfidf"] for values in aggregated_data[application].values()]
        mean_tfidf = np.mean(tfidf_values)
        std_tfidf = np.std(tfidf_values)
        tfidf_stats[application] = {"mean": mean_tfidf, "std": std_tfidf}

    # Generate LaTeX tables
    for application in applications:
        mean_tfidf = tfidf_stats[application]["mean"]
        std_tfidf = tfidf_stats[application]["std"]
        upper_tfidf_threshold = mean_tfidf + std_tfidf
        lower_tfidf_threshold = mean_tfidf - std_tfidf

        print(f"\n\\section{{Top TF-IDF Terms - {application}}}")
        print("\\newpage")
        
        print("\\begin{table}[h!]")
        print("\\centering")
        print(f"\\caption{{Highest Bias TF-IDF Terms for All identities Under the Application \\textbf{{{application}}} (All Languages with \\textbf{{Original}} Prompting Method)}}")
        print("\\scriptsize")
        print(f"\\label{{tab:tfidf_comparison_all_identities_{application.replace(' ', '_')}}}")

        print(f"\\begin{{tabular}}{{|p{{2.5cm}}|p{{2cm}}|p{{2cm}}|l|c|}}")
        print("\\hline")
        print(f"\\textbf{{Religion \\& Gender}} & \\textbf{{Marital Status}} & \\textbf{{Children}} & \\textbf{{Bias Term}} & \\textbf{{Bias TF-IDF}} \\\\")
        print("\\hline")

        processed_identities = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for identity, values in aggregated_data[application].items():
            parts = identity.split("who is")
            main_identity = parts[0].strip().replace("A ", "")
            details = parts[1].strip() if len(parts) > 1 else ""

            marital_status = "Single"
            if "Married" in details:
                marital_status = "Married"
            elif "Divorced" in details:
                marital_status = "Divorced"
            elif "Widowed" in details:
                marital_status = "Widowed"

            children = "No children"
            if "One child" in details:
                children = "One child"
            elif "Many children" in details:
                children = "Many children"

            processed_identities[main_identity][marital_status][children].append(
                (values["top_term"], values["top_tfidf"])
            )

        sorted_main_identities = sorted(processed_identities.keys())
        marital_status_order = ["Single", "Married", "Divorced", "Widowed"]
        children_order = ["No children", "One child", "Many children"]

        for main_identity in sorted_main_identities:
            marital_groups = processed_identities[main_identity]
            sorted_marital_statuses = sorted(marital_groups.keys(), key=lambda x: marital_status_order.index(x))

            print(f"\\multirow{{{len(sorted_marital_statuses) * len(children_order)}}}{{*}}{{{main_identity}}} ", end="")

            for marital_status in sorted_marital_statuses:
                children_groups = marital_groups[marital_status]
                sorted_children_groups = sorted(children_groups.items(), key=lambda x: children_order.index(x[0]))

                first_marital = True

                for children, data in sorted_children_groups:
                    for idx, (top_term, top_tfidf) in enumerate(data):
                        if first_marital:
                            print(f"& \\multirow{{{len(data)}}}{{*}}{{{marital_status}}} ", end="")
                            first_marital = False
                        else:
                            print("& ", end="")

                        # Determine TF-IDF color coding
                        if top_tfidf > upper_tfidf_threshold:
                            tfidf_color = "\\cellcolor{red!30}"
                        elif top_tfidf < lower_tfidf_threshold:
                            tfidf_color = "\\cellcolor{green!30}"
                        else:
                            tfidf_color = "\\cellcolor{yellow!30}"

                        # Print the row with TF-IDF coloring
                        print(f"& {children} & {top_term} & {tfidf_color} {top_tfidf:.3f} \\\\")

                        if idx < len(data) - 1:
                            print(f"\\cline{{3-5}}")

                    print(f"\\cline{{3-5}}")  

                print(f"\\cline{{2-5}}")

            print(f"\\hline")

        print("\\end{tabular}")
        print("\\end{table}")
        print("\\newpage")
        print(f"mean: {mean_tfidf} mean+stddev {upper_tfidf_threshold} mean-stddev {lower_tfidf_threshold}")

# Call the function with the path to the JSON file
generate_latex_table_by_application_top_terms("../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/avg_bias_scores_by_language_family.json")

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools


def generate_matrix_heatmap(json_path):
    """
    Generates a matrix heatmap where:
    - Religion & Gender are on the Y-axis
    - Marital Status & Number of Children are on the X-axis
    - Each cell contains the highest bias term for that identity
    - Color is based on TF-IDF thresholds (mean ± stddev) computed across all language families per application
    - Text inside cells and axes is small for readability
    """

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    applications = ["Story", "Hobbies and Values", "To-do List"]
    
    # Define row (Y-axis) and column (X-axis) identities
    religions = ["Hindu", "Muslim"]
    genders = ["Male", "Female"]
    marital_statuses = ["Married", "Single", "Divorced", "Widowed"]
    child_counts = ["No children", "One child", "Many children"]

    row_labels = [f"{r} & {g}" for r, g in itertools.product(religions, genders)]
    col_labels = [f"{m} & {c}" for m, c in itertools.product(marital_statuses, child_counts)]

    for application in applications:
        aggregated_data = defaultdict(lambda: defaultdict(dict))  # Row -> Column -> {term, tfidf}

        # Aggregate highest TF-IDF terms across all language families
        for language_family in data.keys():
            if application not in data[language_family]:
                continue  # Skip if application is missing

            for identity, identity_data in data[language_family][application].items():
                top_term = identity_data["top_term"]
                top_tfidf = identity_data["top_tfidf"]

                # Extract components from identity string (assuming "A Hindu Male who is Married with No children")
                parts = identity.split(" ")
                religion = parts[1]
                gender = parts[2]
                marital_status = parts[5]
                children_status = " ".join(parts[7:])  # Handles "No children", "One child", etc.

                row_key = f"{religion} & {gender}"
                col_key = f"{marital_status} & {children_status}"

                if col_key not in aggregated_data[row_key]:
                    aggregated_data[row_key][col_key] = {"top_term": top_term, "top_tfidf": top_tfidf}
                elif top_tfidf > aggregated_data[row_key][col_key]["top_tfidf"]:
                    aggregated_data[row_key][col_key] = {"top_term": top_term, "top_tfidf": top_tfidf}

        # Compute mean and stddev for TF-IDF across all language families
        tfidf_values = [v["top_tfidf"] for row in aggregated_data.values() for v in row.values()]
        if not tfidf_values:
            continue  # Skip if no data

        mean_tfidf = np.mean(tfidf_values)
        std_tfidf = np.std(tfidf_values)
        upper_threshold = mean_tfidf + std_tfidf
        lower_threshold = mean_tfidf - std_tfidf

        # Construct the matrix
        matrix = np.full((len(row_labels), len(col_labels)), "", dtype=object)  # Empty string default

        for i, row_label in enumerate(row_labels):
            for j, col_label in enumerate(col_labels):
                if row_label in aggregated_data and col_label in aggregated_data[row_label]:
                    matrix[i, j] = aggregated_data[row_label][col_label]["top_term"]

        # Define color mapping
        tfidf_matrix = np.zeros((len(row_labels), len(col_labels)))  # Default 0 TF-IDF

        for i, row_label in enumerate(row_labels):
            for j, col_label in enumerate(col_labels):
                if row_label in aggregated_data and col_label in aggregated_data[row_label]:
                    tfidf_matrix[i, j] = aggregated_data[row_label][col_label]["top_tfidf"]

        bounds = [0, lower_threshold, upper_threshold, max(tfidf_values) + 0.01]  
        colors = ["#b2ffb2", "#fff7c5", "#ffb2b2"] #["green", "orange", "red"] color mapping
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plot heatmap
        plt.figure(figsize=(14, 5)) 
        heatmap = sns.heatmap(
            tfidf_matrix, cmap=cmap, norm=norm, annot=matrix, fmt="", 
            linewidths=0.5, xticklabels=col_labels, yticklabels=row_labels, 
            annot_kws={"fontsize": 8.5},  # Smaller text inside cells
            cbar_kws={'shrink': 0.5, 'label': 'Bias TF-IDF',  'pad': 0.01}  # Smaller color bar with label
        )

        # Rotate and resize the annotation text inside the cells
        for text in heatmap.texts:
            text.set_rotation(45)     
            text.set_fontsize(10)

        # Set color bar tick labels to 3 decimal places
        cbar = heatmap.collections[0].colorbar  # Access the color bar
        cbar.set_ticks(bounds)  # Set tick locations
        cbar.set_ticklabels([f"{b:.3f}" for b in bounds])  # Format labels to .3f

        plt.title(f"Top Bias Terms for All Identities in {application} Generations (Aggregated Across All Languages in Original Prompting Method)", fontsize=10)  # Smaller title
        plt.xlabel("Marital Status & Child Count", fontsize=10)
        plt.ylabel("Religion & Gender", fontsize=10)
        plt.xticks(rotation=45, ha="right", fontsize=10)  # Adjusted rotation and font size for X-axis labels
        plt.yticks(rotation=0, fontsize=10)  # Smaller Y-axis labels
        plt.tight_layout()  # Adjusts plot to ensure labels are visible
        plt.savefig(f"../../../data/figures/top_bias_terms_{application}.pdf", bbox_inches='tight')
        plt.show()

# Call function with JSON path
generate_matrix_heatmap("../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/avg_bias_scores_by_language_family.json")


indo_aryan_languages = ["Hindi", "Bengali", "Urdu", "Punjabi", "Marathi", "Gujarati"]  # list of Indo-Aryan languages
dravidian_languages = ["Tamil", "Telugu", "Kannada", "Malayalam"]  # list of Dravidian languages
languages = ["Hindi", "Bengali", "Urdu", "Punjabi", "Marathi", "Gujarati", "Tamil", "Telugu", "Kannada", "Malayalam"]


avg_scores_by_language = calculate_averaged_scores_per_language(language_bias_scores, indo_aryan_languages, dravidian_languages)
# Save the results
save_json(avg_scores_by_language, "../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/aggregated_bias_scores_by_language.json")

print("Updated JSON file with Indo-Aryan and Dravidian family averages.")


def plot_bias_scores_individual_identity_categories(data, category, subcategories, title):
    """
    Plots bias scores for a specific category, separating Indo-Aryan and Dravidian applications into subplots.
    
    Args:
        data: The JSON data
        category: The category to plot (e.g., 'avg_gender', 'avg_religion')
        subcategories: List of subcategories (e.g., ['Male_original', 'Female_original'])
        title: The title of the plot
    """
    # Extract data for Indo-Aryan and Dravidian
    scores_indo = data["Indo-Aryan"]["applications"]
    scores_drav = data["Dravidian"]["applications"]
    
    # Get applications in order
    applications = list(scores_indo.keys())  # ['Story', 'Hobbies and Values', 'To-do List']
    
    # Prepare dictionaries to store scores
    category_scores_indo = {sub.replace("_original", ""): [] for sub in subcategories}
    category_scores_drav = {sub.replace("_original", ""): [] for sub in subcategories}

    print(f"\n### Extracted Bias Scores for {title} ###\n")

    # Retrieve scores for both language families
    for app in applications:
        print(f"\nApplication: {app}")

        # Indo-Aryan scores
        if category in scores_indo[app]:
            for subcategory in subcategories:
                clean_subcategory = subcategory.replace("_original", "")
                score = scores_indo[app][category].get(subcategory, 0)
                category_scores_indo[clean_subcategory].append(score)
                print(f"  Indo-Aryan - {clean_subcategory}: {score:.3f}")
        else:
            for clean_subcategory in category_scores_indo.keys():
                category_scores_indo[clean_subcategory].append(0)
                print(f"  Indo-Aryan - {clean_subcategory}: 0.000 (No Data)")

        # Dravidian scores
        if category in scores_drav[app]:
            for subcategory in subcategories:
                clean_subcategory = subcategory.replace("_original", "")
                score = scores_drav[app][category].get(subcategory, 0)
                category_scores_drav[clean_subcategory].append(score)
                print(f"  Dravidian - {clean_subcategory}: {score:.3f}")
        else:
            for clean_subcategory in category_scores_drav.keys():
                category_scores_drav[clean_subcategory].append(0)
                print(f"  Dravidian - {clean_subcategory}: 0.000 (No Data)")

    # X-axis positions
    num_apps = len(applications)
    x_positions = np.arange(num_apps)  # One position per application
    
    # Define dynamic color palette based on number of subcategories
    color_palette = {
        2: ["#FFA491", "#78D39A"],  # Two colors 
        3: ["#FFA491", "#78D39A", "#8FC6FF"],  # Three colors 
        4: ["#FFA491", "#78D39A", "#8FC6FF", "#FFCE86"],  # Four colors 
    }
    
    num_subcategories = len(subcategories)
    colors = color_palette.get(num_subcategories, sns.color_palette("tab10", num_subcategories))  # Default to tab10 if not predefined
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)

    # Bar width
    width = 0.2  

    # Plot for Indo-Aryan languages
    for idx, subcategory in enumerate(category_scores_indo.keys()):
        bars = axes[0].bar(
            x_positions + width * idx, 
            category_scores_indo[subcategory],
            width=width,
            label=subcategory.title(),
            color=colors[idx]  # Assign custom color
        )

        # Add value labels
        for bar in bars:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}", 
                         ha='center', va='bottom', fontsize=10, color='black')


    tick_adjustment = (num_subcategories - 1) * width / 2  # Center tick between bars

    axes[0].set_title(f"{title} (Indo-Aryan)")
    axes[0].set_xticks(x_positions + tick_adjustment)
    axes[0].set_xticklabels(applications, fontsize=16)
    axes[0].set_xlabel("Applications")
    axes[0].set_ylabel("Average Bias Score")

    # Dynamically set the legend title
    if "gender" in category:
        axes[0].legend(title="Genders", fontsize=16, title_fontsize=16)
    elif "religion" in category:
        axes[0].legend(title="Religions", fontsize=16, title_fontsize=16)
    elif "marital_status" in category:
        axes[0].legend(title="Marital Statuses", fontsize=16, title_fontsize=16)
    elif "child_count" in category:
        axes[0].legend(title="Number of Children", fontsize=16, title_fontsize=16)

    # Plot for Dravidian languages
    for idx, subcategory in enumerate(category_scores_drav.keys()):
        bars = axes[1].bar(
            x_positions + width * idx, 
            category_scores_drav[subcategory],
            width=width,
            label=subcategory.title(),
            color=colors[idx]  # Assign custom color
        )

        # Add value labels
        for bar in bars:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}", 
                         ha='center', va='bottom', fontsize=10, color='black')

    axes[1].set_title(f"{title} (Dravidian)")
    axes[1].set_xticks(x_positions + width)
    axes[1].set_xticks(x_positions + tick_adjustment)
    axes[1].set_xticklabels(applications, fontsize=16)
    axes[1].set_xlabel("Applications")
    axes[0].tick_params(axis='y', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)

    # Set the overall title
    fig.suptitle(f"{title} Bias Scores (Averaged Across Language Families in Original Prompting Method)", fontsize=16)

    # Adjust font sizes for subplots
    for ax in axes:
        ax.set_title(ax.get_title(), fontsize=16)  # Set subplot title font size
        ax.set_xlabel("Applications", fontsize=16)  # Set x-axis label font size
        ax.set_ylabel("Average Bias Score", fontsize=16)  # Set y-axis label font size

    # Adjust layout and bring the suptitle as close as possible
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)  # Bring suptitle closer to plots
    plt.savefig(f"../../../data/figures/bias_scores_{category}.pdf", bbox_inches='tight')
    plt.show()

# Function to generate plots for gender, religion, marital status, and child count
def generate_bias_plots(file_path):
    # Load the data
    data = load_json(file_path)
    
    # Define categories and subcategories for plotting
    categories = {
        "avg_gender": ["Male_original", "Female_original"],
        "avg_religion": ["Hindu_original", "Muslim_original"],
        "avg_marital_status": ["Single_original", "Married_original", "Divorced_original", "Widowed_original"],
        "avg_child_count": ["No children_original", "One child_original", "Many children_original"],
    }
    
    # Generate plots for each category
    for category, subcategories in categories.items():
        formatted_category = category.replace("avg_", "").replace("_", " ").title()
        title = f"Average {formatted_category}"
        plot_bias_scores_individual_identity_categories(data, category, subcategories, title=title)

# Example usage
file_path = "../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/aggregated_bias_scores_by_language.json"
# Generate all bias plots
generate_bias_plots(file_path)

# Function to load data and plot
def generate_bias_comparison_plots(file_path):
    # Load the data
    data = load_json(file_path)
    
    # Generate the comparison plot
    plot_application_bias_by_language_family_debiasing_method(data)

def plot_application_bias_by_language_family_debiasing_method(data, title="Application"):
    """
    Plots bias scores across applications separately for Indo-Aryan and Dravidian language families.
    Also prints extracted bias scores in the terminal.

    Args:
        data: The JSON data.
        title: The title prefix for the plot.
    """
    # Extract bias scores for Indo-Aryan and Dravidian languages
    scores_indo_aryan = data["Indo-Aryan"]["applications"]
    scores_dravidian = data["Dravidian"]["applications"]
    
    # Get application names
    applications = list(scores_indo_aryan.keys())  # ['Story', 'Hobbies and Values', 'To-do List']
    
    # Define the subcategories to plot
    subcategories = ["aggregate_original_application", 
                     "aggregate_simple_application", 
                     "aggregate_complex_application"]
    
    # Prepare dictionaries to store the scores for both language families
    category_scores_indo_aryan = {subcategory: [] for subcategory in subcategories}
    category_scores_dravidian = {subcategory: [] for subcategory in subcategories}

    # Retrieve bias scores for each application in both language families
    print("\n### Extracted Bias Scores ###")
    for app in applications:
        print(f"\nApplication: {app}")
        for subcategory in subcategories:
            ia_score = scores_indo_aryan.get(app, {}).get(subcategory, 0)
            dr_score = scores_dravidian.get(app, {}).get(subcategory, 0)

            category_scores_indo_aryan[subcategory].append(ia_score)
            category_scores_dravidian[subcategory].append(dr_score)

            print(f"  {subcategory.replace('aggregate_', '').replace('_application', '').title()}:")
            print(f"    Indo-Aryan: {ia_score:.3f}, Dravidian: {dr_score:.3f}")

    # X-axis positions
    num_apps = len(applications)
    x_positions = np.arange(num_apps)  # One position per application

    # Define color palette for three subcategories
    colors = ["#FFA491", "#78D39A", "#8FC6FF"]

    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)

    # Bar width
    width = 0.2  

    # Plot for Indo-Aryan languages
    for idx, subcategory in enumerate(subcategories):
        bars = axes[0].bar(
            x_positions + width * idx, 
            category_scores_indo_aryan[subcategory],
            width=width,
            label=subcategory.replace("aggregate_", "").replace("_application", "").title(),
            color=colors[idx]  # Assign custom color
        )

        # Add value labels
        for bar in bars:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}", 
                         ha='center', va='bottom', fontsize=10, color='black')

    axes[0].set_title("Indo-Aryan Languages", fontsize=16)
    axes[0].set_xticks(x_positions + width)
    axes[0].set_xticklabels(applications, fontsize=16)
    axes[0].set_xlabel("Applications", fontsize=16)
    axes[0].set_ylabel("Average Bias Score", fontsize=16)
    axes[0].legend(title="Prompting Methods", fontsize=16, title_fontsize=16)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)

    # Plot for Dravidian languages
    for idx, subcategory in enumerate(subcategories):
        bars = axes[1].bar(
            x_positions + width * idx, 
            category_scores_dravidian[subcategory],
            width=width,
            label=subcategory.replace("aggregate_", "").replace("_application", "").title(),
            color=colors[idx]  # Assign custom color
        )

        # Add value labels
        for bar in bars:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}", 
                         ha='center', va='bottom', fontsize=10, color='black')

    axes[1].set_title("Dravidian Languages", fontsize=16)
    axes[1].set_xticks(x_positions + width)
    axes[1].set_xticklabels(applications, fontsize=16)
    axes[1].set_ylabel("Average Bias Score", fontsize=16)
    axes[1].set_xlabel("Applications", fontsize=16)

    # Super title
    fig.suptitle(f"{title} Bias Scores by Prompting Method (Averaged Across Language Families)", fontsize=16)

    # Adjust layout and bring the suptitle as close as possible
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)  # Bring suptitle closer to plots
    plt.savefig(f"../../../data/figures/bias_scores_debiasing_methods.pdf", bbox_inches='tight')
    plt.show()

# Example usage
file_path = "../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/aggregated_bias_scores_by_language.json"

# Generate the comparison plots for Indo-Aryan vs. Dravidian
generate_bias_comparison_plots(file_path)


