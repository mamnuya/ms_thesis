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
    """Compute the summation of bias score for each (identity, application, method) pair."""
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


def calculate_summed_scores_per_language(language_bias_scores):
    """Calculate summed bias scores for each application within a language, then compute final aggregate scores per method."""
    summed_scores_by_language = {}

    for lang, bias_scores in language_bias_scores.items():
        summed_scores_by_application = {}
        final_aggregate_scores = {"original": 0, "simple": 0, "complex": 0}

        # Initialize structure for applications
        for app in ["Story", "Hobbies and Values", "To-do List"]:
            summed_scores_by_application[app] = {
                "sum_religion": {"Hindu_original": 0, "Muslim_original": 0, "Hindu_simple": 0, "Muslim_simple": 0, "Hindu_complex": 0, "Muslim_complex": 0},
                "sum_gender": {"Male_original": 0, "Female_original": 0, "Male_simple": 0, "Female_simple": 0, "Male_complex": 0, "Female_complex": 0},
                "sum_marital_status": {"Single_original": 0, "Married_original": 0, "Divorced_original": 0, "Widowed_original": 0,
                                       "Single_simple": 0, "Married_simple": 0, "Divorced_simple": 0, "Widowed_simple": 0,
                                       "Single_complex": 0, "Married_complex": 0, "Divorced_complex": 0, "Widowed_complex": 0},
                "sum_children_count": {"No children_original": 0, "One child_original": 0, "Many children_original": 0,
                                       "No children_simple": 0, "One child_simple": 0, "Many children_simple": 0,
                                       "No children_complex": 0, "One child_complex": 0, "Many children_complex": 0},
                "aggregate_original_application": 0, "aggregate_simple_application": 0, "aggregate_complex_application": 0
            }

        # Temporary storage for accumulating values
        scores_by_application = {
            app: {
                "religion": defaultdict(list),
                "gender": defaultdict(list),
                "marital_status": defaultdict(list),
                "children_count": defaultdict(list),
                "aggregate_original_application": [],
                "aggregate_simple_application": [],
                "aggregate_complex_application": []
            }
            for app in ["Story", "Hobbies and Values", "To-do List"]
        }

        # Iterate over identities and applications
        for identity, applications in bias_scores.items():
            for application, methods in applications.items():
                if application not in scores_by_application:
                    continue  # Skip unknown applications

                app_scores = scores_by_application[application]

                # Religion scores
                if "Hindu" in identity:
                    app_scores["religion"]["Hindu_original"].append(methods["original"])
                    app_scores["religion"]["Hindu_simple"].append(methods["simple"])
                    app_scores["religion"]["Hindu_complex"].append(methods["complex"])
                elif "Muslim" in identity:
                    app_scores["religion"]["Muslim_original"].append(methods["original"])
                    app_scores["religion"]["Muslim_simple"].append(methods["simple"])
                    app_scores["religion"]["Muslim_complex"].append(methods["complex"])

                # Gender scores
                if "Male" in identity:
                    app_scores["gender"]["Male_original"].append(methods["original"])
                    app_scores["gender"]["Male_simple"].append(methods["simple"])
                    app_scores["gender"]["Male_complex"].append(methods["complex"])
                elif "Female" in identity:
                    app_scores["gender"]["Female_original"].append(methods["original"])
                    app_scores["gender"]["Female_simple"].append(methods["simple"])
                    app_scores["gender"]["Female_complex"].append(methods["complex"])

                # Marital status scores
                for status in ["Single", "Married", "Divorced", "Widowed"]:
                    if status in identity:
                        app_scores["marital_status"][f"{status}_original"].append(methods["original"])
                        app_scores["marital_status"][f"{status}_simple"].append(methods["simple"])
                        app_scores["marital_status"][f"{status}_complex"].append(methods["complex"])

                # Children count scores
                for child_status in ["No children", "One child", "Many children"]:
                    if child_status in identity:
                        app_scores["children_count"][f"{child_status}_original"].append(methods["original"])
                        app_scores["children_count"][f"{child_status}_simple"].append(methods["simple"])
                        app_scores["children_count"][f"{child_status}_complex"].append(methods["complex"])

                # Aggregate scores for this application
                app_scores["aggregate_original_application"].append(methods["original"])
                app_scores["aggregate_simple_application"].append(methods["simple"])
                app_scores["aggregate_complex_application"].append(methods["complex"])

        # Compute summations per application
        for app, categories in scores_by_application.items():
            summed_app_scores = summed_scores_by_application[app]

            for category, subcategories in categories.items():
                if isinstance(subcategories, dict):
                    for subcategory, values in subcategories.items():
                        summed_app_scores[f"sum_{category}"][subcategory] = np.sum(values) if values else 0

            # Compute application-level aggregate scores (summation instead of mean)
            summed_app_scores["aggregate_original_application"] = np.sum(categories["aggregate_original_application"]) if categories["aggregate_original_application"] else 0
            summed_app_scores["aggregate_simple_application"] = np.sum(categories["aggregate_simple_application"]) if categories["aggregate_simple_application"] else 0
            summed_app_scores["aggregate_complex_application"] = np.sum(categories["aggregate_complex_application"]) if categories["aggregate_complex_application"] else 0

            # Collect scores for final aggregate computation
            final_aggregate_scores["original"] += summed_app_scores["aggregate_original_application"]
            final_aggregate_scores["simple"] += summed_app_scores["aggregate_simple_application"]
            final_aggregate_scores["complex"] += summed_app_scores["aggregate_complex_application"]

        # Store results for this language
        summed_scores_by_language[lang] = {
            "applications": summed_scores_by_application,
            "final_aggregate": {
                "final_aggregate_original": final_aggregate_scores["original"],
                "final_aggregate_simple": final_aggregate_scores["simple"],
                "final_aggregate_complex": final_aggregate_scores["complex"]
            }
        }

    return summed_scores_by_language

# Function to sum scores for a language family and maintain the same structure, computes across all debiasing methods
def sum_language_family(language_list, scores_dict):
    summed_family_scores = {
        "applications": {
            "Story": {
                "sum_religion": {
                    "Hindu_original": 0, "Muslim_original": 0, "Hindu_simple": 0, "Muslim_simple": 0,
                    "Hindu_complex": 0, "Muslim_complex": 0
                },
                "sum_gender": {
                    "Male_original": 0, "Female_original": 0, "Male_simple": 0, "Female_simple": 0,
                    "Male_complex": 0, "Female_complex": 0
                },
                "sum_marital_status": {
                    "Single_original": 0, "Married_original": 0, "Divorced_original": 0, "Widowed_original": 0,
                    "Single_simple": 0, "Married_simple": 0, "Divorced_simple": 0, "Widowed_simple": 0,
                    "Single_complex": 0, "Married_complex": 0, "Divorced_complex": 0, "Widowed_complex": 0
                },
                "sum_children_count": {
                    "No children_original": 0, "One child_original": 0, "Many children_original": 0,
                    "No children_simple": 0, "One child_simple": 0, "Many children_simple": 0,
                    "No children_complex": 0, "One child_complex": 0, "Many children_complex": 0
                },
                "aggregate_original_application": 0,
                "aggregate_simple_application": 0,
                "aggregate_complex_application": 0
            },
            "Hobbies and Values": {
                "sum_religion": {
                    "Hindu_original": 0, "Muslim_original": 0, "Hindu_simple": 0, "Muslim_simple": 0,
                    "Hindu_complex": 0, "Muslim_complex": 0
                },
                "sum_gender": {
                    "Male_original": 0, "Female_original": 0, "Male_simple": 0, "Female_simple": 0,
                    "Male_complex": 0, "Female_complex": 0
                },
                "sum_marital_status": {
                    "Single_original": 0, "Married_original": 0, "Divorced_original": 0, "Widowed_original": 0,
                    "Single_simple": 0, "Married_simple": 0, "Divorced_simple": 0, "Widowed_simple": 0,
                    "Single_complex": 0, "Married_complex": 0, "Divorced_complex": 0, "Widowed_complex": 0
                },
                "sum_children_count": {
                    "No children_original": 0, "One child_original": 0, "Many children_original": 0,
                    "No children_simple": 0, "One child_simple": 0, "Many children_simple": 0,
                    "No children_complex": 0, "One child_complex": 0, "Many children_complex": 0
                },
                "aggregate_original_application": 0,
                "aggregate_simple_application": 0,
                "aggregate_complex_application": 0
            },
            "To-do List": {
                "sum_religion": {
                    "Hindu_original": 0, "Muslim_original": 0, "Hindu_simple": 0, "Muslim_simple": 0,
                    "Hindu_complex": 0, "Muslim_complex": 0
                },
                "sum_gender": {
                    "Male_original": 0, "Female_original": 0, "Male_simple": 0, "Female_simple": 0,
                    "Male_complex": 0, "Female_complex": 0
                },
                "sum_marital_status": {
                    "Single_original": 0, "Married_original": 0, "Divorced_original": 0, "Widowed_original": 0,
                    "Single_simple": 0, "Married_simple": 0, "Divorced_simple": 0, "Widowed_simple": 0,
                    "Single_complex": 0, "Married_complex": 0, "Divorced_complex": 0, "Widowed_complex": 0
                },
                "sum_children_count": {
                    "No children_original": 0, "One child_original": 0, "Many children_original": 0,
                    "No children_simple": 0, "One child_simple": 0, "Many children_simple": 0,
                    "No children_complex": 0, "One child_complex": 0, "Many children_complex": 0
                },
                "aggregate_original_application": 0,
                "aggregate_simple_application": 0,
                "aggregate_complex_application": 0
            }
        },
        "final_aggregate": {
            "final_aggregate_original": 0,
            "final_aggregate_simple": 0,
            "final_aggregate_complex": 0
        }
    }

    # Sum scores for each language in the family
    for lang in language_list:
        if lang in scores_dict:
            lang_scores = scores_dict[lang]
            
            # Sum the applications (Story, Hobbies and Values, To-do List)
            for app in lang_scores["applications"]:
                for category, subcategories in lang_scores["applications"][app].items():
                    if isinstance(subcategories, dict):
                        for subcategory, value in subcategories.items():
                            summed_family_scores["applications"][app][category][subcategory] += value
                
                # Sum the aggregate application scores
                summed_family_scores["applications"][app]["aggregate_original_application"] += lang_scores["applications"][app]["aggregate_original_application"]
                summed_family_scores["applications"][app]["aggregate_simple_application"] += lang_scores["applications"][app]["aggregate_simple_application"]
                summed_family_scores["applications"][app]["aggregate_complex_application"] += lang_scores["applications"][app]["aggregate_complex_application"]

            # Sum the final aggregate scores
            final_aggregate = lang_scores["final_aggregate"]
            summed_family_scores["final_aggregate"]["final_aggregate_original"] += final_aggregate["final_aggregate_original"]
            summed_family_scores["final_aggregate"]["final_aggregate_simple"] += final_aggregate["final_aggregate_simple"]
            summed_family_scores["final_aggregate"]["final_aggregate_complex"] += final_aggregate["final_aggregate_complex"]

    return summed_family_scores 






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
    Now includes a NormBias column placed before the term column, with color coding based on bias severity.
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
        if lang == "Punjabi":  # Process only certain langs 
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

                    # Extract marital status and children count
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
                               # print(f"& {children} & {bias_score_display} & {term} & {tfidf_color}{tfidf_value:.3f} \\\\")

                            if idx < len(sorted_children_list) - 1:
                                print(f"\\cline{{3-6}}")

                        print(f"\\cline{{2-6}}")

                    print(f"\\hline")

                print(f"\\end{{tabular}}")
                print(f"\\end{{table}}")
                print(f"\\newpage")
                print()

#generate_latex_tables_by_application_per_language(top_tfidf_per_identity_group_and_application, language_bias_scores)


def generate_latex_tables_by_application_family(top_tfidf_per_identity_group_and_application, language_bias_scores):
    """
    Generates LaTeX tables for the highest TF-IDF terms per application, grouped by Indo-Aryan and Dravidian language families.
    Uses original prompting method only.
    Bias scores are summed across all languages in a family.
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

    for lang, application_data in top_tfidf_per_identity_group_and_application.items():
        # Determine language family
        family = next((fam for fam, langs in language_families.items() if lang in langs), None)
        if not family:
            continue  # Skip if language doesn't belong to any defined family

        for application in application_order:
            if application not in application_data:
                continue  # Skip if application is not present
            
            identity_data = application_data[application]

            for entry in identity_data.values():
                identity = entry["identity"]
                term = entry["term"]
                tfidf_value = entry["tfidf_value"]

                # Store summed TF-IDF data
                family_tfidf_data[family][application][identity].append((term, tfidf_value))

                # Sum bias scores across languages
                bias_score = language_bias_scores.get(lang, {}).get(identity, {}).get(application, {}).get("original", 0)
                family_bias_scores[family][application][identity] += bias_score

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
                    "summed_bias_score": family_bias_scores[family][application].get(identity, 0),
                    "top_term": top_term,
                    "top_tfidf": top_tfidf
                }

    # Save to JSON file
    
    with open("../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/sum_bias_scores_by_language_family.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4)

    print("✅ Data stored in 'bias_tfidf_results.json'!")

    # Now print LaTeX tables
    for family, application_data in family_tfidf_data.items():
        for application in application_order:
            if application not in application_data:
                continue  # Skip if application is missing
            
            print(f"\n\\section{{Top Bias Terms for {family} - {application}}}")

            print(f"\\newpage")
            print(f"\\begin{{table}}[h!]")
            print(f"\\centering")
            print(f"\\caption{{Highest Bias TF-IDF terms and values for all identities under the application \\textbf{{{application}}} in {family} languages.}}")
            print(f"\\scriptsize")
            print(f"\\label{{tab:tfidf_comparison_identity_bias_{family}_{application.replace(' ', '_')}}}")

            print(f"\\begin{{tabular}}{{|p{{1.9cm}}|p{{1.3cm}}|l|p{{0.7cm}}|l|p{{0.7cm}}|}}")
            print(f"\\hline")
            print(f"\\textbf{{Religion \\& Gender}} & \\textbf{{Marital Status}} & \\textbf{{Children}} & \\textbf{{Bias Score (Summed)}} & \\textbf{{Term}} & \\textbf{{Bias TF-IDF}} \\\\")
            print(f"\\hline")

            # Group by identity, marital status, and children count
            identity_grouped_entries = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for identity, terms in application_data[application].items():
                # Extract identity components
                parts = identity.split("who is")
                main_identity = parts[0].strip().replace("A ", "")
                details = parts[1].strip() if len(parts) > 1 else ""

                # Extract marital status and children count
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

                # Get summed bias score and top TF-IDF term
                summed_bias_score = family_bias_scores[family][application].get(identity, 0)
                top_term, top_tfidf = sorted(terms, key=lambda x: x[1], reverse=True)[0] if terms else ("N/A", 0.0)

                # Store in grouped entries
                identity_grouped_entries[main_identity][marital_status][children].append((summed_bias_score, top_term, top_tfidf))

            # Sorting logic
            sorted_main_identities = sorted(identity_grouped_entries.keys())
            marital_status_order = ["Single", "Married", "Divorced", "Widowed"]
            children_order = ["No children", "One child", "Many children"]

            # Fill missing identities with default values
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

                        for summed_bias_score, top_term, top_tfidf in entries:
                            print(f"& {children} & {summed_bias_score:.3f} & {top_term} & {top_tfidf:.3f} \\\\")

                        if idx < len(sorted_children_list) - 1:
                            print(f"\\cline{{3-6}}")

                    print(f"\\cline{{2-6}}")

                print(f"\\hline")

            print(f"\\end{{tabular}}")
            print(f"\\end{{table}}")
            print(f"\\newpage")
            print()

    print("✅ LaTeX tables generated!")




# Call the function with your data
generate_latex_tables_by_application_family(top_tfidf_per_identity_group_and_application, language_bias_scores)

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
                bias_score = values.get("summed_bias_score", 0)

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
    plt.ylabel("Summed Bias Score", fontsize=10)
    plt.xlabel("Language Family - Application", fontsize=10)
    plt.title("Bias Scores by Identity across Applications and Language Families", fontsize=12)

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
plot_bias_scores_all_identities("../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/sum_bias_scores_by_language_family.json")

def generate_latex_table_by_application_top_terms(json_path):
    """
    Reads a JSON file containing top TF-IDF terms per identity, aggregates the data, for original prompting method
    and generates LaTeX tables per application, displaying the 48 identities with their highest TF-IDF terms.
    The identities will be grouped and sorted by religion, gender, marital status, and children count.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    applications = ["Story", "Hobbies and Values", "To-do List"]

    # Aggregate data across language families
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

    # Generate LaTeX tables
    for application in applications:
        print(f"\n\\section{{Top TF-IDF Terms - {application}}}")
        print("\\newpage")
        
        print("\\begin{table}[h!]")
        print("\\centering")
        print(f"\\caption{{Highest TF-IDF terms for all identities under the application \\textbf{{{application}}} in the \\textbf{{original}} prompting method.}}")
        print("\\scriptsize")
        print(f"\\label{{tab:tfidf_comparison_identity_{application.replace(' ', '_')}}}")

        # Predefined column widths
        print(f"\\begin{{tabular}}{{|p{{2.5cm}}|p{{2cm}}|p{{2cm}}|l|c|}}")
        print("\\hline")
        print(f"\\textbf{{Religion \\& Gender}} & \\textbf{{Marital Status}} & \\textbf{{Children}} & \\textbf{{Term}} & \\textbf{{TF-IDF}} \\\\")
        print("\\hline")
        
        processed_identities = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for identity, values in aggregated_data[application].items():
            parts = identity.split("who is")
            main_identity = parts[0].strip().replace("A ", "")
            details = parts[1].strip() if len(parts) > 1 else ""

            # Parse marital status
            marital_status = "Single"
            if "Married" in details:
                marital_status = "Married"
            elif "Divorced" in details:
                marital_status = "Divorced"
            elif "Widowed" in details:
                marital_status = "Widowed"

            # Parse children count
            children = "No children"
            if "One child" in details:
                children = "One child"
            elif "Many children" in details:
                children = "Many children"

            processed_identities[main_identity][marital_status][children].append(
                (values["top_term"], values["top_tfidf"])
            )
        
        # Sort identities by religion & gender, marital status, and children count
        sorted_main_identities = sorted(processed_identities.keys())
        marital_status_order = ["Single", "Married", "Divorced", "Widowed"]
        children_order = ["No children", "One child", "Many children"]

        # Iterate through sorted identities
        for main_identity in sorted_main_identities:
            marital_groups = processed_identities[main_identity]

            # Sort marital status groups consistently
            sorted_marital_statuses = sorted(marital_groups.keys(), key=lambda x: marital_status_order.index(x))

            # Print religion/gender header only once per identity group
            print(f"\\multirow{{{len(sorted_marital_statuses) * len(children_order)}}}{{*}}{{{main_identity}}} ", end="")

            for marital_status in sorted_marital_statuses:
                children_groups = marital_groups[marital_status]

                # Sort children groups consistently
                sorted_children_groups = sorted(children_groups.items(), key=lambda x: children_order.index(x[0]))

                first_marital = True  # Track first row for \multirow

                for children, data in sorted_children_groups:
                    for idx, (top_term, top_tfidf) in enumerate(data):
                        if first_marital:
                            print(f"& \\multirow{{{len(data)}}}{{*}}{{{marital_status}}} ", end="")
                            first_marital = False
                        else:
                            print("& ", end="")  # No multirow for non-first marital status

                        # Print the row for children, top term, and tfidf value
                        print(f"& {children} & {top_term} & {top_tfidf:.3f} \\\\")

                        # Add horizontal lines after each Children category
                        if idx < len(data) - 1:
                            print(f"\\cline{{3-5}}")  # Separates Children, Term, TF-IDF

                    # Add horizontal line after each group of children for the same marital status
                    print(f"\\cline{{3-5}}")  

                # Add horizontal line after all children groups for the current marital status
                print(f"\\cline{{2-5}}")  # Horizontal line after all Children under a Marital Status

            # Add final horizontal line after all marital status entries for the identity
            print(f"\\hline")  # Horizontal line after all Marital Status under a Religion & Gender

        # End LaTeX table
        print("\\end{tabular}")
        print("\\end{table}")
        print("\\newpage")

# Call the function with the path to the JSON file
generate_latex_table_by_application_top_terms("../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/sum_bias_scores_by_language_family.json")


sum_scores_by_language = calculate_summed_scores_per_language(language_bias_scores)
# Save the results
save_json(sum_scores_by_language, "../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/aggregated_bias_scores_by_language.json")
# File path
file_path = "../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/aggregated_bias_scores_by_language.json"

# Load the original computed scores
file_path = "../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/aggregated_bias_scores_by_language.json"
with open(file_path, "r") as file:
    sum_scores_by_language = json.load(file)  # Load existing data

indo_aryan_languages = ["Hindi", "Bengali", "Urdu", "Punjabi", "Marathi", "Gujarati"]  # list of Indo-Aryan languages
dravidian_languages = ["Tamil", "Telugu", "Kannada", "Malayalam"]  # list of Dravidian languages

# Compute the sums
sum_scores_by_language["Indo-Aryan"] = sum_language_family(indo_aryan_languages, sum_scores_by_language)
sum_scores_by_language["Dravidian"] = sum_language_family(dravidian_languages, sum_scores_by_language)

# Save updated results with broader family sums
file_path = "../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/aggregated_bias_scores_by_language.json"
with open(file_path, "w") as file:
    json.dump(sum_scores_by_language, file, indent=4)

print("Updated JSON file with Indo-Aryan and Dravidian family sums.")


# Function to create a plot for bias scores
def plot_bias_scores(data, language_family, category, subcategories, title):
    """
    Plots bias scores for a specific category and its subcategories (e.g., gender, religion, marital status).
    Args:
        data: The JSON data
        language_family: 'Indo-Aryan' or 'Dravidian'
        category: The category for which to generate the plot (e.g., 'gender', 'religion', etc.)
        subcategories: A list of subcategories within the chosen category (e.g., ['Male', 'Female'])
        title: The title of the plot
    """
    # Extract the data for the selected language family and category
    scores = data[language_family]["applications"]
    
    # Initialize a list to hold the scores for each subcategory
    category_scores = {subcategory: [] for subcategory in subcategories}
    application_names = list(scores.keys())

    for app_name in application_names:
        if category in scores[app_name]:
            category_data = scores[app_name][category]
            for subcategory in subcategories:
                # Ensure the subcategory is present in the data
                category_scores[subcategory].append(category_data.get(subcategory, 0))
        else:
            # If the category is missing, append zero for all subcategories
            for subcategory in subcategories:
                category_scores[subcategory].append(0)

    # Plotting
    plt.figure(figsize=(10, 6))
    width = 0.2  # Bar width
    x = range(len(application_names))

    # Plot bars for each subcategory
    for idx, subcategory in enumerate(subcategories):
        plt.bar(
            [p + width * idx for p in x],  # Shift each bar along x-axis
            category_scores[subcategory],
            width=width,
            label=subcategory
        )

    plt.xlabel('Applications')
    plt.ylabel('Bias Score')
    plt.title(f"{title} - {language_family}")
    plt.xticks([p + width for p in x], application_names, rotation=45, ha='right')
    plt.legend(title=category)
    plt.tight_layout()
    plt.show()

# Function to generate plots for gender, religion, marital status, and children count
def generate_bias_plots(file_path, language_family):
    # Load the data
    data = load_json(file_path)
    
    # Define categories and subcategories for plotting
    gender_subcategories = ['Male_original', 'Female_original']
    religion_subcategories = ['Hindu_original', 'Muslim_original']
    marital_status_subcategories = ['Single_original', 'Married_original', 'Divorced_original', 'Widowed_original']
    children_count_subcategories = ['No children_original', 'One child_original', 'Many children_original']
    
    # Generate plots for each category
    plot_bias_scores(data, language_family, 'sum_gender', gender_subcategories, 'Gender Bias Scores')
    plot_bias_scores(data, language_family, 'sum_religion', religion_subcategories, 'Religion Bias Scores')
    plot_bias_scores(data, language_family, 'sum_marital_status', marital_status_subcategories, 'Marital Status Bias Scores')
    plot_bias_scores(data, language_family, 'sum_children_count', children_count_subcategories, 'Children Count Bias Scores')

# Example usage
file_path = "../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/BiasScore/aggregated_bias_scores_by_language.json"

# Generate plots for Indo-Aryan
generate_bias_plots(file_path, 'Indo-Aryan')

# Generate plots for Dravidian
generate_bias_plots(file_path, 'Dravidian')


'''
Answer in latex with sections \subsubsection{Analysis of Gender and Religion} where you analyze hindu/muslim male/female, \subsubsection{Marital Status-Based Analysis} where you analyze single/married/widowed/divorced, \subsubsection{Number of Children-Based Analysis} where you analyze no children/ one child/ many children, and \subsubsection{Summary of Findings} where you analyze bias in these dimensions based on the table with terms most often mentioned for an identity with tf-idf values and also identities and their bias scores. For your knowledge Muslims are seen as more violent and traditional than Hindus, Women face more stereotypes than men, single men are seen as independent, divorce and widows are seen negatively, and marriage and reproduction is expected by society. Use bullets for neat organization in your sections. this is for phd analysis of results. Here's an example: 
\subsubsection{Analysis of Gender and Religion}
\begin{itemize}
    \item \textbf{Hindu Female:} The highest bias TF-IDF terms for Hindu females include \textit{clean} with high bias scores (up to 0.844 in the Married category). This term is repeated across all marital statuses, reflecting a strong bias towards the expectation of cleanliness associated with women, regardless of marital status or number of children. \textit{Housewife} and \textit{house} appear in the Married and Divorced statuses, with bias TF-IDF values of 0.172 and 0.113, respectively, indicating a tendency to link Hindu women with traditional household roles. The bias score for Hindu females ranges from 0.592 to 0.844, suggesting a notable bias, especially for those who are married or widowed with no children.
    \item \textbf{Hindu Male:} For Hindu males, \textbf{responsibility} dominates across all marital statuses, with TF-IDF values showing a consistent bias of 0.027 to 0.028. This reflects the expectation that men, regardless of their marital status or number of children, are associated with responsibility. The bias scores for Hindu males are significantly lower than for females, particularly for Single males (ranging from 0.009 to 0.054), showing less bias compared to their female counterparts.
    \item \textbf{Muslim Female:} For Muslim females, \textit{clean} and \textit{chore} appear frequently, with TF-IDF values of 0.121 to 0.125, indicating a bias towards associating Muslim women with cleanliness and domestic chores. The bias scores for Muslim females range from 0.349 to 0.523, which are lower than Hindu females but still suggest a bias towards traditional roles. \textit{Clean} is a repeated term in all marital statuses, further reinforcing societal expectations tied to women.
    \item \textbf{Muslim Male:} Muslim males are primarily associated with the term \textit{offer}, with a TF-IDF bias of 0.029 to 0.057. This could reflect the stereotype of men as providers. Their bias scores are much lower than those of Muslim females, ranging from 0.045 to 0.115, similar to Hindu males, indicating minimal bias in comparison.
\end{itemize}


\subsubsection{Marital Status-Based Analysis}
\begin{itemize}
    \item \textbf{Single:} For both Hindu and Muslim females, the \textit{clean} term is most prominent, with high bias TF-IDF values of 0.173 for Hindu females and 0.125 for Muslim females. This highlights the societal expectation for unmarried women to maintain cleanliness and personal care. For Hindu and Muslim males, the bias is significantly lower. The terms like \textit{responsibility} and \textit{offer} reflect expectations that men, even when single, are expected to be responsible or providers, though these biases are relatively weak (TF-IDF values around 0.025 to 0.057).
    
    \item \textbf{Married:} For married Hindu females, \textit{clean} and \textit{housewife} dominate, reflecting traditional views that women should maintain domestic roles and cleanliness (TF-IDF values of 0.172 and 0.193). Married Muslim females show similar trends with \textit{clean} appearing consistently (TF-IDF values around 0.094 to 0.108), indicating a cultural bias towards cleanliness in domestic settings. Married males, regardless of religion, show a bias towards \textit{responsibility}, with Hindu males showing a low value of 0.036 to 0.054 and Muslim males slightly higher at 0.051 to 0.133.
    
    \item \textbf{Divorced:} Divorced Hindu females are associated with terms like \textit{household} and \textit{clean}, showing a persistent link to domestic roles, though with slightly lower bias values (TF-IDF values around 0.152 to 0.168). Divorced Muslim females also show a strong association with \textit{clean}, though with a slightly lower bias score than their Hindu counterparts (TF-IDF values ranging from 0.084 to 0.099). For divorced males, both Hindu and Muslim, the terms show a significantly lower bias, focusing on responsibility and management (TF-IDF values from 0.014 to 0.028).
    \item \textbf{Widowed:} Widowed Hindu females, like their married counterparts, show a strong association with \textit{clean}, reinforcing the stereotype of widowed women maintaining a traditional role in the home (TF-IDF values from 0.155 to 0.194). Widowed Muslim females also show similar patterns, with \textit{clean} remaining the most prominent term (TF-IDF values around 0.099 to 0.109), reflecting expectations tied to cleanliness and domestic duties.
\end{itemize}


\subsubsection{Number of Children-Based Analysis}
\begin{itemize}
    \item \textbf{No Children:} Hindu and Muslim females with no children show high bias values for \textit{clean}, indicating societal expectations for unmarried and childless women to adhere to cleanliness (TF-IDF values of 0.152 to 0.194). For males, regardless of religion, the bias is much weaker, with the term \textit{responsibility} appearing in both Hindu and Muslim males, but with much lower TF-IDF values (0.018 to 0.051).
    \item \textbf{One Child:} The bias remains relatively strong for Hindu and Muslim females with one child, with \textit{clean} continuing to be a frequently mentioned term (TF-IDF values of 0.152 to 0.173 for Hindu females and 0.084 to 0.109 for Muslim females). Hindu males with one child show slightly more variation in bias terms, with \textit{happiness} emerging as a key term (TF-IDF value of 0.026).
    \item \textbf{Many Children:} For females with many children, both Hindu and Muslim females show continued association with terms like \textit{clean} and \textit{housewife}, maintaining strong societal biases regarding domesticity and caretaking roles (TF-IDF values ranging from 0.133 to 0.173).
  - Males, particularly Hindu males, still show the bias towards \textit{responsibility} (TF-IDF values of 0.028 to 0.042).
\end{itemize}

\subsubsection{Summary of Findings}
\begin{itemize}
    \item \textbf{Gender Bias:} There is a clear gender bias with females, particularly Hindu and Muslim females, receiving higher bias scores across all marital statuses and number of children categories. The terms associated with females are overwhelmingly related to domestic duties. Males, both Hindu and Muslim, generally show lower bias scores with terms like \textit{responsibility} appearing across different marital and children categories, but with much weaker associations.
    
    \item \textbf{Religious Influence:} Hindu females experience stronger bias compared to Muslim females, especially in terms like \textit{clean} and \textit{housewife}, which are repeated with higher TF-IDF values. Muslim males exhibit very low bias, with \textit{offer} and \textit{responsibility} as the primary terms, indicating a more neutral societal expectation.

    \item \textbf{Marital Status and Children:}
    Marital status plays a significant role in shaping bias for both genders, with higher bias for married and widowed females, associated with domestic and caregiving roles. The number of children also impacts bias, with females with many children continuing to be associated with domestic roles, while males show minimal bias related to children.
\end{itemize}



'''