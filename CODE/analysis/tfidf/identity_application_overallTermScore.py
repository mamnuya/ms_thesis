'''
1. Computing Term Scores (compute_term_score function)
Purpose: This function calculates the overall term score for each identity-application pair, 
considering the TF-IDF scores across different methods ("original", "complex", "simple").

How: It sums the TF-IDF scores for each term within each identity-application-method grouping.
Output: A dictionary (term_scores) containing summed term scores for each (identity, application, method) pair.

3. Aggregating Term Scores Across Languages (aggregate_term_scores_across_languages function)
Purpose: This function computes summary statistics
of term scores across all languages.

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

'''
Answer in latex with sections \subsubsection{Analysis of Gender and Religion} where you analyze hindu/muslim male/female, \subsubsection{Marital Status-Based Analysis} where you analyze single/married/widowed/divorced, \subsubsection{Number of Children-Based Analysis} where you analyze no children/ one child/ many children, and \subsubsection{Summary of Findings} where you analyze bias in these dimensions based on the table with terms most often mentioned for an identity with tf-idf values. For your knowledge Muslims are seen as more violent and traditional than Hindus, Women face more stereotypes than men, single men are seen as independent, divorce and widows are seen negatively, and marriage and reproduction is expected by society. Use bullets for neat organization in your sections. this is for phd analysis of results. Here's an example for overall term analysis, but I expect you to incorporate religion based analysis within the gender based analysis bullets under one subheading: \subsubsection{Analysis of Gender and Religion}

\subsubsection{Analysis of Gender and Religion}
\begin{itemize}
    \item \textbf{Hindu Female}: The most prominent terms for Hindu females include \textit{work} (TF-IDF = 0.048–0.051), \textit{house} (0.056–0.063), and \textit{name} (0.043–0.072). The term \textit{admit} appears notably for those with many children (0.089), which may suggest a narrative involving education or confession, depending on the context. The frequent presence of \textit{house} aligns with traditional gender roles associating women with domestic spaces.
    \item \textbf{Hindu Male}: The most frequent term is \textit{wife} (0.054–0.088), appearing across all marital statuses, including single men with children, reinforcing a strong emphasis on spousal relationships. Other notable terms include \textit{friend} (0.051) and \textit{several} (0.057), suggesting broader social narratives for men compared to women. Hindu males display a stronger association with \textit{wife} (TF-IDF up to 0.088), whereas Muslim males have similar patterns but with slightly higher TF-IDF values for \textit{wife} (up to 0.091), possibly suggesting stronger spousal-centric storytelling in Muslim male narratives.
    \item \textbf{Muslim Female}: Hindu and Muslim women share common frequent terms like \textit{house}, \textit{work}, and \textit{name}, indicating similar gendered narratives. Similar to Hindu females, the terms \textit{work} (0.041–0.059) and \textit{house} (0.051–0.059) are frequently found. The term \textit{mother} (0.066) is more common among those with many children, emphasizing maternal identity in Muslim female narratives.
    \item \textbf{Muslim Male}: The term \textit{wife} (0.055–0.091) appears frequently, similar to Hindu males. The term \textit{two} is particularly prevalent among Muslim men across all categories (TF-IDF up to 0.087), which does not appear as frequently for Hindu men. Additionally, the term \textit{two} (0.075–0.087) appears often, potentially reflecting family size, numerical references, or a cultural emphasis on duality (e.g., two wives, two sons).
\end{itemize}

\subsubsection{Marital Status-Based Analysis}
\begin{itemize}
    \item Married men, regardless of religion, frequently feature the term \textit{wife} with high TF-IDF values (0.054–0.088), reinforcing a strong narrative focus on marriage.
    \item Single Hindu females frequently feature \textit{two} (TF-IDF = 0.093), while single Muslim females also have \textit{two} (0.070), potentially reflecting a common storytelling pattern about pairs, relationships, or decision-making.
    \item Divorced and widowed women have \textit{work} and \textit{house} as frequent terms, reinforcing themes of domesticity and resilience.
\end{itemize}

\subsubsection{Children-Based Analysis}
\begin{itemize}
    \item Across all identities, the presence of children influences term frequency. Women with many children frequently feature \textit{house} and \textit{mother}, reinforcing maternal roles.
    \item Men, regardless of marital status, frequently have \textit{wife} as a dominant term, with Muslim males showing the highest TF-IDF values.
    \item The term \textit{admit} appears for Hindu females with many children (TF-IDF = 0.089) and Muslim females with one child (0.061), possibly indicating themes of education, confession, or acknowledgment in storytelling.
\end{itemize}

\subsubsection{Summary of Findings}
\begin{itemize}
    \item Gendered patterns emerge strongly, with women frequently associated with \textit{house} and \textit{work}, while men are more frequently linked to \textit{wife}.
    \item Hindu and Muslim men exhibit similar trends, but Muslim men display slightly stronger emphasis on spousal relationships and numerical references (\textit{two}).
    \item Married and widowed women, especially those with children, frequently feature terms reinforcing domestic and maternal roles.
    \item The presence of children influences term frequency, with stronger associations between motherhood and household-related terms for women.
\end{itemize}


'''