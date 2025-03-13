# Definition: Compute TF-IDF
# 
# Let:
# - t: bias term from the lexicon
# - d: identity group's data from a given application (document)
# 
# 1. **Term Frequency (TF)**: 
#    The term frequency (TF) for a given bias term t in document d is calculated as the ratio of 
#    occurrences of t in d to the total number of terms in d:
#    
#    TF(t, d) = (occurrences of t in d) / (total terms in d)
#    
# 2. **Document Frequency (DF)**: 
#    The document frequency (DF) of a bias term t is the number of identity groups (documents) 
#    in which the term t appears.
# 
# 3. **Inverse Document Frequency (IDF)**: 
#    The IDF for each term t is computed as:
#    
#    IDF(t) = log((N + 1) / (DF(t) + 1)) + 1
#    
#    Where:
#    - N is the total number of identity groups (documents)
#    - DF(t) is the number of documents containing term t
#    - The "+1" in both the numerator and denominator is for smoothing to avoid division by zero.
#    
# 4. **TF-IDF**: 
#    The TF-IDF score for each bias term t in document d is calculated as:
#    
#    TF-IDF(t, d) = TF(t, d) * IDF(t)
#    
# This method calculates the term frequency (TF) for each bias term within the context of a specific
# identity group and application, adjusted for the frequency of the term across all identity groups 
# (IDF). The resulting TF-IDF score reflects the importance of each bias term in the context of the 
# data, weighted by its relative rarity across all documents.

import json
import math
from collections import defaultdict

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_tf(bias_term_counts, word_counts):
    """Compute Term Frequency (TF) for each bias term per (identity, application) group."""
    tf_scores = defaultdict(lambda: defaultdict(lambda: {"original": {}, "complex": {}, "simple": {}}))

    for (identity, application), counts in bias_term_counts.items():
        for method in ["original", "complex", "simple"]:
            total_words = word_counts[(identity, application)][method]
            for term, count in counts[method].items():
                tf_scores[identity][application][method][term] = count / total_words if total_words > 0 else 0

    return tf_scores

def compute_idf(bias_term_counts, total_documents):
    """Compute Inverse Document Frequency (IDF) for each bias term."""
    idf_scores = defaultdict(float)
    document_frequencies = defaultdict(int)

    # Count the number of (identity, application) groups each term appears in
    for identity_application_counts in bias_term_counts.values():
        seen_terms = set()  # Ensures each term is counted only once per (identity, application) pair
        for method in ["original", "complex", "simple"]:
            seen_terms.update(identity_application_counts[method].keys())
        for term in seen_terms:
            document_frequencies[term] += 1  

    # Compute IDF with log-scaling
    for term, doc_count in document_frequencies.items():
        idf_scores[term] = math.log((total_documents + 1) / (doc_count + 1)) + 1  # Smoothing factor

    return idf_scores

def compute_tf_idf(tf_scores, idf_scores):
    """Compute TF-IDF by multiplying TF by IDF."""
    tf_idf_scores = defaultdict(lambda: defaultdict(lambda: {"original": {}, "complex": {}, "simple": {}}))

    for identity, applications in tf_scores.items():
        for application, methods in applications.items():
            for method in ["original", "complex", "simple"]:
                for term, tf_value in methods[method].items():
                    tf_idf_scores[identity][application][method][term] = tf_value * idf_scores[term]

    return tf_idf_scores

def count_bias_terms_by_identity_application(dataset, bias_lexicon):
    """Count occurrences of bias terms in processed text outputs, grouped by (identity, application)."""
    bias_term_counts = defaultdict(lambda: {"original": defaultdict(int), "complex": defaultdict(int), "simple": defaultdict(int)})
    word_counts = defaultdict(lambda: {"original": 0, "complex": 0, "simple": 0})

    for entry in dataset:
        identity = entry["identity"]
        application = entry["application"]
        identity_application = (identity, application)

        original_words = entry["processed_translated_generated_output"]
        complex_words = entry["complex_processed_translated_debiased_output"]
        simple_words = entry["simple_processed_translated_debiased_output"]

        for category, bias_terms in bias_lexicon.items():
            if category in identity:
                bias_terms_set = set(bias_terms)

                if application == "Hobbies and Values":
                    bias_terms_set.discard("value")  # Remove "value" from the bias terms set when application is Hobbies and Values


                for word in original_words:
                    if word in bias_terms_set:
                        bias_term_counts[identity_application]["original"][word] += 1
                for word in complex_words:
                    if word in bias_terms_set:
                        bias_term_counts[identity_application]["complex"][word] += 1
                for word in simple_words:
                    if word in bias_terms_set:
                        bias_term_counts[identity_application]["simple"][word] += 1

        word_counts[identity_application]["original"] += len(original_words)
        word_counts[identity_application]["complex"] += len(complex_words)
        word_counts[identity_application]["simple"] += len(simple_words)

    return bias_term_counts, word_counts

def save_json(data, filepath):
    """Save JSON data to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Load dataset JSON and compute TF-IDF
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

for lang in languages: 
    dataset_path = f"../../../data/complex_and_simple_debiaspromptsQs/cleaned_tokenized_lemmatized/generated_data_{lang}_10k_mt0xxl_with_complex_and_simple_debiasing.json"
    lexicon_path = "../../../data/lexicon/biasLexiconSynonyms.json"

    dataset = load_json(dataset_path)
    bias_lexicon = load_json(lexicon_path)["bias_lexicon"]

    # Compute bias term counts and word counts
    bias_term_counts, word_counts = count_bias_terms_by_identity_application(dataset, bias_lexicon)

    # Compute TF-IDF
    total_documents = len(bias_term_counts)  # Each (identity, application) pair is a document
    tf_scores = compute_tf(bias_term_counts, word_counts)
    idf_scores = compute_idf(bias_term_counts, total_documents)
    tf_idf_scores = compute_tf_idf(tf_scores, idf_scores)

    # Save TF-IDF results
    save_path = f"../../../data/lexicon_analysis/tfidf/tfidf_values/biasTerms/tfidf_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json"
    save_json(tf_idf_scores, save_path)

    print(f"TF-IDF scores saved successfully for {lang}.")