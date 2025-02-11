'''
Compute tfidf 

t: bias term from lexicon
d: document AKA an identity group's data from the dataset
Compute TF(t,d) = TF(t,d) =  # occurences of t in d/ # terms in d

N = total # identites
df(t) = document frequency of term t AKA number of identities in which term t occurs
Compute IDF(t) = log( (N+1) / (df(t)+1) ) + 1

TFIDF(t,d)=TF(t,d)*IDF(t)

'''

import json
import math
from collections import defaultdict

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_tf(bias_term_counts, word_counts):
    """Compute Term Frequency (TF) for each bias term per identity group."""
    tf_scores = defaultdict(lambda: {"original": {}, "complex": {}, "simple": {}})
    
    for identity, counts in bias_term_counts.items():
        for method in ["original", "complex", "simple"]:
            total_words = word_counts[identity][method]
            for term, count in counts[method].items():
                tf_scores[identity][method][term] = count / total_words if total_words > 0 else 0
    
    return tf_scores

def compute_idf(bias_term_counts, total_identities):
    """Compute Inverse Document Frequency (IDF) for each bias term."""
    idf_scores = defaultdict(float)
    document_frequencies = defaultdict(int)
    
    # Count the number of identity groups each term appears in
    for identity_counts in bias_term_counts.values():
        seen_terms = set()
        for method in ["original", "complex", "simple"]:
            seen_terms.update(identity_counts[method].keys())  
        for term in seen_terms:
            document_frequencies[term] += 1  # Number of identity groups containing the term

    # Compute IDF using log scaling to prevent division by zero
    for term, doc_count in document_frequencies.items():
        idf_scores[term] = math.log((total_identities + 1) / (doc_count + 1)) + 1  # Smoothing factor

    return idf_scores

def compute_tf_idf(tf_scores, idf_scores):
    """Compute TF-IDF by multiplying TF by IDF."""
    tf_idf_scores = defaultdict(lambda: {"original": {}, "complex": {}, "simple": {}})
    
    for identity, methods in tf_scores.items():
        for method in ["original", "complex", "simple"]:
            for term, tf_value in methods[method].items():
                tf_idf_scores[identity][method][term] = tf_value * idf_scores[term]
    
    return tf_idf_scores

def count_bias_terms_by_identity(dataset, bias_lexicon):
    """Count occurrences of bias terms in processed text outputs, grouped by identity."""
    bias_term_counts = defaultdict(lambda: {"original": defaultdict(int), "complex": defaultdict(int), "simple": defaultdict(int)})
    word_counts = defaultdict(lambda: {"original": 0, "complex": 0, "simple": 0})

    for entry in dataset:
        identity = entry["identity"].lower()
        original_words = entry["processed_translated_generated_output"]
        complex_words = entry["complex_processed_translated_debiased_output"]
        simple_words = entry["simple_processed_translated_debiased_output"]

        for category, bias_terms in bias_lexicon.items():
            bias_terms_set = set(bias_terms)

            for word in original_words:
                if word in bias_terms_set:
                    bias_term_counts[identity]["original"][word] += 1
            for word in complex_words:
                if word in bias_terms_set:
                    bias_term_counts[identity]["complex"][word] += 1
            for word in simple_words:
                if word in bias_terms_set:
                    bias_term_counts[identity]["simple"][word] += 1

        word_counts[identity]["original"] += len(original_words)
        word_counts[identity]["complex"] += len(complex_words)
        word_counts[identity]["simple"] += len(simple_words)

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
    bias_term_counts, word_counts = count_bias_terms_by_identity(dataset, bias_lexicon)

    # Compute TF-IDF
    total_identities = len(bias_term_counts)
    tf_scores = compute_tf(bias_term_counts, word_counts)
    idf_scores = compute_idf(bias_term_counts, total_identities)
    tf_idf_scores = compute_tf_idf(tf_scores, idf_scores)

    # Save TF-IDF results
    save_path = f"../../../data/lexicon_analysis/tfidf/tfidf_scores/tfidf_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json"
    save_json(tf_idf_scores, save_path)

    print(f"TF-IDF scores saved successfully for {lang}.")
