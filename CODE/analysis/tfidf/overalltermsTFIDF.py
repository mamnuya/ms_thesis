'''
Compute TF-IDF for each term within the given dataset for identity and application groups.

    t: any term
    d: an identity group's data for a given application (e.g., a specific application of data like "Story", "Hobbies", etc.)
    
    Compute TF(t,d) = (Occurrences of t in d) / (Total number of terms in d)

    N = total number of identity-application pairs (documents)
    df(t) = document frequency of term t (the number of identity-application pairs containing term t)
    Compute IDF(t) = log((N + 1) / (df(t) + 1)) + 1

    TFIDF(t,d) = TF(t,d) * IDF(t)

    term_counts: A dictionary containing term occurrences for each (identity, application) pair, broken down by methods ("original", "complex", "simple").
    word_counts: A dictionary containing the total word count for each (identity, application) pair, broken down by methods.
    total_documents: The total number of identity-application pairs in the dataset (used to compute IDF).

    Returns:
        tf_idf_scores: A dictionary containing the TF-IDF scores for each term within each (identity, application) pair.

'''

import json
import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import json
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from langdetect import detect

# Download necessary NLTK data if not already available
nltk.download('punkt')

# Load spaCy language model 
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

# Download stopwords 
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_tf(term_counts, word_counts):
    """Compute Term Frequency (TF) for each term per (identity, application) group."""
    tf_scores = defaultdict(lambda: defaultdict(lambda: {"original": {}, "complex": {}, "simple": {}}))

    for (identity, application), counts in term_counts.items():
        for method in ["original", "complex", "simple"]:
            total_words = word_counts[(identity, application)][method]
            for term, count in counts[method].items():
                tf_scores[identity][application][method][term] = count / total_words if total_words > 0 else 0

    return tf_scores

def compute_idf(term_counts, total_documents):
    """Compute Inverse Document Frequency (IDF) for each term."""
    idf_scores = defaultdict(float)
    document_frequencies = defaultdict(int)

    # Count the number of (identity, application) groups each term appears in
    for identity_application_counts in term_counts.values():
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

# Function to tokenize & lemmatize text (consistent with lexicon curation)
def tokenize_lemmatize_text(text):
    """Tokenizes and lemmatizes text using spaCy."""
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha]  # Keep only words

def count_terms_by_identity_application(dataset):
    """Count occurrences of all terms excluding stopwords, grouped by (identity, application), ensuring unique term appearances per document."""
    term_counts = defaultdict(lambda: {"original": defaultdict(int), "complex": defaultdict(int), "simple": defaultdict(int)})
    word_counts = defaultdict(lambda: {"original": 0, "complex": 0, "simple": 0})

    for entry in dataset:
        identity = entry["identity"]
        application = entry["application"]
        prompt = entry["prompt"]
        identity_application = (identity, application)

        # Tokenize & lemmatize prompt once per entry
        prompt_tokens = set(tokenize_lemmatize_text(prompt.lower()))

        def filter_words(words):
            """Filters words, ensuring uniqueness and excluding prompt words & stopwords."""
            words = set(words)  # Convert to set once
            return {
                word.lower() for word in words
                if word.lower() not in stop_words  # Exclude stopwords
                and word.lower() not in prompt_tokens  # Exclude words from the prompt
                and not (application == "Hobbies and Values" and word.lower() in {"value", "personal", "interest"})  # Application-specific exclusions
            }

        # Apply filtering to different outputs
        original_words = filter_words(entry["processed_translated_generated_output"])
        complex_words = filter_words(entry["complex_processed_translated_debiased_output"])
        simple_words = filter_words(entry["simple_processed_translated_debiased_output"])

        # Count occurrences of filtered terms (ensuring unique term appearance per document)
        for word in original_words:
            term_counts[identity_application]["original"][word] += 1
        for word in complex_words:
            term_counts[identity_application]["complex"][word] += 1
        for word in simple_words:
            term_counts[identity_application]["simple"][word] += 1

        # Total words count for TF calculation (after filtering)
        word_counts[identity_application]["original"] += len(original_words)
        word_counts[identity_application]["complex"] += len(complex_words)
        word_counts[identity_application]["simple"] += len(simple_words)

    return term_counts, word_counts

def save_json(data, filepath):
    """Save JSON data to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Load dataset JSON and compute TF-IDF
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

for lang in languages: 
    dataset_path = f"../../../data/complex_and_simple_debiaspromptsQs/cleaned_tokenized_lemmatized/generated_data_{lang}_10k_mt0xxl_with_complex_and_simple_debiasing.json"

    dataset = load_json(dataset_path)

    # Compute term counts and word counts
    term_counts, word_counts = count_terms_by_identity_application(dataset)

    # Compute TF-IDF
    total_documents = len(term_counts)  # Each (identity, application) pair is a document
    tf_scores = compute_tf(term_counts, word_counts)
    idf_scores = compute_idf(term_counts, total_documents)
    tf_idf_scores = compute_tf_idf(tf_scores, idf_scores)

    # Save TF-IDF results
    save_path = f"../../../data/lexicon_analysis/tfidf/tfidf_values/allTerms/tfidf_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json"
    save_json(tf_idf_scores, save_path)

    print(f"TF-IDF scores saved successfully for {lang}.")
