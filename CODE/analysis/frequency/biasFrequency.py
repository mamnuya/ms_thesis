'''
Count bias term frequency for each identity group, then normalize by total word count in each identity group
'''

import json
from collections import defaultdict


def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def count_bias_terms_by_identity(dataset, bias_lexicon):
    """
    Count occurrences of bias terms in processed text outputs, grouped by identity category.
    """
    bias_term_counts = defaultdict(lambda: {"original": defaultdict(int), "complex": defaultdict(int), "simple": defaultdict(int)})
    word_counts = defaultdict(lambda: {"original": 0, "complex": 0, "simple": 0})

    for entry in dataset:
        identity = entry["identity"].lower()  # Identity field from dataset
        original_words = entry["processed_translated_generated_output"]
        complex_words = entry["complex_processed_translated_debiased_output"]
        simple_words = entry["simple_processed_translated_debiased_output"]

        # Identify bias terms based on the lexicon
        for category, bias_terms in bias_lexicon.items():
            bias_terms_set = set(bias_terms)  # Convert to set for fast lookup

            # Count bias terms in original, complex, and simple words
            for word in original_words:
                if word in bias_terms_set:
                    bias_term_counts[identity]["original"][word] += 1
            for word in complex_words:
                if word in bias_terms_set:
                    bias_term_counts[identity]["complex"][word] += 1
            for word in simple_words:
                if word in bias_terms_set:
                    bias_term_counts[identity]["simple"][word] += 1

            # Update total word counts for each identity
            word_counts[identity]["original"] += len(original_words)
            word_counts[identity]["complex"] += len(complex_words)
            word_counts[identity]["simple"] += len(simple_words)

    return bias_term_counts, word_counts

def normalize_bias_frequencies(bias_term_counts, word_counts):
    """Normalize bias term frequencies by total word count per identity group."""
    return {
        identity: {
            "original": {
                term: count / word_counts[identity]["original"] if word_counts[identity]["original"] > 0 else 0
                for term, count in bias_term_counts[identity]["original"].items()
            },
            "complex": {
                term: count / word_counts[identity]["complex"] if word_counts[identity]["complex"] > 0 else 0
                for term, count in bias_term_counts[identity]["complex"].items()
            },
            "simple": {
                term: count / word_counts[identity]["simple"] if word_counts[identity]["simple"] > 0 else 0
                for term, count in bias_term_counts[identity]["simple"].items()
            }
        }
        for identity in bias_term_counts
    }

def save_json(data, filepath):
    """Save JSON data to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# Load dataset JSON
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

for lang in languages: 
    dataset_path = f"../../../data/complex_and_simple_debiaspromptsQs/cleaned_tokenized_lemmatized/generated_data_{lang}_10k_mt0xxl_with_complex_and_simple_debiasing.json"
    lexicon_path = "../../../data/lexicon/biasLexiconSynonyms.json"

    dataset = load_json(dataset_path)
    bias_lexicon = load_json(lexicon_path)["bias_lexicon"]

    # Process bias term counting and normalization by identity
    bias_term_counts, word_counts = count_bias_terms_by_identity(dataset, bias_lexicon)
    normalized_bias_freq = normalize_bias_frequencies(bias_term_counts, word_counts)

    # Save results
    save_path = f"../../../data/lexicon_analysis/frequency/normalized_frequency/frequency_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json"
    save_json(normalized_bias_freq, save_path)

    print(f"Normalized bias term frequencies saved successfully for {lang}.")