'''
Program to match the lexicon and collect the bias terms from outputs. 
'''

import json
from collections import defaultdict

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_biased_terms(entry, bias_lexicon):
    """Extract biased terms found in the output for a given entry based on relevant identity categories."""
    identity = entry["identity"]  # Identity of the entry
    original_words = entry["processed_translated_generated_output"]
    complex_words = entry["complex_processed_translated_debiased_output"]
    simple_words = entry["simple_processed_translated_debiased_output"]

    biased_terms_original = set()
    biased_terms_complex = set()
    biased_terms_simple = set()

    # Iterate over bias lexicon categories and apply only if category is in identity
    for category, bias_terms in bias_lexicon.items():
        if category in identity:  # Ensure only relevant bias terms are applied
            bias_terms_set = set(bias_terms)

            biased_terms_original.update(word for word in original_words if word in bias_terms_set)
            biased_terms_complex.update(word for word in complex_words if word in bias_terms_set)
            biased_terms_simple.update(word for word in simple_words if word in bias_terms_set)

    # Convert sets to lists for JSON storage
    return list(biased_terms_original), list(biased_terms_complex), list(biased_terms_simple)


def update_dataset_with_biased_terms(dataset, bias_lexicon):
    """Update dataset by adding biased terms found in each entry's outputs."""
    for entry in dataset:
        biased_terms_original, biased_terms_complex, biased_terms_simple = extract_biased_terms(entry, bias_lexicon)

        entry["biased_terms_original"] = biased_terms_original
        entry["biased_terms_complex"] = biased_terms_complex
        entry["biased_terms_simple"] = biased_terms_simple

    return dataset

def save_json(data, filepath):
    """Save JSON data to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Process and update dataset for each language
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

for lang in languages: 
    dataset_path = f"../../../data/complex_and_simple_debiaspromptsQs/cleaned_tokenized_lemmatized/generated_data_{lang}_10k_mt0xxl_with_complex_and_simple_debiasing.json"
    lexicon_path = "../../../data/lexicon/biasLexiconSynonyms.json"
    save_path = f"../../../data/wordsetdifference_analysis/word_level/biased_terms/bias_term_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json"

    dataset = load_json(dataset_path)
    bias_lexicon = load_json(lexicon_path)["bias_lexicon"]

    # Update dataset with biased terms
    updated_dataset = update_dataset_with_biased_terms(dataset, bias_lexicon)

    # Save updated dataset
    save_json(updated_dataset, save_path)

    print(f"Updated dataset saved successfully for {lang}.")
