'''
Cleans data
clean → tokenize words → lemmatize these words

Saves the lemmatized tokens.

Clean the data → Remove special characters, extra spaces, punctuation, and normalize text (e.g., lowercase).
Tokenize words → Split text into individual words using spaCy.
Lemmatize these words → Convert each word to its base form using spaCy lemmatization.
Save the lemmatized tokens → Store only the lemmatized version in dataset.
'''

import json
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Download necessary NLTK data if not already available
nltk.download('punkt')

# Load spaCy language model 
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

def remove_duplicate_sentences(text):
    """Removes duplicate sentences within a paragraph."""
    sentences = text.split('. ')
    unique_sentences = list(dict.fromkeys(sentences))  # Keeps order while removing duplicates
    return '. '.join(unique_sentences)

def remove_repeated_phrases(text):
    """Removes repetitive phrases in a text."""
    text = re.sub(r'\b(\w+(?:\s+\w+){0,4})\b(?=.*\b\1\b)', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()  # Normalize spaces


# Function to clean text
def clean_text(text):
    """Removes extra spaces, special characters, and normalizes text."""
    if not text or not isinstance(text, str):
        return text  # Return as is if text is None or not a string
    
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.replace('-', ' ')  # Replace dashes with spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.replace('"', '')  # Remove quotation marks
    text = remove_duplicate_sentences(text)  # Remove repeated sentences
    text = remove_repeated_phrases(text)  # Remove repeated phrases
    return text.lower()  # Convert to lowercase

# Function to tokenize & lemmatize text (consistent with lexicon curation)
def tokenize_lemmatize_text(text):
    """Tokenizes and lemmatizes text using spaCy."""
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha]  # Keep only words

# Function to check if an entry is low quality
def is_low_quality_entry(entry):
    """Check if the generated response or debiased responses are low-quality and should be filtered out."""
    
    def words_from_text(text):
        """Extracts words from a given text (removes punctuation and normalizes)."""
        return set(re.findall(r'\b\w+\b', text.lower().strip())) if text else set()

    # Extract relevant fields
    generated_output = entry.get("translated_generated_output", "")
    prompt = entry.get("translated_prompt", "")

    complex_output = entry.get("complex_translated_debiased_output", "")
    complex_prompt = entry.get("complex_debias_prompt", "")

    simple_output = entry.get("simple_translated_debiased_output", "")
    simple_prompt = entry.get("simple_debias_prompt", "")

    # Get sets of words from each field
    prompt_words = words_from_text(prompt)
    output_words = words_from_text(generated_output)

    complex_prompt_words = words_from_text(complex_prompt)
    complex_output_words = words_from_text(complex_output)

    simple_prompt_words = words_from_text(simple_prompt)
    simple_output_words = words_from_text(simple_output)

    # Filtering conditions:
    # 1. Remove if the generated output has no new words beyond the original prompt
    generated_low_quality = output_words.issubset(prompt_words)

    # 2. Remove if the complex debiased output has no new words beyond its debiasing prompt
    complex_low_quality = complex_output_words.issubset(complex_prompt_words)

    # 3. Remove if the simple debiased output has no new words beyond its debiasing prompt
    simple_low_quality = simple_output_words.issubset(simple_prompt_words)

    # If any of these conditions are met, the entry is low quality and should be removed
    return generated_low_quality or complex_low_quality or simple_low_quality

# Process JSON file
def process_json(input_file, output_file):
    """Cleans, lemmatizes, and filters the fields in a JSON file."""
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)  # Load JSON data
    
    processed_data = []
    
    for entry in tqdm(data, desc="Processing JSON Entries"):
        # Clean fields
        entry["cleaned_translated_generated_output"] = clean_text(entry.get("translated_generated_output", ""))
        entry["complex_cleaned_translated_debiased_output"] = clean_text(entry.get("complex_translated_debiased_output", ""))
        entry["simple_cleaned_translated_debiased_output"] = clean_text(entry.get("simple_translated_debiased_output", ""))

        # Lemmatize fields (same as lexicon curation)
        entry["processed_translated_generated_output"] = tokenize_lemmatize_text(entry["cleaned_translated_generated_output"])
        entry["complex_processed_translated_debiased_output"] = tokenize_lemmatize_text(entry["complex_cleaned_translated_debiased_output"])
        entry["simple_processed_translated_debiased_output"] = tokenize_lemmatize_text(entry["simple_cleaned_translated_debiased_output"])

        # Apply filtering rule
        if not is_low_quality_entry(entry):  # Keep only high-quality responses
            processed_data.append(entry)

    # Save processed data
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(processed_data, file, indent=4, ensure_ascii=False)

    print(f"Processed data saved to {output_file}")

# Run the script
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]
for lang in languages:
    input_json = f"../data/complex_and_simple_debiaspromptsQs/raw/generated_data_{lang}_10k_mt0xxl_with_complex_and_simple_debiasing.json"
    output_json = f"../data/complex_and_simple_debiaspromptsQs/cleaned_tokenized_lemmatized/generated_data_{lang}_10k_mt0xxl_with_complex_and_simple_debiasing.json"
    process_json(input_json, output_json)