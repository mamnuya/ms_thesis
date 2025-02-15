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
from langdetect import detect

# Download necessary NLTK data if not already available
nltk.download('punkt')

# Load spaCy language model 
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

def remove_duplicate_sentences(text):
    """Removes duplicate sentences while preserving order."""
    sentences = text.split('. ')
    unique_sentences = list(dict.fromkeys(sentences))  # Keeps the first occurrence, removes duplicates
    return '. '.join(unique_sentences)

def remove_repeated_phrases(text):
    """Removes consecutive repeated phrases in a text."""
    words = text.split()  # Split text into words
    cleaned_words = []
    prev_word = None

    for word in words:
        if word != prev_word:  # Only add if it's not a repeat
            cleaned_words.append(word)
        prev_word = word  # Update previous word

    return ' '.join(cleaned_words)

def remove_repeated_phrases_by_special_chars(text):
    """Removes repeated phrases enclosed by special characters while preserving order."""
    # Define a list of special characters to split the text by
    special_chars = ['-', ',', ';', ':', '!', '?', '.', '(', ')', '[', ']', '{', '}', '|']
    
    # Create a regex pattern that matches any of the special characters
    pattern = r'([{}])'.format(re.escape(''.join(special_chars)))
    
    # Split the text by the special characters and remove any duplicates
    parts = re.split(pattern, text)
    
    seen = set()
    result = []
    
    for part in parts:
        # If the part is not in 'seen' and it's not empty, append it to the result
        if part and part not in seen:
            seen.add(part)
            result.append(part)
    
    return ''.join(result)

def clean_text(text):
    """Removes extra spaces, special characters, duplicate sentences, and normalizes text."""
    if not text or not isinstance(text, str):
        return text  # Return as is if text is None or not a string
    
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = remove_duplicate_sentences(text)  # Remove repeated sentences
    text = remove_repeated_phrases(text)  # Remove consecutive duplicate words
    text = remove_repeated_phrases_by_special_chars(text)  # Remove repeated phrases enclosed by special characters
    text = text.replace('-', ' ')  # Replace dashes with spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.replace('"', '')  # Remove quotation marks
    return text.lower()  # Convert to lowercase

# Function to tokenize & lemmatize text (consistent with lexicon curation)
def tokenize_lemmatize_text(text):
    """Tokenizes and lemmatizes text using spaCy."""
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha]  # Keep only words

def contains_only_indic(text):
    """Returns True if the text contains only foreign languages"""
    
    try:
        # Attempt to detect the language
        lang = detect(text)
        
        # If the detected language is not English, return True
        if lang == 'en':
            return False
        return True
    except Exception: # if lang not recognized, also return true since it's not English
        return True


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

    org_output_foreign = entry.get("generated_output", "")
    complex_output_foreign = entry.get("complex_debiased_output", "")
    simple_output_foreign = entry.get("simple_debiased_output", "")

    # 4. Ensure all outputs contain only Indic characters (no English)
    contains_english = (
        not contains_only_indic(org_output_foreign) or
        not contains_only_indic(complex_output_foreign) or
        not contains_only_indic(simple_output_foreign)
    )

    # If any of these conditions are met, the entry is low quality and should be removed
    return generated_low_quality or complex_low_quality or simple_low_quality or contains_english

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