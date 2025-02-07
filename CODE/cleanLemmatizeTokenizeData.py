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
    return text.lower()  # Convert to lowercase

# Function to tokenize & lemmatize text (consistent with lexicon curation)
def tokenize_lemmatize_text(text):
    """Tokenizes and lemmatizes text using spaCy."""
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha]  # Keep only words

# Function to check if an entry is low quality
def is_low_quality_entry(entry):
    """Check if the generated response is low-quality and should be filtered out."""
    
    generated_output = entry.get("translated_generated_output", "").strip().lower()
    prompt = entry.get("translated_prompt", "").strip().lower()

    # Extract words from the prompt and output (remove punctuation)
    prompt_words = set(re.findall(r'\b\w+\b', prompt))  
    output_words = set(re.findall(r'\b\w+\b', generated_output))  

    # Check if ALL output words are in the prompt (i.e., no new words in response)
    return output_words.issubset(prompt_words)

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