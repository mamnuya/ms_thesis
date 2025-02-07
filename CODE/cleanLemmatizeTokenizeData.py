import json
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Download necessary NLTK data if not already available
nltk.download('punkt')

# Load spaCy language model (you can replace with another model for non-English)
nlp = spacy.load("en_core_web_sm")

# Function to clean text
def clean_text(text):
    """Removes extra spaces, special characters, and normalizes text."""
    if not text or not isinstance(text, str):
        return text  # Return as is if text is None or not a string
    
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.replace('"', '')  # Remove quotation marks
    return text.lower()  # Convert to lowercase

# Function to tokenize text
def tokenize_text(text):
    """Splits text into words (tokens)."""
    return word_tokenize(text)

# Function to lemmatize text
def lemmatize_text(text):
    """Reduces words to their base forms using spaCy."""
    doc = nlp(text)
    return [token.lemma_ for token in doc]

# Function to check if an entry is low quality
def is_low_quality_entry(entry):
    """Check if the generated response is low-quality and should be filtered out."""
    
    generated_output = entry.get("translated_generated_output", "").strip().lower()
    prompt = entry.get("translated_prompt", "").strip().lower()

    # Extract words from the prompt and output (remove punctuation)
    prompt_words = set(re.findall(r'\b\w+\b', prompt))  # Tokenize prompt
    output_words = set(re.findall(r'\b\w+\b', generated_output))  # Tokenize output

    # Check if ALL output words are in the prompt (i.e., no new words in response)
    return output_words.issubset(prompt_words)

# Process JSON file
def process_json(input_file, output_file):
    """Cleans, tokenizes, lemmatizes, and filters the fields in a JSON file."""
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)  # Load JSON data
    
    processed_data = []
    
    for entry in tqdm(data, desc="Processing JSON Entries"):
        # Clean fields
        entry["cleaned_translated_generated_output"] = clean_text(entry.get("translated_generated_output", ""))
        entry["complex_cleaned_translated_debiased_output"] = clean_text(entry.get("complex_translated_debiased_output", ""))
        entry["simple_cleaned_translated_debiased_output"] = clean_text(entry.get("simple_translated_debiased_output", ""))

        # Tokenize fields
        entry["tokenized_translated_generated_output"] = tokenize_text(entry["cleaned_translated_generated_output"])
        entry["complex_tokenized_translated_debiased_output"] = tokenize_text(entry["complex_cleaned_translated_debiased_output"])
        entry["simple_tokenized_translated_debiased_output"] = tokenize_text(entry["simple_cleaned_translated_debiased_output"])

        # Lemmatize fields
        entry["lemmatized_translated_generated_output"] = lemmatize_text(" ".join(entry["tokenized_translated_generated_output"]))
        entry["complex_lemmatized_translated_debiased_output"] = lemmatize_text(" ".join(entry["complex_tokenized_translated_debiased_output"]))
        entry["simple_lemmatized_translated_debiased_output"] = lemmatize_text(" ".join(entry["simple_tokenized_translated_debiased_output"]))

        # Apply filtering rule
        if not is_low_quality_entry(entry):  # Keep only high-quality responses
            processed_data.append(entry)

    # Save processed data
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(processed_data, file, indent=4, ensure_ascii=False)

    print(f"Processed data saved to {output_file}")

# Run the script
input_json = "your_input.json"  # Replace with your actual input file
output_json = "processed_output.json"  # Replace with your desired output file
process_json(input_json, output_json)

