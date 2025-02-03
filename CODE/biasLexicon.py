# pip install nltk
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.5.0/en_core_web_md-3.5.0.tar.gz

import nltk
import spacy
from nltk.corpus import wordnet as wn
import json

# Download necessary resources
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load spaCy model with word vectors
nlp = spacy.load("en_core_web_md")

# Paths to bias lexicon files
bias_lexicon_path = "../data/biasLexicon.json"
synonyms_bias_lexicon_path = "../data/biasLexiconSynonyms.json"

# Function to load JSON file
def load_json(filepath):
    """Load JSON file from a given path."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Function to save JSON file
def save_json(data, filepath):
    """Save JSON file to a given path."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Function to get synonyms from WordNet with semantic filtering
def get_filtered_synonyms(word, similarity_threshold=0.6):
    """Finds one synonym that matches POS and semantic meaning using spaCy similarity."""
    synonyms = set()
    original_word_vector = nlp(word)  # Get word vector for the original word
    
    # Tokenize the original word first
    word_tokens = [token.text for token in nlp(word).doc]  # Tokenizing the input word
    tokenized_word = " ".join(word_tokens)  # Join the tokens to get the tokenized word

    # Only find one synonym that matches the threshold
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()

            # Ensure the synonym is not the same as the original word and is a single word
            if synonym == word or len(synonym.split()) > 1 or len(synonym) <= 2:
                continue

            # Check semantic similarity using spaCy word vectors
            synonym_vector = nlp(synonym)
            similarity_score = original_word_vector.similarity(synonym_vector)

            # Add only if similarity score is above threshold
            if similarity_score >= similarity_threshold:
                synonyms.add(synonym)

    # Tokenize synonyms before returning them
    synonym_tokens = {}
    for synonym in synonyms:
        synonym_tokens[synonym] = [token.text for token in nlp(synonym).doc]  # Tokenizing each synonym

    return tokenized_word, list(synonym_tokens.keys()), synonym_tokens  # Return tokenized word and its synonym

# Function to expand lexicon with better-filtered synonyms
def expand_lexicon(bias_lexicon):
    """Expands bias lexicon with synonyms that pass semantic filtering."""
    expanded_lexicon = {"bias_lexicon": {}}
    all_synonym_tokens = {}  # Dictionary to hold tokens of all added synonyms

    for category, words in bias_lexicon["bias_lexicon"].items():
        expanded_lexicon["bias_lexicon"][category] = set(words)  # Store as a set to avoid duplicates
        
        for word in words:
            tokenized_word, synonyms, synonym_tokens = get_filtered_synonyms(word)
            expanded_lexicon["bias_lexicon"][category].add(tokenized_word)  # Add tokenized word to the category
            expanded_lexicon["bias_lexicon"][category].update(synonyms)  # Add synonym to the same category
            all_synonym_tokens[word] = synonym_tokens  # Store synonym tokens for analysis

    # Convert sets to lists for JSON compatibility
    return {"bias_lexicon": {category: list(words) for category, words in expanded_lexicon["bias_lexicon"].items()}}, all_synonym_tokens

# Load bias lexicon from JSON file
bias_lexicon = load_json(bias_lexicon_path)

# Expand the lexicon with synonyms only and capture synonym tokens
expanded_bias_lexicon, all_synonym_tokens = expand_lexicon(bias_lexicon)

# Save the expanded lexicon to a new JSON file
save_json(expanded_bias_lexicon, synonyms_bias_lexicon_path)

# Print success message and synonym tokens
print(f"Expanded bias lexicon saved successfully to {synonyms_bias_lexicon_path}")
print("Synonym tokens:", all_synonym_tokens)
