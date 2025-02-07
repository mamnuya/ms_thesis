'''
Processess lexicon and adds lexicon synonyms where input file is in format with flat list.
"bias_lexicon": {
        "Identity": {
            "word1", "word2
        }

        clean → tokenize → lemmatize
'''

import nltk
import spacy
from nltk.corpus import wordnet as wn
import json

# Download necessary resources
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load spaCy model with word vectors
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

# Paths to bias lexicon files
bias_lexicon_path = "../../data/lexicon/biasLexicon.json"
synonyms_bias_lexicon_path = "../../data/lexicon/biasLexiconSynonyms.json"

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

# Function to tokenize & lemmatize using spaCy
def tokenize_and_lemmatize(text):
    """Tokenizes and lemmatizes text using spaCy."""
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha]  # Keep only words

# Function to get filtered synonyms from WordNet with semantic filtering
def get_filtered_synonyms(word, similarity_threshold=0.5):
    """Finds synonyms that match semantic meaning using WordNet & spaCy similarity."""
    synonyms = set()
    original_word_vector = nlp(word)  # Get word vector for the original word
    
    # Tokenize and lemmatize the original word
    lemmatized_word = " ".join(tokenize_and_lemmatize(word))

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()

            # Ensure valid synonyms (single-word, different from original, not stopwords)
            if synonym == word or len(synonym.split()) > 1 or len(synonym) <= 2:
                continue

            # Check semantic similarity using spaCy word vectors
            synonym_vector = nlp(synonym)
            similarity_score = original_word_vector.similarity(synonym_vector)

            # Add only if similarity score is above threshold
            if similarity_score >= similarity_threshold:
                synonyms.add(synonym)

    # Tokenize and lemmatize synonyms
    synonym_tokens = {syn: tokenize_and_lemmatize(syn) for syn in synonyms}

    return lemmatized_word, list(synonym_tokens.keys()), synonym_tokens

# Function to expand lexicon with better-filtered synonyms
def expand_lexicon(bias_lexicon):
    """Expands bias lexicon with synonyms that pass semantic filtering."""
    expanded_lexicon = {"bias_lexicon": {}}
    all_synonym_tokens = {}

    for category, words in bias_lexicon["bias_lexicon"].items():
        expanded_lexicon["bias_lexicon"][category] = set(words)  # Store as a set
        
        for word in words:
            lemmatized_word, synonyms, synonym_tokens = get_filtered_synonyms(word)
            expanded_lexicon["bias_lexicon"][category].add(lemmatized_word)  # Add lemmatized word
            expanded_lexicon["bias_lexicon"][category].update(synonyms)  # Add synonyms
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