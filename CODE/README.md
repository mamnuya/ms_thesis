# Code Folder Overview

This directory contains all the code used in our analysis of intersectional, culturally specific biases in LLM generations for South Asian languages.

---
## 🔗 Submodules

- **[indictrans2](https://github.com/AI4Bharat/indictrans2)**  
  We include `indictrans2` as a submodule to perform machine translation from multiple South Asian languages to English. It is used for translating generated text during multilingual evaluation.

---
## 📁 Folder Structure

---
### `requirements.txt`

Lists all Python package dependencies for the project.

Includes NLP tools like **spaCy** and JSON handling utilities.

#### 🔧 Environment Setup

```bash
# Create and activate virtual environment (if not done)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
---
### `datasetGeneration/`
- Scripts for generating the dataset of identity-conditioned prompts.
- Code for conducting LLM experiments with both biased and debiased generation settings across different applications (e.g., storytelling, to-do lists, hobby descriptions).
- `generateDataset_mt0xxl_complex.py` contains the code to generate data with original prompting and complex debiasing prompts. 
- `generateDataset_mt0xxl_complex_and_simple.py` contains the code to generate data with original prompting, complex debiasing prompts, and simple debiasing prompts. This version was implemented to generate the dataset for our study.
- `/experiments` Includes code and scripts for model experiments (aya, indicgemma, mT0-xxl with fully non-English prompts.)

#### Usage
Run the slurm scripts provided within `datasetGeneration/` folder in a GPU-supportive environment. 

📌 *Remember to replace with an actual email for notifications*:  
```bash
#SBATCH --mail-user=EXAMPLE_EMAIL@gmu.edu
``` 
---
### `cleanLemmatizeTokenizeData.py`

Preprocesses generated outputs before TF-IDF computation:

- **Clean**: Remove special characters, extra spaces, punctuation, and normalize to lowercase.
- **Tokenize**: Split sentences into individual words using **spaCy**.
- **Lemmatize**: Convert each word to its base (dictionary) form.
- **Save**: Store lemmatized tokens to ensure uniform TF-IDF computation.

#### Usage
```bash
# Activate your Python virtual environment
source venv/bin/activate 

# Run Code
python cleanLemmatizeTokenizeData.py 
```
---
### `lexiconCuration/`
- `biasLexiconFindSynonymsFlatList.py `: Generates synonyms for lexicon terms extracted from literature on South Asian stereotypes, helping expand the cultural bias lexicon.
- `lexicon_synonym_words_counts.py`: Counts and prints:
  - Number of terms from literature review
  - Number of terms added manually
  - Total number of terms after synonym generation

This ensures transparency in the lexicon development process.

#### Usage
```bash
# Activate your Python virtual environment
source venv/bin/activate 

# Run Code
python biasLexiconFindSynonymsFlatList.py 

python lexicon_synonym_words_counts.py
```

---
### `analysis/tfidf/`
- `biastermsTFIDF.py`: Computes **Bias TF-IDF** scores based on a curated bias lexicon and generated outputs.
- `overalltermsTFIDF.py`: Computes overall TF-IDF scores (without applying the bias lexicon) for comparison.
- `identity_application_biastermScore.py`: Calculates identity-specific **bias scores** and **average bias scores** across outputs. Stores relevant results, and prints/shows/saves tables/diagrams.
- `identity_application_overallTermScore.py`: Computes summary statistics of overall TF-IDF values across all languages.

#### Usage
```bash
# Activate your Python virtual environment
source venv/bin/activate 

# Run Code
python biastermsTFIDF.py

python overalltermsTFIDF.py

python identity_application_biastermScore.py

python identity_application_overallTermScore.py
```
---