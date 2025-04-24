# Data Folder Overview

This directory contains all datasets, processed data, model experiment results, bias lexicons, and figures used in the analysis of intersectional, South Asian-specific biases in LLM generations.

## 📁 Folder Structure

### `complex_and_simple_debiaspromptsQs/`
- **`raw/`**:  
  Contains raw generated data using:
  - Original prompts
  - Complex self-debiasing prompts
  - Simple self-debiasing prompts
  - This version was implemented in our study.

- **`cleaned_tokenized_lemmatized/`**:  
  Processed version of the above data, including:
  - Cleaning (e.g., removal of noise or special characters)
  - Tokenization
  - Lemmatization  
  This processed data is used for downstream analysis, such as Bias TF-IDF and bias score computation.

### `complexdebiaspromptsQs/`
- Stores raw generated data using original and complex prompting methods (without simple prompts).
- Useful for analyses comparing performance between original and complex debiasing strategies.
- This version was not used in our study.

  ### `lexicon/`

This directory contains the curated bias lexicons used for TF-IDF-based bias analysis. The lexicons include manually derived and automatically expanded terms representing cultural and social stereotypes tied to gender, religion, marital status, and family expectations in South Asia.

---

#### 📄 `biasLexicon.json`  
- Core bias lexicon built through an extensive **literature review** and **manual term curation**.  
- Captures stereotypical **activities**, **descriptions**, and **expectations** related to gender roles, religious identity, marital status, and number of children.  
- Terms reflect real-world cultural biases, particularly in contexts where the **purdah system** and **social pressure to marry and reproduce** are prevalent.
- This version was implemented in our study.

---

#### 📄 `biasLexiconSynonyms.json`  
- Expanded version of `biasLexicon.json` that includes **automatically generated synonyms**.  
- Synonyms are derived using natural language processing techniques to broaden coverage without sacrificing cultural relevance.  
- Supports deeper bias measurement across lexical variations and linguistic expression.
- This version was implemented in our study.

---

#### 📄 `biasLexiconSubLists.json`  
- Subdivides the bias lexicon (`biasLexicon.json`) into two categories:  
  - `"activities"` – e.g., *cooking, raising children, going to temple*  
  - `"descriptions"` – e.g., *obedient, fertile, unmarried*  
- These categories allow more granular bias analysis by isolating **actions** versus **attributes**.
- This version was not implemented in our study.

---

#### 📄 `biasLexiconSynonymsSubLists.json`  
- Subdivides the **expanded synonym lexicon** (`biasLexiconSynonyms.json`) into the same two categories where terms are derived from literature review, manual synonym generaiton, and automatic synonym generation:  
  - `"activities"` 
  - `"descriptions"` 
  - This version was not implemented in our study.

### `lexicon_analysis/tfidf/tfidf_values/`

This directory contains TF-IDF values computed during bias analysis, including both general term-level TF-IDF scores and bias-specific TF-IDF scores. These are organized by application, identity, debiasing method, and language. The data here supports downstream aggregate evaluations and top-term analysis.

---

#### 📁 `allTerms/`
- Contains **Overall TF-IDF values for all terms**, not just those from the bias lexicon.
- Files are grouped by language and application.
- Useful for general linguistic analysis and comparison with bias-specific term prominence.

---

#### 📁 `biasTerms/`
- Stores **TF-IDF values for terms in the curated bias lexicon**.
- Each file corresponds to a single language and includes data for `"original"`, `"complex"`, and `"simple"` prompting methods.
- Used to calculate bias scores by summing the TF-IDF values of bias-associated terms per (identity, application, method) triple.

---

#### 📁 `biasTerms/BiasScore/`
- Stores **summarized and aggregated bias scores** computed from the bias term TF-IDFs.

##### 📄 `bias_scores_<language>.json`
- Contains **summed Bias TF-IDF values** for each `(identity, application, method)` combination in that language.
- Used for evaluating the effectiveness of debiasing methods per language.

##### 📄 `avg_bias_scores_by_language_family.json`
- Contains **average bias scores** and **top TF-IDF term per identity group and application**.
- Aggregated by **language family** (Indo-Aryan or Dravidian).
- Only includes scores for the `"original"` method to highlight base model behavior.

##### 📄 `aggregated_bias_scores_by_language.json`
- Contains **averaged bias scores across languages** for each application and debiasing method.
- Includes results for **Indo-Aryan**, **Dravidian**, and **All_Languages** groups.
- Used in high-level evaluations of debiasing performance.

### `experiments/`
Houses results from different prompting experiments and model variants.

- **`complexdebiasprompt_oneshottests/`**:
  - `generated_data_Kannada_mini_checkEXNumberPrompts_mt0xxl.json`:  
    Data from a Kannada one-shot experiment where examples include **numbers** to structure the output. Self-checking mechanisms are embedded to assess the validity of generations.
  
  - `generated_data_Kannada_mini_exAndcheckAll_mt0xxl2.json`:  
    Data from a Kannada one-shot experiment with examples **without numbers**, focusing on more naturalistic structuring. Also includes self-checking.

- **`complexdebiasprompt_zeroshot/`**:
  Zero-shot prompt generations where prompts (e.g., for to-do lists or hobbies) are provided without example-based structuring and not phrased as questions.

- **`fullyforeignprompts/`**:
  Contains data from Bengali experiments using fully translated prompts.
  - Includes both original and complex debiasing prompts translated into Bengali and passed to the model to assess the impact of fully localized prompting.

- **`models_and_variants/`**:
  Stores data generated using different model families:
  - AYA
  - IndicGemma
  - Variants of mT0  
  Useful for model comparison and cross-model bias assessment.

### `figures/`
- Relevant figures (e.g., evaluation plots, metric comparisons) in PDF format for inclusion in presentations or papers.