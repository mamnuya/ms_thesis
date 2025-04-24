# Data Folder Overview

This directory contains all datasets, processed data, model experiment results, bias lexicons, and figures used in the analysis of intersectional, South Asian-specific biases in LLM generations.

## 📁 Folder Structure

### `complex_and_simple_debiaspromptsQs/`
- **`raw/`**:  
  Contains raw generated data using:
  - Original prompts
  - Complex self-debiasing prompts
  - Simple self-debiasing prompts

- **`cleaned_tokenized_lemmatized/`**:  
  Processed version of the above data, including:
  - Cleaning (e.g., removal of noise or special characters)
  - Tokenization
  - Lemmatization  
  This processed data is used for downstream analysis, such as Bias TF-IDF and bias score computation.

### `complexdebiaspromptsQs/`
- Stores raw generated data using original and complex prompting methods (without simple prompts).
- Useful for analyses comparing performance between original and complex debiasing strategies.

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

### `lexicon/`
- `biasLexicon.json`:  
  Core bias lexicon derived from literature review and manually added synonyms related to South Asian gender, religion, marital status, and family size stereotypes.
  
- `biasLexiconSynonyms.json`:  
  Expanded lexicon using automated synonym generation techniques on the initial manual + literature terms.

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

