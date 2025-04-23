# Purdah and Patriarchy: Evaluating and Mitigating South Asian Biases in Open-Ended Multilingual LLM Generations

## Introduction

Large Language Models (LLMs) are critical in AI systems, yet their deployment in culturally diverse contexts, such as South Asia, poses unique challenges. South Asian societies are deeply shaped by gender roles, religious norms, marital expectations, childbearing pressures, and practices like *purdah* and patriarchy.

Biases reflecting these cultural norms are often embedded in the training data of LLMs, posing risks of reinforcing harmful stereotypes and marginalizing vulnerable identities. Existing work on bias in LLMs lacks coverage of South Asian intersectionality across languages, culture, and real-world applications.

This project addresses five research questions with corresponding contributions:

- **RQ1: How do biases manifest in Indo-Aryan and Dravidian languages across different intersectional dimensions (e.g., religion, gender, marital status, family size)?**  
  → We conduct the first multilingual, intersectional study of LLM outputs across 10 South Asian languages, exploring cultural norms embedded in regional language groups.

- **RQ2: What are the specific South Asian biases present in LLMs, especially regarding stigmas related to marriage, reproduction, and practices like purdah?**  
  → We curate a novel lexicon of culturally specific biases, including stereotypes surrounding childlessness, gender roles, and religious norms.

- **RQ3: Can self-debiasing techniques effectively mitigate intersectional and culturally specific biases in LLMs, particularly in South Asian contexts?**  
  → We evaluate simple and complex self-debiasing prompts in open-ended generation tasks rather than constrained formats, providing a realistic benchmark for mitigation strategies.

- **RQ4: What new metrics and methods can be developed to evaluate South Asian-specific bias and the reduction of these biases after self-debiasing?**  
  → We introduce novel metrics such as **Bias TF-IDF** and identity-specific bias scores to evaluate nuanced harms beyond conventional metrics like toxicity or binary gender bias.

- **RQ5: Which generative applications (e.g., storytelling, to-do lists, descriptions of hobbies and values) reveal the highest levels of intersectional bias in South Asian languages?**  
  → We analyze open-ended LLM generations across multiple real-life use cases to assess where and how bias manifests.

## Repository Structure

- 📂 [Code Folder](./CODE): Contains scripts for model experiments, generation, bias scoring (including Bias TF-IDF), prompt formatting, and evaluation metrics.
- 📂 [Data Folder](./data): Includes curated identity templates, translated prompts, generated outputs, the cultural bias lexicon, and calculated metrics.

## Paper
FILL THIS IN 

## Citation

FILL THIS IN

Please cite this work if you find it useful in your own research:

```bibtex
@inproceedings{your_citation_2025,
  title={NONE},
  author={Mamnuya Rinki},
  year={2025},
  booktitle={NONE},
  note={NONE}
}