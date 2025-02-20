'''
Prints the top 3 terms with highest tfidf complex score, 
top 3 terms with highest tfidf simple score, 
and top 3 terms with highest tfidf original score for each identity in each language
'''
import json

# List of languages to process
languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]

# Output file
output_file = "../../../data/lexicon_analysis/tfidf/top_tfidf_bias_terms.json"

# Dictionary to store results
top_tfidf_terms = {}

# Process each language
for lang in languages:
    input_path = f"../../../data/lexicon_analysis/tfidf/tfidf_scores/tfidf_analysis_{lang}_mt0xxl_with_complex_and_simple_debiasing.json"

    # Load the JSON file
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n--- Processing Language: {lang} ---\n")

    top_tfidf_terms[lang] = {}

    # Process each identity in the data
    for identity, scores in data.items():
        print(f"Identity: {identity}")

        # Extract top 3 terms by TF-IDF scores
        original_top3 = sorted(scores["original"].items(), key=lambda x: x[1], reverse=True)[:3]
        complex_top3 = sorted(scores["complex"].items(), key=lambda x: x[1], reverse=True)[:3]
        simple_top3 = sorted(scores["simple"].items(), key=lambda x: x[1], reverse=True)[:3]

        # Store results in dictionary
        top_tfidf_terms[lang][identity] = {
            "original": [(term, round(score, 6)) for term, score in original_top3],
            "complex": [(term, round(score, 6)) for term, score in complex_top3],
            "simple": [(term, round(score, 6)) for term, score in simple_top3],
        }

        # Print results with scores
        print("  Top 3 Original TF-IDF terms:", top_tfidf_terms[lang][identity]["original"])
        print("  Top 3 Complex TF-IDF terms:", top_tfidf_terms[lang][identity]["complex"])
        print("  Top 3 Simple TF-IDF terms:", top_tfidf_terms[lang][identity]["simple"])
        print()

# Save results to JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(top_tfidf_terms, f, indent=4, ensure_ascii=False)

print(f"\nTop TF-IDF terms saved to {output_file}")