'''
Program to prints the manually added synonyms that were added to the bias terms from lexicon review.

Counts and prints how many words in lexicon from literature review, how many words added manually, and how many words after programmed synonym generation.
'''

import json

# Load the JSON file
with open("../../data/lexicon/biasLexicon.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Dictionary of identity categories and terms to remove, these are words directly from literature review
terms_to_remove = {
    "Divorced": ["disaffection", "conflict", "stress", "violence", "depression", "helplessness", "aggression", "altered", "grief", "embarrassment", "hopelessness", "restlessness", "gloom", "unhealthy", "shock"],
    "Female who is Divorced": ["damage", "insomnia", "panic", "weak", "abuse", "lost", "insecure", "weak", "troublemaker", "burden", "hatred", "distrust", "aggression", "vengeance", "tolerated", "inferior", "reject"],
    "Male who is Divorced": ["suppress", "strain", "depression", "failure", "struggle", "distant"],
    "Female who is Divorced with One child": ["vulnerable", "decline", "anxiety", "stress"],
    "Female who is Divorced with Many children": ["vulnerable", "decline", "anxiety", "stress"],
    "Male who is Divorced with One child": ["mourn", "loss", "sad", "powerless", "useless", "worry", "guilt", "anger"],
    "Male who is Divorced with Many children": ["mourn", "loss", "sad","powerless", "useless", "worry", "guilt", "anger"],
    "Single": ["selfish", "ugly", "sad", "lonely", "immature", "insecure", "angry","miserable","inferior", "deficient", "pressure", "excluded", "isolated", "independent", "freedom", "creative", "flexible"],
    "Female who is Single": ["victim", "stress", "suffer"],
    "Male who is Single": ["bachelor", "womanizer", "flexible", "independent"],
    "Single with One child": ["impure", "discourage", "embarrass", "harass", "violent", "shame", "unchaste", "tempt", "misbehave", "stray", "illegitimate"],
    "Single with Many children": ["impure", "discourage", "embarrass", "harass", "violent", "shame", "unchaste", "tempt", "misbehave", "stray", "illegitimate"],
    "Female who is Single with One child": ["shame", "abuse", "violent"],
    "Female who is Single with Many children": ["shame", "abuse", "violent"],
    "Married": ["happy", "functional", "dignified", "valued", "mature", "stable", "honest", "kind", "loving", "extroverted", "agreeable", "conscientious", "attractive", "satisfied", "warm"],
    "Female who is Married": ["dignity", "respect", "secure"],
    "Hindu Female who is Married": ["dowry", "demand"],
    "Male who is Married with No children": ["abuse", "emasculate", "demotivated", "defective", "disgrace"],
    "Female who is Married with No children": ["disaster", "consequence", "abuse", "taunt", "harassment", "excluded", "unacceptable", "failure", "weak", "vulnerable", "ostracized", "cheater", "abandon"],
    "Female who is Married with One child": ["power", "status", "secure", "natural", "homemaker", "divine"],
    "Female who is Married with Many children": ["power", "status", "secure", "natural", "homemaker", "divine"],
    "Married with One child": ["status", "prestige", "security", "acceptance", "success", "happiness", "fertile", "social", "pride", "recognition"],
    "Married with Many children": ["status", "prestige", "security", "acceptance", "success", "happiness", "fertile", "social", "pride", "recognition"],
    "Married with No children": ["hollow", "fruitless", "dried", "barren"],
    "Muslim": ["violent", "militant", "misogynistic", "terrorist", "untrustworthy", "bad", "strict", "rude", "evil", "rapist", "villain", "jihad"],
    "Hindu": ["violent", "perpetrator", "conniving", "racist", "supremacist", "discriminatory", "evil", "intolerant", "anger", "casteist", "innocent", "backward", "victim", "uneducated", "weird"],
    "Female": ["affectionate", "helpful", "friendly", "kind", "sympathetic", "sensitive", "gentle", "soft", "submissive", "deference", "isolated", "depression", "forced", "excluded", "liability", "burden", "neglect", "murder", "inferior", "emotional", "unstable",
               "household", "chores", "domestic", "childcare", "shopping", "cook", "clean", "laundry", "dishes", "clothes", "iron", "care"],
    "Male": ["risk", "violent", "masculine", "threatening", "dominance", "control", "strong", "hero", "stoic", "provide", "protect", "assertive", "leader",
             "restrain", "charisma", "robust", "resilient", "confident", "breadwinner", "power", "authority", "competitive", "objective", "ambitious",
             "responsible", "aggressive", "forceful", "grocery", "market", "management", "repair", "maintenance", "bill", "car"],
    "Muslim Female": ["oppressed", "conservative"],
    "Female with No children": ["shame", "anger", "resentment", "blame", "tension", "hostility", "abuse", "anxiety", "depression", "suicide", "stress", "ostracism", "taunt", "distress", "suffer", "torture", "dependence"]
}

# Function to remove specified terms
def remove_terms(data, terms_to_remove):
    for identity, words in terms_to_remove.items():
        if identity in data["bias_lexicon"]:
            data["bias_lexicon"][identity] = [word for word in data["bias_lexicon"][identity] if word not in words]
    return data

total_terms = sum(len(terms) for terms in terms_to_remove.values())
print(f"Total number of terms to remove (ie. in sourced lexicon): {total_terms} \n")

# Function to count and print words along with their counts
def count_words(cleaned_data):
    word_counts = {}
    for identity, words in cleaned_data["bias_lexicon"].items():  # Ensure we're accessing the correct key
        count = len(words)  # Count number of words for this identity
        if count > 0:  # Only print if there are manually added words
            print(f"{identity} {words}: {count} manually added")
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1  # Increment count for each word

    return word_counts

# Clean the lexicon to remove words directly from literature review
cleaned_data = remove_terms(data, terms_to_remove)
# Count words and print them
word_counts = count_words(cleaned_data)

# Verify the total count matches total_words_cleaned
total_words_cleaned = sum(word_counts.values())
print(f"\nTOTAL number of words manually added: {total_words_cleaned}")

with open("../../data/lexicon/biasLexiconSynonyms.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Count the words in the "bias_lexicon" dictionary
total_words = sum(len(words) for words in data["bias_lexicon"].values())

print(f"\nTOTAL number of words in bias_lexicon after synonym generation: {total_words}")

# Save the cleaned lexicon back to a new file
#with open("cleaned_bias_lexicon.json", "w", encoding="utf-8") as outfile:
#    json.dump(cleaned_data, outfile, indent=4, ensure_ascii=False)

print("Manually Added Synonyms: Printed.")