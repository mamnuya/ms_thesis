import itertools
import json
import torch
import sys
import os

# Add the directory containing IndicTransToolkit to sys.path
sys.path.append('/scratch/mrinki/ms_thesis/IndicTransToolkit')
# Now import IndicProcessor from the local IndicTransToolkit folder
from IndicTransToolkit import IndicProcessor

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#, BitsAndBytesConfig

#import bitsandbytes as bnb
import re
cache_dir = "/scratch/mrinki/huggingface_cache"

# Load Aya for generation and debiasing
aya_model_name = "CohereForAI/aya-101" #"CohereForAI/aya-101" #bigscience/mt0-large


aya_tokenizer = AutoTokenizer.from_pretrained(aya_model_name, cache_dir=cache_dir)

# Configure 8-bit quantization
#quantization_config = BitsAndBytesConfig(
#    load_in_8bit=True,  # Enable 8-bit quantization
#    llm_int8_enable_fp32_cpu_offload=True  # Use FP32 offload if needed
#)

# Load the model with 8-bit quantization
aya_model = AutoModelForSeq2SeqLM.from_pretrained(
    aya_model_name,
    cache_dir=cache_dir,
    device_map="auto",  # Automatically assign layers to available GPUs
    torch_dtype=torch.float16
    #,
    #quantization_config=quantization_config
)


# Load AI4Bharat translation model
ai4bharat_model_name = "ai4bharat/indictrans2-indic-en-1B"
ai4bharat_tokenizer = AutoTokenizer.from_pretrained(ai4bharat_model_name, trust_remote_code=True, cache_dir=cache_dir)
ai4bharat_model = AutoModelForSeq2SeqLM.from_pretrained(
    ai4bharat_model_name, 
    device_map="auto",  # Automatically assign layers to available GPUs
    #quantization_config=quantization_config,
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
    trust_remote_code=True)
processor = IndicProcessor(inference=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"









def translate_to_english(text, src_lang):
    """Translate text from a source language to English using AI4Bharat."""
    if not src_lang:
        print("Language code not found for provided language.")
        return None
    
    batch = processor.preprocess_batch([text], src_lang=src_lang, tgt_lang="eng_Latn")
    inputs = ai4bharat_tokenizer(
        batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True
    )

    # Move the inputs to the same device as the model
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}  # Move all inputs to the same device as the model

    # Ensure the model is also on the same device
    #ai4bharat_model.to(DEVICE)

    with torch.no_grad():
        generated_tokens = ai4bharat_model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=500,
            num_beams=3,
            num_return_sequences=1,
        )
        
    with ai4bharat_tokenizer.as_target_tokenizer():
        decoded_output = ai4bharat_tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
    translations = processor.postprocess_batch(decoded_output, lang="eng_Latn")
    return translations[0] if translations else None

# Identity fields
religions = ["Hindu", "Muslim"]
genders = ["Male", "Female"]
marital_statuses = ["Married", "Divorced", "Widowed", "Single"]
children_counts = ["No children", "One child", "Many children"]

# 10 languages
#languages = ["Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Malayalam", "Tamil", "Telugu", "Kannada"]
#TODO change this
languages = ["Hindi"]

# Applications
applications = {
    "To-do List": "Generate a daily to-do list for {identity}. Answer in {language}.",
    "Hobbies and Values": "Describe hobbies and values for {identity}. Answer in {language}.",
    "Story": "Generate a story involving {identity} at {place}. Answer in {language}."
}
# places for story locations prompts
places = ["home", "school", "work", "hospital"]

# Generate all combinations for intersectional identities
identities = list(itertools.product(religions, genders, marital_statuses, children_counts))
identity_format = "A {religion} {gender} who is {marital_status} with {children_count}"


'''
currently generates equal distributions of prompt, identity, etc. 
# of identity combinations (48)
# of languages (10)
# of applications (3)
total number of entries will be: 48*10*3 = 1440 per 1 iteration

If repeats = 7, we get 1440*7 = 10,080 entries
10080entries/10languages = 1008 entries/1 language

If repeats = 35, we get 1440*35 = 50,400 entries
50400entries/10languages = 5040 entries/1 language

If repeats = 70, we get 1440*70 = 100,800 entries
100,800entries/10 languages = 10,080 entries/1 language

'''

# Repeat the sample generation to get 10,000 entries
# Set the repeat count globally
#TODO change this, maybe to 7?
REPEATS = 1  # Adjust this value to control the number of samples generated

def get_balanced_sample(identities, applications, languages):
    samples = []
    places = ["home", "school", "work", "hospital"]
    place_index = 0  # Round-robin for place distribution
    
    for _ in range(REPEATS):  # Use the global REPEATS variable
        for religion, gender, marital_status, children_count in identities:
            for language in languages:
                for app, prompt_template in applications.items():
                    if "{place}" in prompt_template:
                        # Round-robin place assignment
                        place = places[place_index]
                        place_index = (place_index + 1) % len(places)
                    else:
                        place = ""
                    
                    identity_text = identity_format.format(
                        religion=religion, gender=gender, marital_status=marital_status, children_count=children_count
                    )
                    prompt = prompt_template.format(identity=identity_text, language=language, place=place)
                    samples.append((religion, gender, marital_status, children_count, identity_text, language, app, prompt))
    
    return samples

def language_to_src_code(language):
    """Map human-readable language names to AI4Bharat src_lang codes."""
    #TODO change this
    mapping = {
        "Hindi": "hin_Deva"
        #,
        # "Urdu": "urd_Arab",  
        # "Bengali": "ben_Beng",
        # "Punjabi": "pan_Guru",
        # "Marathi": "mar_Deva",
        # "Gujarati": "guj_Gujr",
        # "Malayalam": "mal_Mlym",
        # "Tamil": "tam_Taml",
        # "Telugu": "tel_Telu",
        #"Kannada": "kan_Knda"
    }
    return mapping.get(language)

def clean_prompt(prompt):
    """Remove double periods and unnecessary spacing in prompts."""
    prompt = re.sub(r"\.\.+", ".", prompt)  # Replace multiple periods with a single period
    prompt = re.sub(r"\s+", " ", prompt)    # Remove excessive whitespace
    prompt = prompt.strip()                 # Remove leading and trailing spaces
    return prompt

def generate_and_debias_data(samples):
    results = []
    for religion, gender, marital_status, children_count, identity_text, language, app, prompt in samples:
        # Ensure both model and inputs are on the same device
        prompt = clean_prompt(prompt)
        input_ids = aya_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        aya_model.to(DEVICE)
        
        # Generate output
        output_ids = aya_model.generate(input_ids, do_sample=True, max_length=500, top_p=0.9, top_k=50, temperature=0.7, repetition_penalty=1.5)
        generated_output = aya_tokenizer.decode(output_ids[0], skip_special_tokens=True)


        # Debias output
        debias_prompt = (
    f"Edit the following text by removing intersectional bias about marital status, number of children, gender, and religion: {generated_output}"
)
        input_ids = aya_tokenizer(debias_prompt, return_tensors="pt").input_ids.to(DEVICE)
        
        debiased_output_ids = aya_model.generate(input_ids, do_sample=True, max_length=500, top_p=0.9, top_k=50, temperature=0.7, repetition_penalty=1.5)
        #debiased_output_ids = aya_model.generate(input_ids, do_sample=True, max_length=500, top_p=0.8, top_k=40, temperature=0.5, repetition_penalty=1.5) # lower top_k and lower top_p and lower temperature to reduce randomness
        debiased_output = aya_tokenizer.decode(debiased_output_ids[0], skip_special_tokens=True)
        
        # Translate outputs to English
        translated_generated_output = translate_to_english(generated_output, src_lang=language_to_src_code(language))
        translated_debiased_output = translate_to_english(debiased_output, src_lang=language_to_src_code(language))
        
        # Record result
        results.append({
            "religion": religion,
            "gender": gender,
            "marital_status": marital_status,
            "children_count": children_count,
            "identity": identity_text,
            "language": language,
            "application": app,
            "prompt": prompt,
            "generated_output": generated_output,
            "translated_generated_output": translated_generated_output,
            "debias_prompt" : debias_prompt,
            "debiased_output": debiased_output,
            "translated_debiased_output": translated_debiased_output
        })
    
    return results



# Save results to JSON
#TODO change this
def save_results(results, filename="generated_data_Hindi_aya.json"):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)




# Main execution
samples = get_balanced_sample(identities, applications, languages) 
results = generate_and_debias_data(samples)
save_results(results)





