import os 
import json
import random 

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Using the model you have: gemma3:latest
# (llama3.2:3b was not in your 'ollama list')
translation_model = ChatOllama(
    model='gemma3:latest',
    temperature=0.7
)

@tool
def get_n_random_words(language: str, n: int) -> list:
    """
    Select random words from language JSON file.
    Use this tool when you need random vocabulary words.
    Input language MUST be 'spanish' or 'german' (lowercase only).
    """ 
    base_dir = os.path.dirname(__file__)  # ensures correct relative path
    path = os.path.join(base_dir, "data", f"{language}.json")

    with open(path, "r", encoding="utf-8") as f:
        word_list = json.load(f)

    keys = list(word_list.keys())
    n = min(n, len(keys))
    random_keys = random.sample(keys, n)

    return [word_list[k]["word"] for k in random_keys]

@tool
def get_n_random_words_by_difficulty_level(language: str, difficulty_level: str, n: int) -> list:
    """
    Get random words filtered by difficulty level.
    """

    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "data", f"{language}.json")

    with open(path, "r", encoding="utf-8") as f:
        word_list = json.load(f)

    words_filtered = [
        val["word"]
        for val in word_list.values()
        if val.get("difficulty_level") == difficulty_level
    ]

    n = min(n, len(words_filtered))
    return random.sample(words_filtered, n)

@tool
def translate_words(random_words: list, source_language: str, target_language: str) -> dict:
    """
    Translates a list of words from a source language to a target language.
    """
    prompt = (
        f"You are a precise translation engine.\n"
        f"Translate each of the following {len(random_words)} words from {source_language} to {target_language}.\n"
        f"Return ONLY valid JSON with this exact structure:\n"
        f'{{"translations": [{{"source": "<original>", "target": "<translated>"}}, ...]}}\n'
        f"No explanations, no extra fields, no markdown.\n"
        f"Words: {json.dumps(random_words, ensure_ascii=False)}"
    )
    
    response = translation_model.invoke([HumanMessage(content=prompt)])
    text = response.content
    
    try:
        parsed = json.loads(text)
    except Exception:
        import re 
        match = re.search(r'\{.*\}', text, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {"translations": []}
        
    translations_list = parsed.get("translations", [])
    model_map = {item.get("source", ""): item.get("target", "") for item in translations_list if isinstance(item, dict)}
    
    ordered_translations = [
        {"source": w, "target": model_map.get(w, model_map.get(w.capitalize(), w))}
        for w in random_words
    ]
    
    return {"translations": ordered_translations}

@tool
def save_word_to_study_list(banglish: str, english: str, difficulty: str = "medium"):
    """Saves a Banglish word and its English meaning to a local file for practice."""
    entry = f"{banglish} | {english} | {difficulty}\n"
    with open("study_list.txt", "a", encoding="utf-8") as f:
        f.write(entry)
    return f"Successfully saved '{banglish}' to your study list."

# Final tool registry
local_tools = [
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words,
    save_word_to_study_list
]

def setup_tools():
    return local_tools