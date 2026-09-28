"""
Experiment 2: Tokenization, Filtration & Script Validation
Experiment 3: Stop Word Removal, Stemming & Lemmatization

Modular implementation for preprocessing restaurant reviews and food queries.
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Ensure required NLTK resources
def ensure_nltk_resources():
    for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk_resources()


# ====================================================================
# EXPERIMENT 2: TOKENIZATION, FILTRATION & SCRIPT VALIDATION
# ====================================================================

def validate_script(text: str) -> Dict[str, Any]:
    """
    Validates whether the input text contains supported English (Latin) script.
    Identifies foreign scripts, emojis, and non-text noise.
    """
    if not text or not text.strip():
        return {
            "is_valid": False,
            "script": "Empty",
            "english_char_percentage": 0.0,
            "status": "Invalid: Input is empty."
        }

    total_chars = len(text.strip())
    latin_chars = 0
    non_latin_scripts = set()

    for ch in text:
        if ch.isspace() or ch in string.punctuation or ch.isdigit():
            latin_chars += 1
            continue
        try:
            name = unicodedata.name(ch, "")
            if "LATIN" in name:
                latin_chars += 1
            elif "DEVANAGARI" in name:
                non_latin_scripts.add("Devanagari (Hindi/Marathi)")
            elif "ARABIC" in name:
                non_latin_scripts.add("Arabic")
            elif "CYRILLIC" in name:
                non_latin_scripts.add("Cyrillic")
            elif "CJK" in name:
                non_latin_scripts.add("East Asian (CJK)")
            elif "EMOJI" in name or "SYMBOL" in name:
                non_latin_scripts.add("Symbols/Emoji")
            else:
                non_latin_scripts.add(name.split()[0] if name else "Unknown")
        except Exception:
            pass

    pct = round((latin_chars / total_chars) * 100, 2)
    is_valid = pct >= 60.0

    if is_valid:
        status_msg = f"Valid English/Latin Script ({pct}% standard characters)."
    else:
        scripts_detected = ", ".join(non_latin_scripts) if non_latin_scripts else "Unsupported symbols"
        status_msg = f"Unsupported Script ({pct}% English characters). Detected: {scripts_detected}."

    return {
        "is_valid": is_valid,
        "script": "Latin/English" if is_valid else "Mixed / Unsupported",
        "english_char_percentage": pct,
        "non_latin_detected": list(non_latin_scripts),
        "status": status_msg
    }


def tokenize_text(text: str) -> Dict[str, Any]:
    """
    Performs sentence tokenization and word tokenization using NLTK.
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return {
        "sentence_tokens": sentences,
        "word_tokens": words,
        "sentence_count": len(sentences),
        "word_count": len(words)
    }


def filter_tokens(tokens: List[str], remove_punct: bool = True, remove_digits: bool = False, lower: bool = True) -> List[str]:
    """
    Filters tokens by removing unnecessary punctuation, non-alphanumeric noise, and optional casing.
    """
    filtered = []
    punct_set = set(string.punctuation) | {'``', "''", '’', '‘', '“', '”', '…', '—', '--'}

    for t in tokens:
        clean = t.strip()
        if lower:
            clean = clean.lower()
        if remove_punct and (clean in punct_set or all(c in punct_set for c in clean)):
            continue
        if remove_digits and clean.isdigit():
            continue
        if clean:
            filtered.append(clean)
    return filtered


def run_experiment_2(text: str) -> Dict[str, Any]:
    """
    Runs the full Experiment 2 pipeline:
    - Script Validation
    - Sentence Tokenization
    - Word Tokenization
    - Filtration
    """
    script_eval = validate_script(text)
    tok = tokenize_text(text)
    filtered = filter_tokens(tok["word_tokens"], remove_punct=True, lower=False)

    return {
        "original_text": text,
        "script_validation": script_eval,
        "sentence_tokens": tok["sentence_tokens"],
        "word_tokens": tok["word_tokens"],
        "filtered_tokens": filtered,
        "total_sentences": tok["sentence_count"],
        "total_words": tok["word_count"],
        "total_filtered": len(filtered)
    }


# ====================================================================
# EXPERIMENT 3: STOP WORD REMOVAL, STEMMING & LEMMATIZATION
# ====================================================================

def get_wordnet_pos(treebank_tag: str) -> str:
    """Maps Penn Treebank POS tags to WordNet POS tags for accurate lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def process_stop_stem_lemma(tokens: List[str], preserve_negations: bool = True) -> Dict[str, Any]:
    """
    Executes Experiment 3:
    1. Stop Word Removal (with optional negation preservation for sentiment context)
    2. Porter Stemming
    3. WordNet Lemmatization with POS awareness
    """
    stop_words = set(stopwords.words('english'))
    if preserve_negations:
        negations = {"not", "no", "nor", "neither", "never", "none", "hardly", "scarcely", "barely", "t"}
        stop_words = stop_words - negations

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # POS tag for accurate lemmatization
    pos_tags = pos_tag(tokens)

    comparison_records = []
    tokens_no_stopwords = []
    stemmed_tokens = []
    lemmatized_tokens = []

    for word, tag in pos_tags:
        is_stop = word.lower() in stop_words
        wn_pos = get_wordnet_pos(tag)
        stem = stemmer.stem(word)
        lemma = lemmatizer.lemmatize(word.lower(), pos=wn_pos)

        record = {
            "original": word,
            "is_stopword": is_stop,
            "pos_tag": tag,
            "stemmed": stem,
            "lemmatized": lemma
        }
        comparison_records.append(record)

        if not is_stop:
            tokens_no_stopwords.append(word)
            stemmed_tokens.append(stem)
            lemmatized_tokens.append(lemma)

    return {
        "tokens_without_stopwords": tokens_no_stopwords,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "comparison_table": comparison_records,
        "removed_stopwords_count": len([r for r in comparison_records if r["is_stopword"]]),
        "retained_tokens_count": len(tokens_no_stopwords)
    }


def run_experiment_3(filtered_tokens: List[str]) -> Dict[str, Any]:
    """
    Convenience wrapper to run Experiment 3 directly on filtered tokens.
    """
    return process_stop_stem_lemma(filtered_tokens)
