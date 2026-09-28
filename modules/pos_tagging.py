"""
Experiment 6: Part-of-Speech (POS) Tagging

Performs POS tagging using NLTK Penn Treebank tagger.
Maps tags to plain-English descriptions and syntactic categories.
Feeds structured POS tuples to the downstream Chunking module.
"""

from typing import List, Dict, Any, Tuple
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

PENN_TREEBANK_TAG_DICT = {
    "CC": ("Coordinating conjunction", "Conjunction"),
    "CD": ("Cardinal number", "Numeral"),
    "DT": ("Determiner", "Determiner"),
    "EX": ("Existential there", "Adverb"),
    "FW": ("Foreign word", "Other"),
    "IN": ("Preposition or subordinating conjunction", "Preposition"),
    "JJ": ("Adjective", "Adjective"),
    "JJR": ("Adjective, comparative", "Adjective"),
    "JJS": ("Adjective, superlative", "Adjective"),
    "LS": ("List item marker", "Other"),
    "MD": ("Modal", "Verb"),
    "NN": ("Noun, singular or mass", "Noun"),
    "NNS": ("Noun, plural", "Noun"),
    "NNP": ("Proper noun, singular", "Noun"),
    "NNPS": ("Proper noun, plural", "Noun"),
    "PDT": ("Predeterminer", "Determiner"),
    "POS": ("Possessive ending", "Other"),
    "PRP": ("Personal pronoun", "Pronoun"),
    "PRP$": ("Possessive pronoun", "Pronoun"),
    "RB": ("Adverb", "Adverb"),
    "RBR": ("Adverb, comparative", "Adverb"),
    "RBS": ("Adverb, superlative", "Adverb"),
    "RP": ("Particle", "Adverb"),
    "SYM": ("Symbol", "Other"),
    "TO": ("to", "Preposition"),
    "UH": ("Interjection", "Interjection"),
    "VB": ("Verb, base form", "Verb"),
    "VBD": ("Verb, past tense", "Verb"),
    "VBG": ("Verb, gerund or present participle", "Verb"),
    "VBN": ("Verb, past participle", "Verb"),
    "VBP": ("Verb, non-3rd person singular present", "Verb"),
    "VBZ": ("Verb, 3rd person singular present", "Verb"),
    "WDT": ("Wh-determiner", "Determiner"),
    "WP": ("Wh-pronoun", "Pronoun"),
    "WP$": ("Possessive wh-pronoun", "Pronoun"),
    "WRB": ("Wh-adverb", "Adverb")
}


def tag_tokens(tokens: List[str]) -> List[Tuple[str, str]]:
    """Generates standard (word, pos_tag) tuples using NLTK."""
    return pos_tag(tokens)


def analyze_pos(tokens: List[str]) -> Dict[str, Any]:
    """
    Annotates tokens with POS tags, plain English meanings, and high-level syntactic categories.
    """
    raw_tagged = tag_tokens(tokens)
    tag_details = []
    category_counts = {}

    for word, tag in raw_tagged:
        desc, cat = PENN_TREEBANK_TAG_DICT.get(tag, ("Punctuation / Other", "Other"))
        category_counts[cat] = category_counts.get(cat, 0) + 1

        tag_details.append({
            "word": word,
            "tag": tag,
            "description": desc,
            "category": cat
        })

    return {
        "tagged_tokens": raw_tagged,
        "detailed_tags": tag_details,
        "category_distribution": category_counts,
        "total_tokens": len(tokens)
    }


def run_experiment_6(tokens: List[str]) -> Dict[str, Any]:
    """
    Runs Experiment 6 on a token list.
    Returns structured POS tags and explanations.
    """
    return analyze_pos(tokens)
