"""
Experiment 4: Morphological Analysis & Word Generation

Performs morphological decomposition (base/root, prefixes, suffixes)
and inflectional/derivational word generation for culinary/restaurant terms.
Supports search query expansion.
"""

from typing import List, Dict, Any, Set
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Curated morphological rules and dictionary for food and restaurant domain
FOOD_MORPHOLOGY_LEXICON = {
    "serve": {
        "root": "serve",
        "category": "Verb",
        "prefixes": ["re"],
        "suffixes": ["s", "d", "ing", "r", "ice", "ant"],
        "generated_forms": ["serve", "serves", "served", "serving", "server", "servers", "service", "services", "servant"]
    },
    "cook": {
        "root": "cook",
        "category": "Verb",
        "prefixes": ["un", "over", "under", "pre"],
        "suffixes": ["s", "ed", "ing", "er", "ery"],
        "generated_forms": ["cook", "cooks", "cooked", "cooking", "cooker", "cookery", "overcooked", "undercooked"]
    },
    "recommend": {
        "root": "recommend",
        "category": "Verb",
        "prefixes": [],
        "suffixes": ["s", "ed", "ing", "ation", "able"],
        "generated_forms": ["recommend", "recommends", "recommended", "recommending", "recommendation", "recommendations", "recommendable"]
    },
    "bake": {
        "root": "bake",
        "category": "Verb",
        "prefixes": [],
        "suffixes": ["s", "d", "ing", "r", "ry"],
        "generated_forms": ["bake", "bakes", "baked", "baking", "baker", "bakery", "bakeries"]
    },
    "taste": {
        "root": "taste",
        "category": "Noun/Verb",
        "prefixes": [],
        "suffixes": ["s", "d", "ing", "y", "ful", "less"],
        "generated_forms": ["taste", "tastes", "tasted", "tasting", "tasty", "tasteful", "tasteless"]
    },
    "spice": {
        "root": "spice",
        "category": "Noun/Verb",
        "prefixes": [],
        "suffixes": ["s", "d", "ing", "y", "iness"],
        "generated_forms": ["spice", "spices", "spiced", "spicing", "spicy", "spiciness"]
    },
    "order": {
        "root": "order",
        "category": "Noun/Verb",
        "prefixes": ["re", "pre"],
        "suffixes": ["s", "ed", "ing"],
        "generated_forms": ["order", "orders", "ordered", "ordering", "reorder", "preorder"]
    },
    "dine": {
        "root": "dine",
        "category": "Verb",
        "prefixes": [],
        "suffixes": ["s", "d", "ing", "r"],
        "generated_forms": ["dine", "dines", "dined", "dining", "diner", "diners"]
    },
    "fry": {
        "root": "fry",
        "category": "Verb",
        "prefixes": ["deep", "stir"],
        "suffixes": ["s", "ed", "ing", "er"],
        "generated_forms": ["fry", "fries", "fried", "frying", "fryer", "deep-fried", "stir-fried"]
    },
    "drink": {
        "root": "drink",
        "category": "Verb/Noun",
        "prefixes": [],
        "suffixes": ["s", "ing", "er", "able"],
        "generated_forms": ["drink", "drinks", "drank", "drunk", "drinking", "drinker", "drinkable"]
    },
    "flavor": {
        "root": "flavor",
        "category": "Noun/Verb",
        "prefixes": ["un"],
        "suffixes": ["s", "ed", "ing", "ful", "less"],
        "generated_forms": ["flavor", "flavors", "flavored", "flavoring", "flavorful", "flavorless"]
    },
    "sweet": {
        "root": "sweet",
        "category": "Adjective",
        "prefixes": [],
        "suffixes": ["er", "est", "ly", "ness", "en"],
        "generated_forms": ["sweet", "sweeter", "sweetest", "sweetly", "sweetness", "sweeten", "sweets"]
    }
}

KNOWN_PREFIXES = ["un", "re", "dis", "pre", "over", "under", "non", "mis", "in", "im"]
KNOWN_SUFFIXES = ["ing", "ed", "es", "s", "er", "or", "tion", "ation", "ment", "able", "ible", "ful", "less", "ness", "ly", "y", "al", "ic"]


class MorphologicalAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def analyze_word(self, word: str) -> Dict[str, Any]:
        """
        Decomposes a single word into prefix, root/base, suffix, and generated forms.
        """
        clean_word = word.lower().strip()

        # Check curated domain lexicon first
        for key, entry in FOOD_MORPHOLOGY_LEXICON.items():
            if clean_word in entry["generated_forms"] or clean_word == key:
                prefix = ""
                for p in entry["prefixes"]:
                    if clean_word.startswith(p) and clean_word != p:
                        prefix = p
                        break
                suffix = ""
                for s in sorted(entry["suffixes"], key=len, reverse=True):
                    if clean_word.endswith(s) and clean_word != s:
                        suffix = s
                        break
                return {
                    "input_word": word,
                    "root_word": entry["root"],
                    "prefix": prefix if prefix else "None",
                    "suffix": suffix if suffix else "None",
                    "category": entry.get("category", "General"),
                    "generated_forms": entry["generated_forms"]
                }

        # Rule-based fallback using WordNet & Affix Stripping
        lemma = self.lemmatizer.lemmatize(clean_word, pos='v')
        if lemma == clean_word:
            lemma = self.lemmatizer.lemmatize(clean_word, pos='n')

        identified_prefix = "None"
        for p in sorted(KNOWN_PREFIXES, key=len, reverse=True):
            if clean_word.startswith(p) and len(clean_word) > len(p) + 2:
                potential_root = clean_word[len(p):]
                if wordnet.synsets(potential_root):
                    identified_prefix = p
                    lemma = potential_root
                    break

        identified_suffix = "None"
        for s in sorted(KNOWN_SUFFIXES, key=len, reverse=True):
            if clean_word.endswith(s) and len(clean_word) > len(s) + 2:
                identified_suffix = s
                break

        # Generate standard inflectional forms
        generated = self._generate_forms(lemma)

        return {
            "input_word": word,
            "root_word": lemma,
            "prefix": identified_prefix,
            "suffix": identified_suffix,
            "category": "Derived / Inflected",
            "generated_forms": sorted(list(generated))
        }

    def _generate_forms(self, root: str) -> Set[str]:
        """Generates standard English inflections for a base root word."""
        forms = {root}
        if root.endswith('e'):
            forms.add(root + 's')
            forms.add(root + 'd')
            forms.add(root[:-1] + 'ing')
            forms.add(root + 'r')
        elif root.endswith('y') and len(root) > 2 and root[-2] not in "aeiou":
            forms.add(root[:-1] + 'ies')
            forms.add(root[:-1] + 'ied')
            forms.add(root + 'ing')
        else:
            forms.add(root + 's')
            forms.add(root + 'ed')
            forms.add(root + 'ing')
            forms.add(root + 'er')

        # Add common derivational suffixes
        forms.add(root + 'able')
        return forms

    def expand_query(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Morphological search query expansion:
        Expands input tokens into their morphological variants to boost restaurant search recall.
        """
        expansion_map = {}
        all_expanded_tokens = set()

        for token in tokens:
            cleaned = token.lower().strip()
            if len(cleaned) < 3:
                expansion_map[token] = [token]
                all_expanded_tokens.add(cleaned)
                continue

            analysis = self.analyze_word(cleaned)
            forms = analysis["generated_forms"]
            expansion_map[token] = forms
            all_expanded_tokens.update(forms)

        return {
            "original_tokens": tokens,
            "expansion_map": expansion_map,
            "all_expanded_tokens": sorted(list(all_expanded_tokens)),
            "expanded_query_string": " ".join(sorted(list(all_expanded_tokens)))
        }


# Singleton instance
_morph_instance = None

def get_morphological_analyzer() -> MorphologicalAnalyzer:
    global _morph_instance
    if _morph_instance is None:
        _morph_instance = MorphologicalAnalyzer()
    return _morph_instance


def run_experiment_4(tokens: List[str]) -> Dict[str, Any]:
    """
    Runs Experiment 4 on a list of input tokens.
    Provides morphological breakdown for significant words and query expansion.
    """
    analyzer = get_morphological_analyzer()
    detailed_analyses = []

    for t in tokens:
        if len(t.strip()) > 2:
            res = analyzer.analyze_word(t.strip())
            detailed_analyses.append(res)

    expansion = analyzer.expand_query(tokens)

    return {
        "word_analyses": detailed_analyses,
        "query_expansion": expansion
    }
