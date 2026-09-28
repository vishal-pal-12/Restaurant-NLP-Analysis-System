"""
Experiment 5: N-Gram Language Model

Implements Unigram, Bigram, and Trigram models with sentence boundary markers (<s>, </s>),
frequency counting with Counter, Maximum Likelihood Estimation (MLE) probabilities,
common food phrase extraction, and next-word query prediction.
"""

from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams


class NGramLanguageModel:
    def __init__(self, add_boundaries: bool = True):
        self.add_boundaries = add_boundaries
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.bigram_transitions = defaultdict(Counter)
        self.trigram_transitions = defaultdict(Counter)
        self.total_words = 0

    def prepare_sentences(self, text: str) -> List[List[str]]:
        """
        Tokenizes text into sentences and words, adding sentence boundary markers <s> and </s>.
        """
        sentences = sent_tokenize(text)
        tokenized_sentences = []

        for s in sentences:
            words = [w.lower() for w in word_tokenize(s) if w.isalnum()]
            if not words:
                continue
            if self.add_boundaries:
                tokens = ["<s>"] + words + ["</s>"]
            else:
                tokens = words
            tokenized_sentences.append(tokens)

        return tokenized_sentences

    def fit_text(self, corpus_text: str):
        """Builds n-gram statistics from a corpus of restaurant reviews or food queries."""
        sentences = self.prepare_sentences(corpus_text)

        for sent in sentences:
            # Unigrams
            for w in sent:
                self.unigram_counts[w] += 1
                self.total_words += 1

            # Bigrams
            b_grams = list(ngrams(sent, 2))
            for b in b_grams:
                self.bigram_counts[b] += 1
                self.bigram_transitions[b[0]][b[1]] += 1

            # Trigrams
            t_grams = list(ngrams(sent, 3))
            for t in t_grams:
                self.trigram_counts[t] += 1
                self.trigram_transitions[(t[0], t[1])][t[2]] += 1

    def extract_ngrams(self, text: str) -> Dict[str, Any]:
        """
        Extracts unigrams, bigrams, and trigrams from input text with counts and MLE probabilities.
        """
        sentences = self.prepare_sentences(text)
        local_unigrams = []
        local_bigrams = []
        local_trigrams = []

        for sent in sentences:
            local_unigrams.extend(sent)
            local_bigrams.extend(list(ngrams(sent, 2)))
            local_trigrams.extend(list(ngrams(sent, 3)))

        u_counter = Counter(local_unigrams)
        b_counter = Counter(local_bigrams)
        t_counter = Counter(local_trigrams)

        total_u = sum(u_counter.values()) if u_counter else 1

        # Calculate MLE probabilities
        unigram_stats = [
            {"ngram": w, "count": count, "probability": round(count / total_u, 4)}
            for w, count in u_counter.most_common()
        ]

        bigram_stats = [
            {
                "ngram": f"{b[0]} {b[1]}",
                "count": count,
                "conditional_prob": round(count / (u_counter[b[0]] if u_counter[b[0]] else count), 4)
            }
            for b, count in b_counter.most_common()
        ]

        trigram_stats = [
            {
                "ngram": f"{t[0]} {t[1]} {t[2]}",
                "count": count,
                "conditional_prob": round(count / (b_counter[(t[0], t[1])] if b_counter[(t[0], t[1])] else count), 4)
            }
            for t, count in t_counter.most_common()
        ]

        return {
            "unigrams": unigram_stats,
            "bigrams": bigram_stats,
            "trigrams": trigram_stats,
            "total_unigrams": len(local_unigrams),
            "total_bigrams": len(local_bigrams),
            "total_trigrams": len(local_trigrams)
        }

    def predict_next_words(self, context: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predicts the next most likely words given preceding 1 or 2 words using Trigram/Bigram backoff.
        """
        words = [w.lower() for w in word_tokenize(context) if w.isalnum()]
        if not words:
            return []

        suggestions = []

        # Check Trigram transition if at least 2 words
        if len(words) >= 2:
            key = (words[-2], words[-1])
            if key in self.trigram_transitions:
                total = sum(self.trigram_transitions[key].values())
                for next_w, cnt in self.trigram_transitions[key].most_common(top_k):
                    suggestions.append({
                        "next_word": next_w,
                        "model": "Trigram",
                        "probability": round(cnt / total, 3)
                    })

        # Backoff to Bigram
        if len(suggestions) < top_k and len(words) >= 1:
            key1 = words[-1]
            if key1 in self.bigram_transitions:
                total1 = sum(self.bigram_transitions[key1].values())
                for next_w, cnt in self.bigram_transitions[key1].most_common(top_k):
                    if not any(s["next_word"] == next_w for s in suggestions):
                        suggestions.append({
                            "next_word": next_w,
                            "model": "Bigram",
                            "probability": round(cnt / total1, 3)
                        })

        return suggestions[:top_k]


# Pretrained default model on restaurant reviews
_ngram_instance = None

def get_ngram_model() -> NGramLanguageModel:
    global _ngram_instance
    if _ngram_instance is None:
        _ngram_instance = NGramLanguageModel(add_boundaries=True)
        # Train on default restaurant reviews corpus
        import os
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "restaurant_reviews.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            full_text = " ".join(df['review'].dropna().tolist())
            _ngram_instance.fit_text(full_text)
    return _ngram_instance


def run_experiment_5(text: str) -> Dict[str, Any]:
    """
    Runs Experiment 5 on the given text.
    Extracts n-grams with sentence boundary markers and provides autocomplete predictions.
    """
    model = get_ngram_model()
    # Also fit current text to ensure local statistics
    model.fit_text(text)
    results = model.extract_ngrams(text)

    # Next-word predictions for the last 1 or 2 tokens
    words = [w.lower() for w in word_tokenize(text) if w.isalnum()]
    predictions = model.predict_next_words(" ".join(words[-2:]) if len(words) >= 2 else (words[-1] if words else ""))

    results["next_word_predictions"] = predictions
    return results
