"""
Experiment 9: Text Similarity & Restaurant Matching

Implements TF-IDF vectorization and Cosine Similarity calculation to match:
1. User Food/Restaurant Query against restaurant profiles & menus (Restaurant Recommendation).
2. User Review against historical database reviews (Review Duplicate/Similarity Detection).
Outputs descending similarity ranking scores and recommendation cards.
"""

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_RESTAURANT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "restaurant_data.csv")
DEFAULT_REVIEWS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "restaurant_reviews.csv")


class TextSimilarityEngine:
    def __init__(self, rest_path: str = DEFAULT_RESTAURANT_PATH, rev_path: str = DEFAULT_REVIEWS_PATH):
        self.rest_path = rest_path
        self.rev_path = rev_path
        self.rest_df: pd.DataFrame = None
        self.rev_df: pd.DataFrame = None

        self.rest_vectorizer: TfidfVectorizer = None
        self.rest_tfidf_matrix = None

        self.rev_vectorizer: TfidfVectorizer = None
        self.rev_tfidf_matrix = None

        self.load_and_index()

    def load_and_index(self):
        """Loads CSV databases and computes TF-IDF matrices."""
        if os.path.exists(self.rest_path):
            self.rest_df = pd.read_csv(self.rest_path)
            # Combine name, cuisine, popular dishes, and description for rich matching
            combined_text = (
                self.rest_df['restaurant_name'].fillna('') + " " +
                self.rest_df['location'].fillna('') + " " +
                self.rest_df['cuisine'].fillna('') + " " +
                self.rest_df['popular_dishes'].fillna('') + " " +
                self.rest_df['description'].fillna('')
            )
            self.rest_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            self.rest_tfidf_matrix = self.rest_vectorizer.fit_transform(combined_text)

        if os.path.exists(self.rev_path):
            self.rev_df = pd.read_csv(self.rev_path)
            self.rev_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            self.rev_tfidf_matrix = self.rev_vectorizer.fit_transform(self.rev_df['review'].fillna(''))

    def match_restaurants(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity between a food/restaurant query and the restaurant catalog.
        Returns top-K matching restaurants sorted descending.
        """
        if self.rest_vectorizer is None or self.rest_tfidf_matrix is None:
            self.load_and_index()

        query_vec = self.rest_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.rest_tfidf_matrix)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            row = self.rest_df.iloc[idx]
            results.append({
                "rank": rank,
                "restaurant_name": row["restaurant_name"],
                "location": row["location"],
                "cuisine": row["cuisine"],
                "rating": row["rating"],
                "popular_dishes": row["popular_dishes"],
                "description": row["description"],
                "similarity_score": round(score, 4),
                "similarity_percentage": f"{round(score * 100, 1)}%"
            })

        return results

    def match_reviews(self, review_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity between input review and historical reviews.
        """
        if self.rev_vectorizer is None or self.rev_tfidf_matrix is None:
            self.load_and_index()

        q_vec = self.rev_vectorizer.transform([review_text])
        scores = cosine_similarity(q_vec, self.rev_tfidf_matrix)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            row = self.rev_df.iloc[idx]
            results.append({
                "rank": rank,
                "historical_review": row["review"],
                "sentiment": row["sentiment"],
                "similarity_score": round(score, 4),
                "similarity_percentage": f"{round(score * 100, 1)}%"
            })

        return results


# Singleton instance
_similarity_instance = None

def get_similarity_engine() -> TextSimilarityEngine:
    global _similarity_instance
    if _similarity_instance is None:
        _similarity_instance = TextSimilarityEngine()
    return _similarity_instance


def run_experiment_9(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Runs Experiment 9:
    Computes text similarity against restaurant catalog and review history.
    """
    engine = get_similarity_engine()
    rest_matches = engine.match_restaurants(query, top_k=top_k)
    rev_matches = engine.match_reviews(query, top_k=top_k)

    return {
        "query": query,
        "restaurant_matches": rest_matches,
        "review_matches": rev_matches,
        "top_recommended_restaurant": rest_matches[0]["restaurant_name"] if rest_matches else "N/A",
        "highest_similarity_score": rest_matches[0]["similarity_score"] if rest_matches else 0.0
    }
