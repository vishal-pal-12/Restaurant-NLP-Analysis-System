"""
Experiment 1: Sentiment Analysis

Implements TF-IDF vectorization, Logistic Regression & Naive Bayes classifiers,
model evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix),
and real-time sentiment prediction for restaurant reviews.
"""

import os
import pickle
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "restaurant_reviews.csv")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sentiment_model.pkl")


class SentimentAnalyzer:
    def __init__(self, data_path: str = DEFAULT_DATA_PATH, model_path: str = DEFAULT_MODEL_PATH):
        self.data_path = data_path
        self.model_path = model_path
        self.vectorizer: TfidfVectorizer = None
        self.model: LogisticRegression = None
        self.nb_model: MultinomialNB = None
        self.evaluation_results: Dict[str, Any] = {}
        self.classes_ = ["Negative", "Neutral", "Positive"]

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Reviews dataset not found at {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['review'] = df['review'].astype(str).str.strip()
        df['sentiment'] = df['sentiment'].astype(str).str.strip()
        return df

    def train(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Trains Logistic Regression and Naive Bayes on TF-IDF features.
        Computes evaluation metrics and confusion matrix.
        """
        df = self.load_data()
        X = df['review']
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=2500,
            sublinear_tf=True
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Logistic Regression
        self.model = LogisticRegression(max_iter=1000, C=1.5, random_state=random_state)
        self.model.fit(X_train_vec, y_train)
        y_pred = self.model.predict(X_test_vec)

        # Naive Bayes
        self.nb_model = MultinomialNB(alpha=0.8)
        self.nb_model.fit(X_train_vec, y_train)
        y_pred_nb = self.nb_model.predict(X_test_vec)

        # Metrics for Logistic Regression
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Metrics for Naive Bayes
        nb_acc = accuracy_score(y_test, y_pred_nb)

        self.evaluation_results = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "classes": self.classes_,
            "classification_report": report,
            "nb_accuracy": round(float(nb_acc), 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        # Save trained artifacts
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "model": self.model,
                "nb_model": self.nb_model,
                "evaluation": self.evaluation_results
            }, f)

        return self.evaluation_results

    def load_or_train(self) -> None:
        """Loads saved models if available, otherwise trains afresh."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data["vectorizer"]
                    self.model = data["model"]
                    self.nb_model = data.get("nb_model")
                    self.evaluation_results = data.get("evaluation", {})
                return
            except Exception:
                pass
        self.train()

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts sentiment for a given user review text.
        Returns predicted label, confidence probability, and breakdown.
        """
        if self.model is None or self.vectorizer is None:
            self.load_or_train()

        vec = self.vectorizer.transform([text])
        pred_label = self.model.predict(vec)[0]
        probs = self.model.predict_proba(vec)[0]

        prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(self.model.classes_, probs)}
        confidence = float(np.max(probs))

        # Sentiment polarity score approximation
        pos_p = prob_dict.get("Positive", 0.0)
        neg_p = prob_dict.get("Negative", 0.0)
        polarity = round(pos_p - neg_p, 4)

        return {
            "text": text,
            "predicted_sentiment": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "polarity_score": polarity,
            "evaluation_metrics": self.evaluation_results
        }


# Singleton instance for quick pipeline access
_analyzer_instance = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SentimentAnalyzer()
        _analyzer_instance.load_or_train()
    return _analyzer_instance


def run_experiment_1(text: str) -> Dict[str, Any]:
    """Runs Experiment 1 on a given text."""
    analyzer = get_sentiment_analyzer()
    return analyzer.predict(text)
