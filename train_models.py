"""
Train and Evaluate All Machine Learning & Deep Learning Models
1. Sentiment Analysis Model (Exp 1)
2. Chunking Feature & Training Size Benchmark (Exp 7)
3. Word Sense Disambiguation LSTM & GRU Models (Exp 10)
Saves models to `models/` and charts/metrics to `outputs/`.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from modules.sentiment import SentimentAnalyzer
from modules.chunking import run_chunking_training_experiment
from modules.wsd import WSDDisambiguator

BASE_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def train_sentiment():
    print("=" * 60)
    print("1. TRAINING SENTIMENT ANALYSIS MODEL (TF-IDF + Logistic Regression)")
    print("=" * 60)
    analyzer = SentimentAnalyzer()
    metrics = analyzer.train(test_size=0.2, random_state=42)
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score: {metrics['f1_score'] * 100:.2f}%")
    print(f"Naive Bayes Baseline Accuracy: {metrics['nb_accuracy'] * 100:.2f}%")

    # Plot Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=metrics['classes'], yticklabels=metrics['classes'])
    plt.title("Sentiment Classification - Confusion Matrix")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")
    plt.tight_layout()
    cm_plot_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    return metrics


def train_chunking_benchmark():
    print("\n" + "=" * 60)
    print("2. RUNNING CHUNKING TRAINING SIZE & FEATURE SELECTION EXPERIMENT")
    print("=" * 60)
    exp_res = run_chunking_training_experiment()

    df_size = pd.DataFrame(exp_res["size_comparison_table"])
    print("\nTraining Size vs Accuracy Comparison Table:")
    print(df_size[["training_size_pct", "train_samples", "test_samples", "accuracy", "f1_score"]].to_string(index=False))

    # Plot Training Size vs Accuracy
    plt.figure(figsize=(7, 4.5))
    plt.plot([60, 70, 80], df_size["accuracy"], marker='o', linewidth=2.5, color='#1E88E5', label='Accuracy (%)')
    plt.plot([60, 70, 80], df_size["f1_score"], marker='s', linewidth=2.5, color='#FF4B4B', label='F1-Score (%)')
    plt.title("Chunking Model Performance vs Training Data Size")
    plt.xlabel("Training Set Size (%)")
    plt.ylabel("Score (%)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    chunk_plot_path = os.path.join(OUTPUTS_DIR, "chunking_training_size_curve.png")
    plt.savefig(chunk_plot_path, dpi=300)
    plt.close()
    print(f"Saved chunking size curve to: {chunk_plot_path}")

    return exp_res


def train_wsd_deep_learning():
    print("\n" + "=" * 60)
    print("3. TRAINING WORD SENSE DISAMBIGUATION (LSTM & GRU DEEP LEARNING)")
    print("=" * 60)
    disambiguator = WSDDisambiguator()

    # Train LSTM
    print("Training Bidirectional LSTM Model...")
    lstm_metrics = disambiguator.train_model(rnn_type="LSTM", epochs=35, lr=0.005)
    print(f"LSTM Training Accuracy: {lstm_metrics['training_accuracy']}%")
    print(f"LSTM Validation Accuracy: {lstm_metrics['validation_accuracy']}%")
    print(f"LSTM Test Accuracy: {lstm_metrics['test_accuracy']}%")

    # Plot WSD Training Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(lstm_metrics["loss_history"]) + 1), lstm_metrics["loss_history"],
             color='#9C27B0', linewidth=2.2, label='Cross-Entropy Loss')
    plt.title("WSD Neural Model (LSTM) - Training Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    wsd_plot_path = os.path.join(OUTPUTS_DIR, "wsd_loss_curve.png")
    plt.savefig(wsd_plot_path, dpi=300)
    plt.close()
    print(f"Saved WSD loss curve to: {wsd_plot_path}")

    # Quick test predictions
    test_cases = [
        ("The restaurant served a delicious main course with red wine.", "course"),
        ("I registered for an advanced machine learning course at university.", "course"),
        ("The waiter handed us the restaurant bill after the meal.", "bill"),
        ("The parliamentary assembly passed the water rights bill.", "bill")
    ]

    print("\nSample WSD Real-Time Predictions:")
    for sent, word in test_cases:
        res = disambiguator.disambiguate(sent, target_word=word)
        print(f"  Sentence: \"{sent}\"")
        print(f"  Target: '{word}' -> Predicted Sense: {res['predicted_sense']} ({res['confidence']}%) | Meaning: {res['sense_meaning']}\n")

    return lstm_metrics


if __name__ == "__main__":
    s_metrics = train_sentiment()
    c_metrics = train_chunking_benchmark()
    w_metrics = train_wsd_deep_learning()

    all_metrics = {
        "sentiment": s_metrics,
        "chunking": c_metrics["size_comparison_table"],
        "wsd": w_metrics
    }
    with open(os.path.join(OUTPUTS_DIR, "model_benchmarks.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("=" * 60)
    print("ALL MODELS SUCCESSFULLY TRAINED AND BENCHMARKED!")
    print(f"Saved summary metrics to: {os.path.join(OUTPUTS_DIR, 'model_benchmarks.json')}")
    print("=" * 60)
