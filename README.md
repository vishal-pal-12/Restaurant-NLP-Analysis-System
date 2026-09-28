# Restaurant Review & Food Query Analysis System Using Natural Language Processing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.10-green.svg)](https://www.nltk.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, production-grade College NLP Mini Project that integrates **all 10 university NLP laboratory experiments** into one unified, real-world Restaurant Review and Food Query Analysis application.

Available as:
1. **Interactive Streamlit Web Dashboard** (`app.py`)
2. **Real-Time Interactive VS Code Terminal Console** (`predict.py`)
3. **End-to-End Master Jupyter Notebook** (`restaurant_nlp_mini_project.ipynb`)
4. **Automated Unit & Integration Test Suite** (`test_pipeline.py`)

---

## 📌 Problem Statement & Objectives

Restaurants and food platforms (such as Zomato, Swiggy, and Google Reviews) receive thousands of customer reviews and search queries daily. Manually analyzing this unstructured text is infeasible, and traditional keyword search fails due to:
1. **Unstructured Mixed Sentiment:** Reviews often have positive and negative aspects intertwined (*"The pasta was delicious but the service was terrible"*).
2. **Linguistic Variations & Inflections:** Users query variations (*serve, serving, cooked*) or multi-word culinary concepts (*"wood-fired thin-crust pizza"*).
3. **Contextual Polysemy (Ambiguity):** Words have multiple meanings based on context (e.g., **"course"** as a restaurant meal course vs. an educational degree course).

This project integrates 10 core Natural Language Processing techniques into a single pipeline to automate sentiment classification, entity extraction, phrase chunking, semantic restaurant recommendation, and neural context disambiguation.

---

## 🔬 The 10 Integrated NLP Experiments

| Experiment | Module | NLP Technique / Model | Real-World Problem Solved |
|---|---|---|---|
| **Exp 1** | `modules/sentiment.py` | TF-IDF + Logistic Regression / Naive Bayes | Classifies reviews as Positive, Negative, or Neutral with confidence and polarity scores (85.71% test accuracy). |
| **Exp 2** | `modules/preprocessing.py` | NLTK Tokenizer + Unicode Script Validator | Verifies standard English characters and isolates clean linguistic sentences and word tokens. |
| **Exp 3** | `modules/preprocessing.py` | Porter Stemmer + WordNet Lemmatizer | Normalizes words to canonical root forms while preserving critical negation words (*"not", "never"*). |
| **Exp 4** | `modules/morphology.py` | Root/Affix Decomposition & Rule Generator | Expands search queries using inflectional word variants (*cook -> cooks, cooked, cooking, cooker*). |
| **Exp 5** | `modules/ngram.py` | MLE Probabilities with `<s>` and `</s>` Boundaries | Extracts Unigrams, Bigrams, and Trigrams and powers autocomplete next-word search predictions. |
| **Exp 6** | `modules/pos_tagging.py` | Penn Treebank POS Tagger | Categorizes syntactic components (Nouns, Verbs, Adjectives) with human-readable descriptions. |
| **Exp 7** | `modules/chunking.py` | Grammar Chunking & Feature Selection Benchmark | Extracts descriptive food noun phrases (*"spicy Indian curry"*), and benchmarks 60%, 70%, and 80% training sizes. |
| **Exp 8** | `modules/ner.py` | Domain Gazetteer + Maximum Entropy NER | Extracts structured domain entities: `RESTAURANT`, `FOOD`, `LOCATION`, `DATE`, `PERSON`. |
| **Exp 9** | `modules/similarity.py` | TF-IDF Vector Space + Cosine Similarity | Matches user food queries against restaurant menu catalogs and ranks top matching venues. |
| **Exp 10** | `modules/wsd.py` | Bidirectional LSTM/GRU Neural Network | Resolves polysemous ambiguous words (*course, bill, order*) based on sentence context. |

---

## 📁 Repository Structure

```text
Restaurant-NLP-Analysis-System/
│
├── app.py                            # Interactive Streamlit Web Application
├── predict.py                        # Interactive VS Code Terminal Prediction Console
├── train_models.py                   # Standalone trainer for ML & Deep Learning models
├── test_pipeline.py                  # Automated unit and integration test suite (11/11 passing)
├── restaurant_nlp_mini_project.ipynb # Master 28-cell End-to-End Jupyter Notebook
│
├── run_app.bat                       # One-click Windows launcher for Streamlit App
├── run_predict.bat                   # One-click Windows launcher for Terminal Console
├── open_jupyter.bat                  # One-click Windows launcher for Jupyter Notebook
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation and guide
├── FINAL_REPORT.md                   # Comprehensive 24-section formal college submission report
├── .gitignore                        # Git exclusion rules
│
├── .vscode/                          # Pre-configured VS Code debug & task configs
│   ├── launch.json
│   ├── settings.json
│   └── tasks.json
│
├── data/                             # Curated Domain Datasets
│   ├── restaurant_reviews.csv        # 70 labeled customer reviews
│   ├── restaurant_data.csv           # 35 restaurant profiles across Pune, Mumbai, Delhi, Bangalore
│   └── wsd_dataset.csv               # 60 contextual sentences for polysemous words
│
├── modules/                          # 10 Interconnected NLP Pipeline Modules
│   ├── __init__.py
│   ├── sentiment.py                  # Exp 1: Sentiment Analysis
│   ├── preprocessing.py              # Exp 2 & 3: Tokenization, Validation, Stemming & Lemmatization
│   ├── morphology.py                 # Exp 4: Morphological Analysis & Query Expansion
│   ├── ngram.py                      # Exp 5: N-Gram Language Model & Autocomplete
│   ├── pos_tagging.py                # Exp 6: Part-of-Speech Tagging
│   ├── chunking.py                   # Exp 7: Phrase Chunking & Training Size Evaluation
│   ├── ner.py                        # Exp 8: Named Entity Recognition
│   ├── similarity.py                 # Exp 9: Text Similarity & Restaurant Recommender
│   ├── wsd.py                        # Exp 10: Deep Learning WSD (Bi-LSTM / GRU)
│   └── pipeline.py                   # Unified Pipeline Orchestrator connecting all 10 stages
│
├── models/                           # Serialized Model Artifacts
│   ├── sentiment_model.pkl           # Trained TF-IDF + Logistic Regression Classifier
│   ├── wsd_model.pt                  # Trained PyTorch Bi-LSTM Neural Weights
│   └── wsd_meta.json                 # Vocabulary and label encoder mapping
│
└── outputs/                          # Generated Plots & Benchmark Records
    ├── confusion_matrix.png          # Sentiment Classifier Confusion Matrix
    ├── chunking_training_size_curve.png # Exp 7: Training Size vs Accuracy Curve
    ├── wsd_loss_curve.png            # Exp 10: Bi-LSTM Neural Cross-Entropy Training Loss
    └── model_benchmarks.json         # Complete empirical evaluation metrics
```

---

## ⚡ Quick Start & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/vishal-pal-12/Restaurant-NLP-Analysis-System.git
cd Restaurant-NLP-Analysis-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Option A: Launch Interactive Streamlit Web App
```bash
streamlit run app.py
```
*(Or double-click `run_app.bat` on Windows)*. The browser will open at `http://localhost:8501`.

### 4. Option B: Run Interactive Terminal Predictor (VS Code Console)
```bash
python predict.py
```
*(Or double-click `run_predict.bat`)*. You can type:
- Any Restaurant Name (e.g., `Tasty Bites`, `Little Italy`, `Bukhara`)
- Any Dish Name (e.g., `biryani`, `pizza`, `butter chicken`)
- Any Review (e.g., `The food was cold but service was very good`)
- Special commands: `list` (view 35 catalog restaurants), `random` (random dataset sample), `exit`.

### 5. Option C: Run Master Jupyter Notebook
```bash
jupyter notebook restaurant_nlp_mini_project.ipynb
```
*(Or double-click `open_jupyter.bat`)*. Contains all 10 experiments, visualizations, and an interactive real-time input cell.

### 6. Option D: Run Automated Test Suite
```bash
python test_pipeline.py
```
Runs 11 automated test cases verifying that each module and the unified pipeline execute without error:
```text
Ran 11 tests in 1.482s - OK
```

---

## 📈 Empirical Model Evaluation Results

| Experiment | Model Architecture | Metric | Result |
|---|---|---|---|
| **Exp 1: Sentiment Analysis** | TF-IDF + Logistic Regression | Accuracy / F1-Score | **85.71% / 85.87%** |
| **Exp 1 (Baseline)** | TF-IDF + Multinomial Naive Bayes | Accuracy / F1-Score | 78.57% / 78.00% |
| **Exp 7: Chunking (60% Split)** | Context Feature Classifier | Accuracy / F1-Score | 76.32% / 76.90% |
| **Exp 7: Chunking (70% Split)** | Context Feature Classifier | Accuracy / F1-Score | 82.14% / 82.70% |
| **Exp 7: Chunking (80% Split)** | Context Feature Classifier | Accuracy / F1-Score | **89.47% / 89.47%** |
| **Exp 10: Neural WSD** | PyTorch Bidirectional LSTM | Test Accuracy | **88.89%** |

---

## 📜 Academic Submission & College Viva Guide
A comprehensive 24-section formal project report following standard engineering curriculum standards is provided in **[`FINAL_REPORT.md`](FINAL_REPORT.md)**, covering Problem Definition, Mathematical Formulations, Architecture Flowcharts, Algorithms, Test Cases, and Viva Q&A.

---

## 👨‍💻 Author
- **Vishal Pal** ([@vishal-pal-12](https://github.com/vishal-pal-12))
- Project Domain: Natural Language Processing (NLP) & Computational Linguistics
- License: [MIT](LICENSE)
