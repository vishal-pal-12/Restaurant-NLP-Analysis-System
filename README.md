# Restaurant Review & Food Query Analysis System Using Natural Language Processing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.10-green.svg)](https://www.nltk.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, production-grade **College NLP Mini Project** integrating **all 10 university NLP syllabus practicals** into a unified, real-world Restaurant Review and Food Query Analysis platform.

Available across 4 primary interfaces:
1. 🌐 **Modern Interactive Streamlit Web Application** (`app.py`)
2. 💻 **Real-Time Interactive VS Code Terminal Console** (`predict.py`)
3. 📓 **Master 30-Cell End-to-End Jupyter Notebook with Live Prediction Console** (`restaurant_nlp_mini_project.ipynb`)
4. 🧪 **Automated Unit & Integration Test Suite** (`test_pipeline.py` — 11/11 tests passing)

---

## 📌 Problem Statement & Overview

Restaurants and food aggregator platforms (such as Zomato, Swiggy, and Google Reviews) receive thousands of customer reviews and food search queries daily. Manually parsing and extracting actionable insights from this unstructured text is infeasible, and traditional keyword search fails due to:
1. **Unstructured Mixed Sentiment:** Reviews often have positive and negative aspects intertwined (*"The pasta was delicious but the service was terrible"*).
2. **Linguistic Variations & Inflections:** Users query variations (*serve, serving, cooked*) or multi-word culinary concepts (*"wood-fired thin-crust pizza"*).
3. **Contextual Polysemy (Ambiguity):** Words have multiple meanings based on context (e.g., **"course"** as a restaurant meal course vs. an educational degree course).

This project integrates 10 core Natural Language Processing techniques into a single pipeline to automate sentiment classification, entity extraction, phrase chunking, semantic restaurant recommendation, and neural context disambiguation.

---

## 🔬 The 10 Integrated NLP Phases (Syllabus-Accurate)

| Phase | Module | NLP Technique / Model | Real-World Problem Solved |
|---|---|---|---|
| **Phase-1** | `modules/sentiment.py` | TF-IDF + Logistic Regression / Naive Bayes | **Study applications of NLP & Sentiment Analysis:** Classifies reviews as Positive, Negative, or Neutral with confidence polarity (85.71% test accuracy). |
| **Phase-2** | `modules/preprocessing.py` | NLTK Tokenizer + Unicode Script Validator | **Tokenization, Filtration & Script Validation:** Verifies Latin/English script characters, isolates clean sentence and word tokens, and filters punctuation noise. |
| **Phase-3** | `modules/preprocessing.py` | Porter Stemmer + WordNet Lemmatizer | **Stop Word Removal, Stemming & Lemmatization:** Normalizes tokens to canonical base dictionary forms while strictly preserving negation words (*not, no, never*). |
| **Phase-4** | `modules/morphology.py` | Root/Affix Stripping & Rule Generator | **Morphological Analysis & Generation:** Generates inflectional and derivational variants (*cook -> cooks, cooked, cooking, cooker*) to expand search queries. |
| **Phase-5** | `modules/ngram.py` | MLE Probabilities with `<s>` and `</s>` Boundaries | **N-Gram Language Modeling:** Computes Unigram, Bigram, and Trigram probabilities and delivers next-word autocomplete search recommendations. |
| **Phase-6** | `modules/pos_tagging.py` | Penn Treebank POS Tagger | **Part-of-Speech Tagging:** Categorizes syntactic components (Nouns, Verbs, Adjectives, Adverbs) with human-readable explanations. |
| **Phase-7** | `modules/chunking.py` | Grammar Chunking & Feature Selection Benchmark | **Chunking & Feature Selection:** Extracts multi-word food noun phrases (*"spicy Indian curry"*), and benchmarks **60%, 70%, and 80% training size splits**. |
| **Phase-8** | `modules/ner.py` | Domain Gazetteer + Maximum Entropy NER | **Named Entity Recognition:** Extracts structured entities: `RESTAURANT`, `FOOD`, `LOCATION`, `DATE`, `PERSON`, `ORGANIZATION`. |
| **Phase-9** | `modules/similarity.py` | TF-IDF Vector Space + Cosine Similarity | **Text Similarity & Recommendation:** Matches user food queries against 35 catalog restaurant profiles and ranks top matching venues. |
| **Phase-10** | `modules/wsd.py` | Bidirectional LSTM/GRU Neural Network | **Word Sense Disambiguation (Deep Learning):** Resolves polysemous ambiguous culinary words (*course, bill, order*) using neural context understanding (88.89% test accuracy). |

---

## 🌟 Modern Streamlit Web UI (`app.py`)

The Streamlit web application features an attractive, high-performance UI modeled after modern catalog platforms:
- **Typography & Aesthetics:** Styled with Google **Inter font**, custom CSS variables, and **Linear Gradient Hero Banners** (`#0F172A` to `#2563EB`).
- **Target Showcase Card (`.target-restaurant-hero`):** Prominently displays the selected venue's Cuisine, Rating, Price Range, Location, and Popular Dishes with an emerald border accent.
- **Dynamic Recommendation Cards (`.restaurant-card`):** Hover lift animations, drop-shadow elevation, and prominent green match percentage pills (`.match-tag`, e.g., `94.8% Match`).
- **Linguistic Phase Badges (`.phase-badge`):** Color-coded pills identifying each of the 10 university syllabus phases.

### 5 Dedicated Navigation Modes:
1. 🍽️ **Restaurant Predictor & 10-Phase NLP Analysis**:
   - Interactive dropdown covering all **35 catalog restaurants**.
   - Live search input by name, dish, ID, or keyword.
   - Hero showcase card and top recommended alternatives with match percentages.
   - **9 deep linguistic analysis tabs** covering every phase sequentially.
2. 🔍 **Custom Query / Review Search Pipeline**:
   - Quick-select prompt chips (*"authentic Italian pizza"*, *"spicy hyderabadi biryani"*, etc.).
   - Custom free-form query and review input with real-time cosine recommendation and 10-phase analysis.
3. 🔬 **NLP 10-Phase Interactive Laboratory**:
   - 10 standalone diagnostic workbenches allowing individual hands-on testing of each NLP phase.
4. 📊 **Restaurant Catalog & Datasets**:
   - Full tabular browser for the 35 restaurant profiles, 70 labeled customer reviews (with sentiment distribution charts), and 60 WSD training samples.
5. 📑 **Formal Project Report (24 Sections)**:
   - Comprehensive academic report rendered in-browser with one-click Markdown file download.

---

## 💻 Interactive VS Code Prediction Console (`predict.py`)

A fast, interactive terminal predictor designed for live evaluations and terminal demonstrations in VS Code:
- Enter any **Restaurant Name** (e.g., `Tasty Bites`, `Little Italy`, `Bukhara`).
- Enter any **Food / Dish** (e.g., `biryani`, `pizza`, `butter chicken`).
- Enter any **Customer Review** (e.g., `The food was cold but service was great`).
- Built-in commands:
  - `list` → View all 35 catalog restaurants with cuisines and locations.
  - `random` → Pick a random review or restaurant from the dataset.
  - `exit` → Exit the console.
- Prints structured output labeled `[Phase-1]` through `[Phase-10]` in real-time.

---

## 📓 Master Jupyter Notebook (`restaurant_nlp_mini_project.ipynb`)

- **30 Structured Cells:** Perfectly organized from Phase-1 to Phase-10.
- **Syllabus-Strict Phase Separation:**
  - `Phase-2`: Tokenization, Filtration & Script Validation.
  - `Phase-3`: Stop Word Removal, Stemming & Lemmatization.
  - `Phase-6`: POS Tagging & Syntactic Analysis.
  - `Phase-7`: Chunking, Feature Selection & Training Size Benchmark (60%, 70%, 80%).
- **Direct Live Input Console Cell:** Located right after Phase-10. Simply enter any restaurant name or dish to run and display the complete 10-phase NLP breakdown interactively.

---

## 📁 Repository Structure

```text
Restaurant-NLP-Analysis-System/
│
├── app.py                            # Streamlit Web Application (5 Modern Modes)
├── predict.py                        # Interactive VS Code Terminal Prediction Console
├── train_models.py                   # Standalone trainer for ML & Deep Learning models
├── test_pipeline.py                  # Automated unit and integration test suite (11/11 passing)
├── restaurant_nlp_mini_project.ipynb # Master 30-cell End-to-End Jupyter Notebook
│
├── run_app.bat                       # One-click Windows launcher for Streamlit App
├── run_predict.bat                   # One-click Windows launcher for Terminal Console
├── open_jupyter.bat                  # One-click Windows launcher for Jupyter Notebook
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Comprehensive Project Documentation
├── FINAL_REPORT.md                   # 24-Section formal college submission report
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
│   ├── sentiment.py                  # Phase-1: Sentiment Analysis & NLP Applications
│   ├── preprocessing.py              # Phase-2 & Phase-3: Preprocessing, Validation & Normalization
│   ├── morphology.py                 # Phase-4: Morphological Analysis & Query Expansion
│   ├── ngram.py                      # Phase-5: N-Gram Language Model & Autocomplete
│   ├── pos_tagging.py                # Phase-6: Part-of-Speech Tagging
│   ├── chunking.py                   # Phase-7: Phrase Chunking & Training Size Evaluation
│   ├── ner.py                        # Phase-8: Named Entity Recognition
│   ├── similarity.py                 # Phase-9: Text Similarity & Restaurant Recommender
│   ├── wsd.py                        # Phase-10: Deep Learning WSD (Bi-LSTM / GRU)
│   └── pipeline.py                   # Unified Pipeline Orchestrator connecting all 10 stages
│
├── models/                           # Serialized Model Artifacts
│   ├── sentiment_model.pkl           # Trained TF-IDF + Logistic Regression Classifier
│   ├── wsd_model.pt                  # Trained PyTorch Bi-LSTM Neural Weights
│   └── wsd_meta.json                 # Vocabulary and label encoder mapping
│
└── outputs/                          # Generated Plots & Benchmark Records
    ├── confusion_matrix.png          # Sentiment Classifier Confusion Matrix
    ├── chunking_training_size_curve.png # Phase-7: Training Size vs Accuracy Curve
    ├── wsd_loss_curve.png            # Phase-10: Bi-LSTM Neural Training Loss Curve
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

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```
*(Or double-click `run_app.bat` on Windows)*. Opens automatically in your browser at `http://localhost:8501`.

### 4. Run VS Code Terminal Predictor
```bash
python predict.py
```
*(Or double-click `run_predict.bat`)*. Enter any restaurant name (e.g. `Tasty Bites`), dish (`biryani`), or customer review.

### 5. Run Master Jupyter Notebook
```bash
jupyter notebook restaurant_nlp_mini_project.ipynb
```
*(Or double-click `open_jupyter.bat`)*. Execute all 30 cells sequentially or test the interactive console after Phase-10.

### 6. Run Automated Test Suite
```bash
python test_pipeline.py
```
Verifies all 10 modules and the unified pipeline:
```text
Ran 11 tests in 1.482s - OK
```

---

## 📈 Empirical Model Evaluation Results

| Phase / Module | Model Architecture | Metric | Result |
|---|---|---|---|
| **Phase-1: Sentiment Analysis** | TF-IDF + Logistic Regression | Accuracy / F1-Score | **85.71% / 85.87%** |
| **Phase-1 (Baseline)** | TF-IDF + Multinomial Naive Bayes | Accuracy / F1-Score | 78.57% / 78.00% |
| **Phase-2: Script Validation** | Unicode Character Analyzer | Latin Character Accuracy | **100%** |
| **Phase-7: Chunking (60% Split)** | Context Feature Classifier | Accuracy / F1-Score | 76.32% / 76.90% |
| **Phase-7: Chunking (70% Split)** | Context Feature Classifier | Accuracy / F1-Score | 82.14% / 82.70% |
| **Phase-7: Chunking (80% Split)** | Context Feature Classifier | Accuracy / F1-Score | **89.47% / 89.47%** |
| **Phase-8: NER** | Gazetteer + MaxEnt | Entity Precision | **92.30%** |
| **Phase-10: Neural WSD** | PyTorch Bidirectional LSTM | Test Accuracy | **88.89%** |

---

## 📜 Academic Submission & College Viva Guide

A comprehensive 24-section formal project report following standard engineering university curriculum guidelines is provided in **[`FINAL_REPORT.md`](FINAL_REPORT.md)**, covering Problem Definition, Mathematical Formulations, Architecture Flowcharts, Algorithms, Test Cases, and Viva Q&A.

---

## 👨‍💻 Author
- **Vishal Pal** ([@vishal-pal-12](https://github.com/vishal-pal-12))
- Domain: Natural Language Processing (NLP) & Computational Linguistics
- License: [MIT](LICENSE)
