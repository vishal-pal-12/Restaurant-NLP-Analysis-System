# College NLP Mini Project Report

## PROJECT TITLE:
**Restaurant Review & Food Query Analysis System Using Natural Language Processing**

**Domain:** Natural Language Processing, Sentiment Analysis, Information Extraction, Deep Learning  
**Project Type:** College NLP Mini Project / Capstone Submission  
**Academic Year:** 2025–2026  

---

## TABLE OF CONTENTS

1. Introduction
2. Problem Statement
3. Objectives
4. Scope
5. Software Requirements
6. System Architecture
7. Dataset
8. Experiment 1 — Sentiment Analysis
9. Experiment 2 — Tokenization, Filtration & Script Validation
10. Experiment 3 — Stop Word Removal, Stemming & Lemmatization
11. Experiment 4 — Morphological Analysis & Word Generation
12. Experiment 5 — N-Gram Language Model
13. Experiment 6 — Part-of-Speech (POS) Tagging
14. Experiment 7 — Chunking, Feature Selection & Training Size Analysis
15. Experiment 8 — Named Entity Recognition (NER)
16. Experiment 9 — Text Similarity & Restaurant Matching
17. Experiment 10 — Word Sense Disambiguation using LSTM/GRU
18. Results & Evaluation Metrics
19. Screenshots & Visual Interface
20. Advantages
21. Limitations
22. Applications
23. Future Scope
24. Conclusion

---

## 1. INTRODUCTION

The rapid digitization of the hospitality industry has resulted in an exponential influx of customer-generated content on food delivery apps (e.g., Zomato, Swiggy) and review portals (e.g., Google Maps, TripAdvisor). Diners write detailed opinions regarding food quality, ambiance, delivery punctuality, pricing, and staff hospitality. Concurrently, consumers frequently submit complex natural language queries such as *"Find authentic Italian restaurants in Pune serving wood-fired pizza"* or *"Spicy chicken biryani with fast delivery"*.

Traditional keyword search engines fail to comprehend linguistic nuances such as morphological inflections, syntactic chunking, polysemous word senses, and emotional polarity. This mini project presents the **Restaurant Review & Food Query Analysis System**, an integrated end-to-end NLP system that unifies 10 core Natural Language Processing practical experiments into a single cohesive pipeline and interactive web interface.

---

## 2. PROBLEM STATEMENT

Restaurants receive thousands of unstructured customer reviews and food search queries daily. Manually parsing and extracting actionable insights from this text is computationally prohibitive and labor-intensive. Key challenges include:

1. **Sentiment Ambiguity:** Determining whether a customer had a satisfactory, neutral, or negative dining experience from informal phrasing.
2. **Text Noise & Script Contamination:** Raw reviews contain typos, excessive punctuation, and unsupported foreign scripts.
3. **Lexical Redundancy:** High frequency of functional stopwords obscuring domain keywords.
4. **Morphological Variation:** Words appear in multiple inflections (*serve, serves, served, serving, server*), causing missed query matches.
5. **Contextual Co-occurrence:** Inability to predict common food collocations without probabilistic n-gram models.
6. **Syntactic Ambiguity:** Extracting key food concepts (e.g., *"spicy Indian food"*) requires syntactic parsing and POS tagging.
7. **Entity Recognition:** Isolating restaurant names, dishes, locations, and dining dates from generic vocabulary.
8. **Semantic Matching:** Recommending the best restaurant matching a user's natural query without strict keyword matching.
9. **Polysemy:** Words having multiple meanings based on context (e.g., *"course"* as a meal course vs. educational course).

---

## 3. OBJECTIVES

The primary objective of this project is to construct a production-ready, modular NLP application that:
- Connects all 10 university NLP laboratory experiments into a single uninterrupted processing pipeline.
- Automatically validates scripts and sanitizes restaurant reviews and queries.
- Deconstructs text into morphological roots and expands search terms dynamically.
- Models culinary phrases using Unigram, Bigram, and Trigram language models.
- Performs POS tagging and extracts multi-word food noun phrases through chunking.
- Analyzes the effect of linguistic feature selection and dataset training sizes (60%, 70%, 80%) on chunking accuracy.
- Identifies domain-specific named entities (`RESTAURANT`, `FOOD`, `LOCATION`, `PERSON`, `DATE`, `ORGANIZATION`).
- Calculates cosine similarity to match user queries with restaurant catalogs.
- Resolves word polysemy using deep recurrent neural networks (Bi-directional LSTM / GRU).
- Provides an intuitive, modern Streamlit graphical user interface.

---

## 4. SCOPE

- **Target Domain:** Restaurant customer reviews, menus, food delivery queries, and dining feedback.
- **Languages Supported:** English (with automated script validation to filter non-Latin text).
- **Target Users:** Restaurant owners, food analysts, college evaluators, and end diners searching for culinary recommendations.
- **Deployment Platform:** Lightweight local or cloud Streamlit web application running with CPU-optimized PyTorch and Scikit-learn backends.

---

## 5. SOFTWARE REQUIREMENTS

### Software & Language:
- **Operating System:** Windows 10/11, Linux, or macOS
- **Programming Language:** Python 3.10+ / 3.12 / 3.14
- **Web UI Framework:** Streamlit 1.55+
- **Deep Learning Framework:** PyTorch 2.14+ (CPU / CUDA compatible)
- **NLP & ML Libraries:**
  - NLTK 3.10+ (Punkt, Stopwords, WordNet, Averaged Perceptron, MaxEnt Chunkers)
  - Scikit-Learn 1.8+ (TF-IDF, Logistic Regression, MultinomialNB, Cosine Similarity)
  - Pandas 2.2+ & NumPy 2.3+
  - Matplotlib 3.10+ & Seaborn 0.13+

---

## 6. SYSTEM ARCHITECTURE

```
+--------------------------------------------------------------------------+
|                  USER INPUT REVIEW / FOOD QUERY                          |
|         e.g., "Tasty Bites in Pune serves delicious pizza..."            |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| 1. SENTIMENT ANALYSIS MODULE (Exp 1)                                     |
|    TF-IDF Vectorizer -> Logistic Regression / Naive Bayes                |
|    Output: Positive / Negative / Neutral + Confidence Polarity           |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| 2. TOKENIZATION, FILTRATION & SCRIPT VALIDATION MODULE (Exp 2)           |
|    Unicode Script Validator -> Sentence & Word Tokenizer -> Noise Filter |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| 3. STOP WORD REMOVAL, STEMMING & LEMMATIZATION (Exp 3)                   |
|    Preserve Negations -> Porter Stemmer -> WordNet POS-Aware Lemma       |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| 4. MORPHOLOGICAL ANALYSIS & WORD GENERATION (Exp 4)                      |
|    Root / Affix Stripping -> Inflection Generator -> Query Expansion     |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| 5. N-GRAM LANGUAGE MODEL (Exp 5)                                         |
|    <s> Boundary Markers </s> -> Unigrams, Bigrams, Trigrams -> MLE Prob   |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| 6. PART-OF-SPEECH (POS) TAGGING (Exp 6)                                  |
|    NLTK Penn Treebank Tagging -> Tag Explanations -> Syntactic Mapping   |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| 7. CHUNKING & TRAINING SIZE / FEATURE SELECTION (Exp 7)                  |
|    Regex Food Phrase Chunker -> 60% / 70% / 80% Training Size Benchmark   |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| 8. NAMED ENTITY RECOGNITION (NER) (Exp 8)                                |
|    Gazetteer + MaxEnt -> RESTAURANT, FOOD, LOCATION, DATE, PERSON        |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| 9. TEXT SIMILARITY & RESTAURANT RECOMMENDATION (Exp 9)                   |
|    TF-IDF Vector Space -> Cosine Similarity Matching vs 35+ Restaurants  |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
| 10. WORD SENSE DISAMBIGUATION (Exp 10)                                   |
|    Deep Learning: Embedding -> Bi-LSTM / GRU -> Dense -> FOOD vs EDU    |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                     FINAL SYNTHESIZED DASHBOARD                          |
+--------------------------------------------------------------------------+
```

---

## 7. DATASETS

The system operates autonomously using 3 locally curated domain datasets:

1. **`data/restaurant_reviews.csv` (70 records):**
   - Columns: `review`, `sentiment`
   - Balanced distribution of positive, negative, and neutral real-world restaurant reviews covering diverse foods, delivery speeds, hygiene, and ambiance.
2. **`data/restaurant_data.csv` (35 records):**
   - Columns: `restaurant_name`, `location`, `cuisine`, `rating`, `popular_dishes`, `description`
   - Detailed restaurant catalog across Pune, Mumbai, Delhi, and Bangalore.
3. **`data/wsd_dataset.csv` (60 records):**
   - Columns: `sentence`, `target_word`, `sense`
   - Labelled contextual sentences for ambiguous words (*course, bill, order*).

---

## 8. EXPERIMENT 1 — SENTIMENT ANALYSIS

- **Methodology:** Text is vectorized using TF-IDF with bi-gram range `(1, 2)` and sublinear term frequency scaling. A regularized Logistic Regression classifier and Multinomial Naive Bayes are trained using an 80/20 train/test split.
- **Mathematical Formulation:**
  $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
- **Evaluation:** Confusion matrix, accuracy, precision, recall, and F1-score are computed. Real-time predictions generate class confidence distributions.

---

## 9. EXPERIMENT 2 — TOKENIZATION, FILTRATION & SCRIPT VALIDATION

- **Script Validation:** Employs Unicode character database analysis (`unicodedata.name`) to ensure input text is standard Latin/English. Rejects foreign non-Latin scripts (Devanagari, Cyrillic, East Asian) or pure symbol junk.
- **Tokenization:** Sentence tokenization via NLTK `sent_tokenize` and word tokenization via `word_tokenize`.
- **Filtration:** Strips non-alphanumeric punctuation marks (`!`, `,`, `.`, `?`, `"`, `'`).

---

## 10. EXPERIMENT 3 — STOP WORD REMOVAL, STEMMING & LEMMATIZATION

- **Stop Word Removal:** Filters English stopwords while carefully preserving negation terms (*not, no, never*) to preserve sentiment semantics.
- **Stemming:** Applies the Porter Stemmer rule-based suffix stripping algorithm.
- **Lemmatization:** Uses the WordNet Lemmatizer with dynamic POS mapping (Noun, Verb, Adjective, Adverb) to return canonical dictionary base forms (*foods* $\rightarrow$ *food*, *serving* $\rightarrow$ *serve*).

---

## 11. EXPERIMENT 4 — MORPHOLOGICAL ANALYSIS & WORD GENERATION

- **Decomposition:** Strips known inflectional prefixes (*un-, re-, pre-*) and suffixes (*-ing, -ed, -s, -er, -tion*).
- **Word Generation:** Produces all valid morphological inflections and derivations for food terms:
  - *serve* $\rightarrow$ *serves, served, serving, server, service, servant*
  - *cook* $\rightarrow$ *cooks, cooked, cooking, cooker, cookery*
  - *recommend* $\rightarrow$ *recommends, recommended, recommendation*
- **Search Expansion:** Automatically expands query terms with generated variants to boost restaurant search recall.

---

## 12. EXPERIMENT 5 — N-GRAM LANGUAGE MODEL

- **Sentence Boundaries:** Appends start marker `<s>` and termination marker `</s>` to sentences.
- **Modeling:** Computes Unigram, Bigram, and Trigram frequencies using `collections.Counter`.
- **Maximum Likelihood Estimation (MLE):**
  $$P(w_i | w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}$$
  $$P(w_i | w_{i-2}, w_{i-1}) = \frac{\text{Count}(w_{i-2}, w_{i-1}, w_i)}{\text{Count}(w_{i-2}, w_{i-1})}$$
- **Autocomplete:** Suggests top 3 candidate next-words using Trigram-to-Bigram backoff.

---

## 13. EXPERIMENT 6 — PART-OF-SPEECH (POS) TAGGING

- **Tagger:** NLTK Penn Treebank POS tagger.
- **Mapping:** Translates cryptic tags into plain English:
  - `JJ` $\rightarrow$ Adjective (*delicious, spicy, crispy*)
  - `NN / NNS` $\rightarrow$ Noun (*pizza, restaurant, service*)
  - `VB / VBZ` $\rightarrow$ Verb (*serves, cooked, dine*)
  - `RB` $\rightarrow$ Adverb (*very, extremely, quickly*)
- **Pipeline Integration:** Passes tagged token tuples directly to Chunking and Lemmatization modules.

---

## 14. EXPERIMENT 7 — CHUNKING, FEATURE SELECTION & TRAINING SIZE ANALYSIS

- **Grammar-based Chunker:** Uses regex grammar patterns:
  - `FOOD_NP`: `<DT>?<RB.*>*<JJ.*>*<NN.*>+` (e.g., *"the very spicy Indian food"*, *"delicious pizza"*)
  - `SERVICE_CHUNK`: `<JJ.*>+<NN.*>*(?:service|staff|waiter)`
- **Syllabus Benchmark Experiment:**
  - Evaluates how feature selection and training data size affect chunking performance:
    - **Features Tested:** Word Only vs. Word + POS vs. Full Features (Word + POS + Context + Length).
    - **Training Data Splits:** 60%, 70%, 80%.
  - **Results:** Accuracy increases monotonically from 76.32% (at 60% split) to 82.14% (at 70% split) and 89.47% (at 80% split). Incorporating POS and surrounding word context boosts chunk boundary detection significantly over single-word features.

---

## 15. EXPERIMENT 8 — NAMED ENTITY RECOGNITION (NER)

- **Target Entity Types:**
  - `RESTAURANT`: *Tasty Bites, Cafe Goodluck, Barbeque Nation*
  - `FOOD`: *pizza, biryani, pasta, burger, dosa, coffee*
  - `LOCATION`: *Pune, Mumbai, Delhi, FC Road, Bandra*
  - `PERSON`: *Chef Sanjeev, waiter Ramesh*
  - `DATE`: *Sunday, yesterday, tonight, 25th December*
  - `ORGANIZATION`: *Zomato, Swiggy, McDonald's*
- **Architecture:** Hybrid gazetteer matching + NLTK MaxEnt chunker with interactive color-coded HTML badges in the web UI.

---

## 16. EXPERIMENT 9 — TEXT SIMILARITY & RESTAURANT MATCHING

- **Mechanism:** Builds TF-IDF representation of restaurant names, cuisines, popular dishes, and descriptions.
- **Cosine Metric:**
  $$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$
- **Recommendation:** Ranks catalog items in descending order of similarity, returning top matching restaurants with ratings, location, and match percentages.

---

## 17. EXPERIMENT 10 — WORD SENSE DISAMBIGUATION USING LSTM/GRU

- **Disambiguation Task:** Resolves polysemous culinary words in context:
  - Target word: *"course"*
    - **Sense 1: `FOOD`** (*"The main course was delicious and hot."*)
    - **Sense 2: `EDUCATION`** (*"I enrolled in an online machine learning course."*)
- **Deep Learning Pipeline:**
  1. Vocabulary encoding with `<PAD>` and `<UNK>` tokens (padding to sequence length 22).
  2. Dense Word Embedding layer (dim=48).
  3. Bidirectional LSTM / GRU layer (hidden units=64).
  4. Dropout layer (rate=0.3) for regularization.
  5. Fully-connected Linear layer + Softmax activation.
- **Results:** Achieves 100% training accuracy and 88.89% test accuracy, learning context words (*main, meal, delicious* vs. *enrolled, university, lecture*) with high confidence (>99%).

---

## 18. RESULTS & EVALUATION METRICS

| NLP Module | Model / Technique | Key Metric | Score |
|---|---|---|---|
| **Exp 1: Sentiment Analysis** | TF-IDF + Logistic Regression | Accuracy / F1-Score | **85.71% / 85.87%** |
| **Exp 1 Baseline** | TF-IDF + Multinomial Naive Bayes | Accuracy | **78.57%** |
| **Exp 2: Script Validation** | Unicode Character Analysis | Latin Accuracy | **100%** |
| **Exp 5: N-Gram Model** | Unigram/Bigram/Trigram MLE | Perplexity / Coverage | **Optimal (<s>, </s>)** |
| **Exp 7: Chunking (60% Train)** | Logistic Regression + Full Feats | Accuracy / F1 | **76.32% / 76.90%** |
| **Exp 7: Chunking (70% Train)** | Logistic Regression + Full Feats | Accuracy / F1 | **82.14% / 82.70%** |
| **Exp 7: Chunking (80% Train)** | Logistic Regression + Full Feats | Accuracy / F1 | **89.47% / 89.47%** |
| **Exp 8: NER** | Hybrid Gazetteer + MaxEnt | Entity Precision | **92.3%** |
| **Exp 9: Text Similarity** | TF-IDF + Cosine Similarity | Top-1 Retrieval Match | **High Relevance** |
| **Exp 10: WSD Model** | Bidirectional PyTorch LSTM | Test Accuracy | **88.89%** |

---

## 19. SCREENSHOTS & VISUAL INTERFACE

The web dashboard is organized into 4 distinct views:
1. **🚀 Complete NLP Pipeline:** Single-click end-to-end execution showing all 10 modules in interactive cards.
2. **🧪 Individual Experiments (1–10):** Diagnostic tabs allowing custom testing of each module.
3. **📊 Datasets & Benchmark Metrics:** Data viewer, class distribution bar charts, and training loss plots.
4. **📑 College Project Report:** Full built-in 24-section report with one-click `.md` download.

*(Saved plots generated during benchmark execution: `outputs/confusion_matrix.png`, `outputs/chunking_training_size_curve.png`, `outputs/wsd_loss_curve.png`)*

---

## 20. ADVANTAGES

1. **Integrated Real-World Pipeline:** Rather than disjoint code snippets, all 10 experiments function as interconnected modules.
2. **Autonomous Offline Execution:** Includes pre-packaged datasets and saved model weights requiring zero external cloud API keys.
3. **Deep Learning Integration:** Employs PyTorch Bi-LSTM/GRU neural networks for semantic context disambiguation.
4. **High Academic Rigor:** Conforms directly to syllabus requirements, including the feature selection and training size comparison for chunking.
5. **Interactive UI:** Provides a polished Streamlit dashboard accessible to non-technical evaluators.

---

## 21. LIMITATIONS

1. **Domain Focus:** Optimized primarily for food, restaurant, and dining queries; generalized out-of-domain text may reduce NER coverage.
2. **Language Scope:** Currently processes English text; non-Latin foreign scripts are flagged by the script validation module.
3. **Dataset Scale:** Uses lightweight local datasets (70 reviews, 35 restaurants) optimized for rapid training and college demonstration.

---

## 22. APPLICATIONS

- **Restaurant Reputation Management:** Automated sentiment and food complaint analysis for restaurant managers.
- **Smart Food Search Engines:** Semantic query recommendation for food delivery portals (Zomato/Swiggy).
- **Customer Feedback Mining:** Extracting specific food item chunks to identify menu weaknesses (*cold pizza*, *overcooked pasta*).
- **Academic Demonstration:** Comprehensive teaching tool for computer science and NLP laboratory viva exams.

---

## 23. FUTURE SCOPE

- **Multilingual Support:** Extending script validation and tokenization to Indic languages (Hindi, Marathi) using IndicBERT.
- **Aspect-Based Sentiment Analysis (ABSA):** Linking sentiment directly to extracted chunks (e.g., *Service: Positive, Food: Negative*).
- **Transformer Integration:** Fine-tuning lightweight RoBERTa or DistilBERT for zero-shot WSD and sentiment classification.
- **Live Scraping API:** Integrating Google Places API and Swiggy web scrapers for real-time review ingestion.

---

## 24. CONCLUSION

The **Restaurant Review & Food Query Analysis System** successfully demonstrates how traditional rule-based linguistics, statistical natural language processing, machine learning classifiers, and deep recurrent neural networks can be harmoniously integrated into a production-grade NLP application. By uniting all 10 university NLP experiments into a continuous pipeline, the project bridges theoretical concepts with practical software engineering, providing an invaluable tool for restaurant analytics and modern computational linguistics.
