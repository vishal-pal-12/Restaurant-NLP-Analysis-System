"""
Complete Restaurant NLP Pipeline

Connects all 10 NLP experiments into one unified execution workflow:
Input Review / Food Query
    -> 1. Sentiment Analysis
    -> 2. Tokenization, Filtration & Script Validation
    -> 3. Stop Word Removal, Stemming & Lemmatization
    -> 4. Morphological Analysis & Word Generation
    -> 5. N-Gram Language Modeling (<s>, </s>)
    -> 6. Part-of-Speech (POS) Tagging
    -> 7. Chunking & Feature / Training Size Benchmark
    -> 8. Named Entity Recognition (RESTAURANT, FOOD, LOCATION, etc.)
    -> 9. Text Similarity & Restaurant Matching
    -> 10. Word Sense Disambiguation (LSTM/GRU)
    -> Final Integrated NLP Analysis Result
"""

from typing import Dict, Any
from .sentiment import run_experiment_1
from .preprocessing import run_experiment_2, run_experiment_3
from .morphology import run_experiment_4
from .ngram import run_experiment_5
from .pos_tagging import run_experiment_6
from .chunking import run_experiment_7
from .ner import run_experiment_8
from .similarity import run_experiment_9
from .wsd import run_experiment_10


def run_complete_nlp_pipeline(input_text: str) -> Dict[str, Any]:
    """
    Executes all 10 NLP modules sequentially on the input text.
    Passes processed data from earlier modules into downstream stages.
    """
    if not input_text or not input_text.strip():
        input_text = "The restaurant serves delicious pizza and the service is excellent."

    # Stage 1: Sentiment Analysis (Experiment 1)
    sentiment_result = run_experiment_1(input_text)

    # Stage 2: Tokenization, Filtration & Script Validation (Experiment 2)
    exp2_result = run_experiment_2(input_text)

    # Stage 3: Stop Words, Stemming & Lemmatization (Experiment 3)
    exp3_result = run_experiment_3(exp2_result["filtered_tokens"])

    # Stage 4: Morphological Analysis & Word Generation (Experiment 4)
    exp4_result = run_experiment_4(exp3_result["tokens_without_stopwords"])

    # Stage 5: N-Gram Language Model with Boundary Markers (Experiment 5)
    exp5_result = run_experiment_5(input_text)

    # Stage 6: POS Tagging (Experiment 6)
    exp6_result = run_experiment_6(exp2_result["filtered_tokens"])

    # Stage 7: Chunking with POS tags + Training Size Benchmark (Experiment 7)
    exp7_result = run_experiment_7(exp6_result["tagged_tokens"])

    # Stage 8: Named Entity Recognition (Experiment 8)
    exp8_result = run_experiment_8(input_text)

    # Stage 9: Text Similarity & Restaurant Matching (Experiment 9)
    # Uses expanded query tokens from Morphology (Exp 4) if applicable
    search_query = exp4_result["query_expansion"]["expanded_query_string"]
    if not search_query:
        search_query = input_text
    exp9_result = run_experiment_9(input_text, top_k=5)

    # Stage 10: Word Sense Disambiguation (Experiment 10)
    # Detects if target ambiguous words exist in sentence, defaults to 'course'
    target_word = "course"
    lower_in = input_text.lower()
    if "course" in lower_in:
        target_word = "course"
    elif "bill" in lower_in:
        target_word = "bill"
    elif "order" in lower_in:
        target_word = "order"

    exp10_result = run_experiment_10(input_text, target_word=target_word)

    # Final Synthesized Summary
    final_summary = {
        "input_text": input_text,
        "sentiment": sentiment_result["predicted_sentiment"],
        "sentiment_confidence": f"{round(sentiment_result['confidence'] * 100, 1)}%",
        "script_valid": exp2_result["script_validation"]["is_valid"],
        "token_count": exp2_result["total_words"],
        "filtered_count": exp2_result["total_filtered"],
        "extracted_chunks_count": exp7_result["total_chunks"],
        "named_entities_found": [f"{e['entity']} ({e['type']})" for e in exp8_result["entities"]],
        "top_restaurant_recommendation": exp9_result["top_recommended_restaurant"],
        "similarity_score": f"{round(exp9_result['highest_similarity_score'] * 100, 1)}%",
        "disambiguated_sense": f"{target_word.upper()} -> {exp10_result['predicted_sense']} ({exp10_result['confidence']}%)"
    }

    return {
        "input_text": input_text,
        "summary": final_summary,
        "experiment_1_sentiment": sentiment_result,
        "experiment_2_tokenization": exp2_result,
        "experiment_3_preprocessing": exp3_result,
        "experiment_4_morphology": exp4_result,
        "experiment_5_ngram": exp5_result,
        "experiment_6_pos": exp6_result,
        "experiment_7_chunking": exp7_result,
        "experiment_8_ner": exp8_result,
        "experiment_9_similarity": exp9_result,
        "experiment_10_wsd": exp10_result
    }
