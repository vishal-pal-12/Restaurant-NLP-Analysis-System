"""
Restaurant Review & Food Query Analysis System Using Natural Language Processing
Interactive Streamlit Web Dashboard integrating all 10 College NLP Phases.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Modular NLP Components
from modules.sentiment import run_experiment_1, get_sentiment_analyzer
from modules.preprocessing import run_experiment_2, run_experiment_3, validate_script
from modules.morphology import run_experiment_4, get_morphological_analyzer
from modules.ngram import run_experiment_5, get_ngram_model
from modules.pos_tagging import run_experiment_6, PENN_TREEBANK_TAG_DICT
from modules.chunking import run_experiment_7, run_chunking_training_experiment
from modules.ner import run_experiment_8, ENTITY_COLORS
from modules.similarity import run_experiment_9, get_similarity_engine
from modules.wsd import run_experiment_10, get_wsd_disambiguator
from modules.pipeline import run_complete_nlp_pipeline

# Page Configuration
st.set_page_config(
    page_title="Restaurant NLP Intelligence System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Modern, Professional Academic CSS
st.markdown("""
<style>
    .main-title-container {
        padding: 1.2rem 1.8rem;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
    }
    .metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .summary-card-pos {
        background: #F0FDF4;
        border-left: 4px solid #16A34A;
        border-radius: 8px;
        padding: 14px 18px;
        color: #14532D;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .summary-card-neg {
        background: #FEF2F2;
        border-left: 4px solid #DC2626;
        border-radius: 8px;
        padding: 14px 18px;
        color: #7F1D1D;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .summary-card-neu {
        background: #F8FAFC;
        border-left: 4px solid #64748B;
        border-radius: 8px;
        padding: 14px 18px;
        color: #334155;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .recommend-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        transition: transform 0.1s ease;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .badge-food { background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .badge-rest { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
    .badge-loc  { background-color: #DBEAFE; color: #1E40AF; border: 1px solid #BFDBFE; }
    .badge-chunk{ background-color: #F3E8FF; color: #6B21A8; border: 1px solid #E9D5FF; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Top Presentation Header
st.markdown("""
<div class="main-title-container">
    <div class="main-title">🍽️ Restaurant Review & Food Query Analysis System</div>
    <div class="main-subtitle">Natural Language Processing Mini Project • Seamlessly Integrating All 10 NLP Phases</div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
nav_main, nav_lab, nav_bench, nav_report = st.tabs([
    "🚀 Live 10-Phase NLP Pipeline",
    "🔬 10-Phase Interactive Laboratory",
    "📈 Model Benchmarks & Datasets",
    "📑 Formal Project Report"
])

# Quick Pre-configured Scenarios
QUICK_PRESETS = {
    "Select a pre-configured scenario or type freely below:": "",
    "Scenario 1 (Positive Dining Experience & Course Meal)": "Tasty Bites in Pune serves delicious thin-crust pizza and the main course was great on Sunday!",
    "Scenario 2 (Food Discovery Search Query)": "Find authentic restaurants serving spicy mutton biryani in Pune.",
    "Scenario 3 (Customer Quality Complaint)": "The pizza was cold and greasy, worst dining experience ever.",
    "Scenario 4 (Context Ambiguity Test - Educational)": "I completed an advanced machine learning course at the university.",
    "Scenario 5 (Restaurant Brand Search)": "Little Italy in Pune serves gourmet wood-fired pizza and truffle risotto."
}

# ==============================================================================
# TAB 1: LIVE 10-PHASE PIPELINE (MAIN WORKSPACE)
# ==============================================================================
with nav_main:
    col_input, col_output = st.columns([1, 1.35], gap="large")

    if "live_query_input" not in st.session_state:
        st.session_state["live_query_input"] = ""

    with col_input:
        st.markdown("#### Input Workspace")
        
        selected_scenario = st.selectbox(
            "Quick Demonstration Scenarios:",
            list(QUICK_PRESETS.keys()),
            index=0
        )
        if selected_scenario != "Select a pre-configured scenario or type freely below:":
            st.session_state["live_query_input"] = QUICK_PRESETS[selected_scenario]

        col_b1, col_b2 = st.columns(2)
        rev_file = os.path.join(DATA_DIR, "restaurant_reviews.csv")
        rest_file = os.path.join(DATA_DIR, "restaurant_data.csv")

        if col_b1.button("🎲 Sample from Reviews CSV", use_container_width=True):
            if os.path.exists(rev_file):
                df_rev_tmp = pd.read_csv(rev_file)
                st.session_state["live_query_input"] = str(df_rev_tmp.sample(1).iloc[0]["review"])
                st.rerun()

        if col_b2.button("🍜 Sample from Catalog CSV", use_container_width=True):
            if os.path.exists(rest_file):
                df_rest_tmp = pd.read_csv(rest_file)
                row_tmp = df_rest_tmp.sample(1).iloc[0]
                st.session_state["live_query_input"] = f"Find restaurants in {row_tmp['location']} serving {row_tmp['popular_dishes']}"
                st.rerun()

        user_text = st.text_area(
            "Enter Restaurant Name, Food Dish, Review, or Query:",
            value=st.session_state["live_query_input"],
            height=130,
            placeholder="Type any restaurant name, dish, customer review, or search query here..."
        )

        if user_text != st.session_state["live_query_input"]:
            st.session_state["live_query_input"] = user_text

        col_run, col_clear = st.columns([2, 1])
        run_clicked = col_run.button("⚡ Execute 10-Phase NLP Pipeline", type="primary", use_container_width=True)
        clear_clicked = col_clear.button("Clear Input", use_container_width=True)

        if clear_clicked:
            st.session_state["live_query_input"] = ""
            st.rerun()

    with col_output:
        st.markdown("#### Real-Time Pipeline Results")
        active_text = user_text.strip()

        if (run_clicked or active_text) and active_text:
            with st.spinner("Executing all 10 NLP phases..."):
                pipeline_data = run_complete_nlp_pipeline(active_text)
                summary = pipeline_data["summary"]

            # Executive Metrics Banner
            s_label = summary["sentiment"]
            s_conf = summary["sentiment_confidence"]
            top_rec = summary["top_restaurant_recommendation"]
            top_score = summary["similarity_score"]
            chunks_cnt = summary["extracted_chunks_count"]
            wsd_sense = summary["disambiguated_sense"]

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Predicted Sentiment", s_label, delta=s_conf)
            col_m2.metric("Script Validation", "Supported (English)" if summary["script_valid"] else "Invalid Script")
            col_m3.metric("Food Chunks Extracted", f"{chunks_cnt} Phrases")
            col_m4.metric("Top Recommendation", top_rec, delta=top_score)

            if s_label == "Positive":
                st.markdown(f'<div class="summary-card-pos">✓ Predicted Sentiment: POSITIVE ({s_conf} confidence) | Top Match: {top_rec} ({top_score})</div>', unsafe_allow_html=True)
            elif s_label == "Negative":
                st.markdown(f'<div class="summary-card-neg">✗ Predicted Sentiment: NEGATIVE ({s_conf} confidence) | Top Match: {top_rec} ({top_score})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="summary-card-neu">~ Predicted Sentiment: NEUTRAL ({s_conf} confidence) | Top Match: {top_rec} ({top_score})</div>', unsafe_allow_html=True)

            # 10 Phases Expandable Breakdown
            st.markdown("##### Detailed 10-Phase Pipeline Breakdown:")

            # Phase 1
            with st.expander("Phase 1: Sentiment & Opinion Classification (TF-IDF + Logistic Regression)", expanded=True):
                p1 = pipeline_data["experiment_1_sentiment"]
                col_p1a, col_p1b = st.columns([1, 1.5])
                with col_p1a:
                    st.write(f"**Classification:** `{p1['predicted_sentiment']}`")
                    st.write(f"**Confidence:** `{round(p1['confidence'] * 100, 1)}%`")
                    st.write(f"**Polarity Score:** `{p1['polarity_score']}`")
                with col_p1b:
                    st.bar_chart(pd.DataFrame(list(p1["probabilities"].items()), columns=["Class", "Probability"]).set_index("Class"), height=160)

            # Phase 2
            with st.expander("Phase 2: Text Tokenization, Filtration & Script Validation", expanded=False):
                p2 = pipeline_data["experiment_2_tokenization"]
                st.caption(f"Script Status: {p2['script_validation']['status']} ({p2['script_validation']['english_char_percentage']}% valid)")
                st.write("**Sentence Tokens:**", p2["sentence_tokens"])
                st.write("**Filtered Word Tokens (Punctuation Removed):**", p2["filtered_tokens"])

            # Phase 3
            with st.expander("Phase 3: Stop Word Removal, Stemming & Lemmatization (Negation-Preserving)", expanded=False):
                p3 = pipeline_data["experiment_3_preprocessing"]
                st.caption(f"Filtered {p3['removed_stopwords_count']} stop words. Retained {p3['retained_tokens_count']} canonical roots.")
                st.dataframe(pd.DataFrame(p3["comparison_table"])[['original', 'pos_tag', 'is_stopword', 'stemmed', 'lemmatized']], height=180, use_container_width=True)

            # Phase 4
            with st.expander("Phase 4: Morphological Analysis & Search Query Expansion", expanded=False):
                p4 = pipeline_data["experiment_4_morphology"]
                st.write(f"**Expanded Search Query:** `{p4['query_expansion']['expanded_query_string']}`")
                if p4["word_analyses"]:
                    st.dataframe(pd.DataFrame(p4["word_analyses"])[['input_word', 'root_word', 'generated_forms']], height=160, use_container_width=True)

            # Phase 5
            with st.expander("Phase 5: N-Gram Language Model with Boundary Markers & Autocomplete", expanded=False):
                p5 = pipeline_data["experiment_5_ngram"]
                col_ng1, col_ng2 = st.columns(2)
                with col_ng1:
                    st.caption("Top Bigrams (with <s>, </s> markers):")
                    st.dataframe(pd.DataFrame(p5["bigrams"][:5]), height=150, use_container_width=True)
                with col_ng2:
                    st.caption("Next-Word Autocomplete Suggestions:")
                    if p5.get("next_word_predictions"):
                        st.dataframe(pd.DataFrame(p5["next_word_predictions"]), height=150, use_container_width=True)
                    else:
                        st.caption("No autocomplete predictions for single word.")

            # Phase 6
            with st.expander("Phase 6: Part-of-Speech (POS) Tagging & Syntactic Analysis", expanded=False):
                p6 = pipeline_data["experiment_6_pos"]
                st.dataframe(pd.DataFrame(p6["detailed_tags"])[['word', 'tag', 'description', 'category']], height=180, use_container_width=True)

            # Phase 7
            with st.expander("Phase 7: Food Phrase Chunking & Training Size Evaluation", expanded=False):
                p7 = pipeline_data["experiment_7_chunking"]
                st.write("**Extracted Food & Service Noun Phrases:**")
                if p7["rule_based_chunks"]:
                    st.dataframe(pd.DataFrame(p7["rule_based_chunks"]), height=140, use_container_width=True)
                st.caption("Syllabus Training Size Benchmark Comparison (60%, 70%, 80%):")
                st.table(pd.DataFrame(p7["training_size_comparison"])[['training_size_pct', 'train_samples', 'test_samples', 'accuracy', 'f1_score']])

            # Phase 8
            with st.expander("Phase 8: Named Entity Recognition (NER) for Restaurant Domain", expanded=True):
                p8 = pipeline_data["experiment_8_ner"]
                st.markdown(f"<div style='font-size: 1.15rem; line-height: 2.2;'>{p8['annotated_html']}</div>", unsafe_allow_html=True)
                if p8["entities"]:
                    st.dataframe(pd.DataFrame(p8["entities"])[["entity", "type", "start", "end"]], height=140, use_container_width=True)

            # Phase 9
            with st.expander("Phase 9: Text Similarity & Semantic Restaurant Recommendations", expanded=True):
                p9 = pipeline_data["experiment_9_similarity"]
                for r in p9["restaurant_matches"][:3]:
                    st.markdown(f"""
                    <div class="recommend-card">
                        <b>#{r['rank']} {r['restaurant_name']}</b> &nbsp;—&nbsp; {r['cuisine']} ({r['location']})
                        <span style="float: right; color: #2563EB; font-weight: 700;">Match: {r['similarity_percentage']}</span><br>
                        <small>Popular Dishes: {r['popular_dishes']}</small>
                    </div>
                    """, unsafe_allow_html=True)

            # Phase 10
            with st.expander("Phase 10: Word Sense Disambiguation using Deep Learning (Bi-LSTM)", expanded=True):
                p10 = pipeline_data["experiment_10_wsd"]
                st.write(f"Target Polysemous Word: **'{p10['target_word']}'**")
                st.write(f"Predicted Context Sense: **{p10['predicted_sense']}** ({p10['confidence']}% confidence)")
                st.caption(f"Contextual Meaning: {p10['sense_meaning']}")
                st.bar_chart(pd.DataFrame(list(p10["probabilities"].items()), columns=["Sense", "Probability"]).set_index("Sense"), height=160)

        else:
            st.info("👈 Enter a restaurant name, dish, customer review, or query on the left, then click **'⚡ Execute 10-Phase NLP Pipeline'**.")


# ==============================================================================
# TAB 2: 10-PHASE INTERACTIVE LABORATORY (ISOLATED EXECUTION)
# ==============================================================================
with nav_lab:
    st.markdown("### 🔬 10-Phase Interactive Laboratory")
    st.write("Demonstrate each individual phase with custom inputs and isolated parameters:")

    phase_tabs = st.tabs([
        "Phase 1: Sentiment",
        "Phase 2: Tokenization",
        "Phase 3: Stop/Stem/Lemma",
        "Phase 4: Morphology",
        "Phase 5: N-Grams",
        "Phase 6: POS Tagging",
        "Phase 7: Chunking",
        "Phase 8: NER",
        "Phase 9: Similarity",
        "Phase 10: Neural WSD"
    ])

    # Phase 1 Lab
    with phase_tabs[0]:
        st.subheader("Phase 1 — Sentiment / Opinion Analysis")
        analyzer = get_sentiment_analyzer()
        test_p1 = st.text_input("Enter review to classify sentiment:", "The pasta was bland and completely tasteless, definitely not returning.", key="lab_p1")
        if st.button("Classify Review Sentiment", key="btn_p1"):
            res1 = analyzer.predict(test_p1)
            st.success(f"Sentiment: {res1['predicted_sentiment']} (Confidence: {res1['confidence'] * 100:.1f}%)")
            st.json(res1["probabilities"])

    # Phase 2 Lab
    with phase_tabs[1]:
        st.subheader("Phase 2 — Tokenization, Filtration & Script Validation")
        test_p2 = st.text_area("Input text for tokenization & script validation:", "Tasty Bites in Pune serves authentic Italian thin-crust pizza! Order online now?", key="lab_p2")
        if st.button("Run Tokenization & Validation", key="btn_p2"):
            res2 = run_experiment_2(test_p2)
            st.write(f"**Script Validation Status:** {res2['script_validation']['status']}")
            col_l2a, col_l2b = st.columns(2)
            col_l2a.write("**Sentence Tokens:**", res2["sentence_tokens"])
            col_l2b.write("**Filtered Word Tokens:**", res2["filtered_tokens"])

    # Phase 3 Lab
    with phase_tabs[2]:
        st.subheader("Phase 3 — Stop Word Removal, Stemming & Lemmatization")
        test_p3 = st.text_input("Input sentence:", "I am looking for restaurants serving delicious foods and not cold drinks.", key="lab_p3")
        if st.button("Process Stopwords, Stemming & Lemmatization", key="btn_p3"):
            toks_p3 = run_experiment_2(test_p3)["filtered_tokens"]
            res3 = run_experiment_3(toks_p3)
            st.dataframe(pd.DataFrame(res3["comparison_table"]), use_container_width=True)

    # Phase 4 Lab
    with phase_tabs[3]:
        st.subheader("Phase 4 — Morphological Analysis & Search Query Expansion")
        test_p4 = st.text_input("Enter culinary base word (e.g., serve, cook, recommend, bake):", "serve", key="lab_p4")
        if st.button("Analyze Word Morphology", key="btn_p4"):
            m_res = get_morphological_analyzer().analyze_word(test_p4)
            st.json(m_res)
            st.write(f"**Generated Inflectional Forms:** `{', '.join(m_res['generated_forms'])}`")

    # Phase 5 Lab
    with phase_tabs[4]:
        st.subheader("Phase 5 — N-Gram Language Model with Boundary Markers")
        test_p5 = st.text_area("Input corpus text:", "delicious Italian pizza and spicy Indian food in Pune", key="lab_p5")
        if st.button("Generate N-Grams", key="btn_p5"):
            res5 = run_experiment_5(test_p5)
            col_l5a, col_l5b, col_l5c = st.columns(3)
            col_l5a.dataframe(pd.DataFrame(res5["unigrams"]), use_container_width=True)
            col_l5b.dataframe(pd.DataFrame(res5["bigrams"]), use_container_width=True)
            col_l5c.dataframe(pd.DataFrame(res5["trigrams"]), use_container_width=True)

    # Phase 6 Lab
    with phase_tabs[5]:
        st.subheader("Phase 6 — Part-of-Speech (POS) Tagging")
        test_p6 = st.text_input("Input sentence:", "They serve authentic Italian pasta carbonara in Pune.", key="lab_p6")
        if st.button("Generate POS Tags", key="btn_p6"):
            toks_p6 = run_experiment_2(test_p6)["filtered_tokens"]
            res6 = run_experiment_6(toks_p6)
            st.dataframe(pd.DataFrame(res6["detailed_tags"]), use_container_width=True)

    # Phase 7 Lab
    with phase_tabs[6]:
        st.subheader("Phase 7 — Food Phrase Chunking & Training Size Evaluation")
        test_p7 = st.text_input("Sentence for chunking:", "The friendly waiter served very spicy Indian food and delicious pizza.", key="lab_p7")
        if st.button("Extract Phrases & Benchmark", key="btn_p7"):
            toks_p7 = run_experiment_2(test_p7)["filtered_tokens"]
            tags_p7 = run_experiment_6(toks_p7)["tagged_tokens"]
            res7 = run_experiment_7(tags_p7)
            st.dataframe(pd.DataFrame(res7["rule_based_chunks"]), use_container_width=True)
            st.caption("Training Size vs Accuracy Curve:")
            bench7 = run_chunking_training_experiment()
            st.dataframe(pd.DataFrame(bench7["all_experiments"]), use_container_width=True)

    # Phase 8 Lab
    with phase_tabs[7]:
        st.subheader("Phase 8 — Named Entity Recognition (NER)")
        test_p8 = st.text_area("Input text with restaurants, foods, locations, or dates:", "Tasty Bites in Pune serves delicious pizza on Sunday. Chef Sanjeev visited yesterday.", key="lab_p8")
        if st.button("Identify Named Entities", key="btn_p8"):
            res8 = run_experiment_8(test_p8)
            st.markdown(res8["annotated_html"], unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(res8["entities"]), use_container_width=True)

    # Phase 9 Lab
    with phase_tabs[8]:
        st.subheader("Phase 9 — Text Similarity & Restaurant Recommendations")
        test_p9 = st.text_input("Search query or dish description:", "spicy mutton biryani with raita", key="lab_p9")
        if st.button("Find Similar Restaurants", key="btn_p9"):
            res9 = run_experiment_9(test_p9, top_k=5)
            st.dataframe(pd.DataFrame(res9["restaurant_matches"])[["rank", "restaurant_name", "location", "cuisine", "similarity_percentage", "rating"]], use_container_width=True)

    # Phase 10 Lab
    with phase_tabs[9]:
        st.subheader("Phase 10 — Word Sense Disambiguation using Deep Learning (Bi-LSTM)")
        test_p10 = st.text_input("Context sentence with polysemous word:", "The main course was delicious and hot.", key="lab_p10")
        target_w = st.selectbox("Target ambiguous word:", ["course", "bill", "order"], key="w_p10")
        if st.button("Disambiguate Contextual Sense", key="btn_p10"):
            res10 = run_experiment_10(test_p10, target_word=target_w)
            st.success(f"Disambiguated Sense: {res10['predicted_sense']} ({res10['confidence']}% confidence)")
            st.write(f"Contextual Meaning: {res10['sense_meaning']}")
            st.json(res10["probabilities"])


# ==============================================================================
# TAB 3: MODEL BENCHMARKS & DATASETS
# ==============================================================================
with nav_bench:
    st.markdown("### 📈 Model Evaluation Benchmarks & Curated Datasets")

    col_bench1, col_bench2 = st.columns(2)
    with col_bench1:
        st.markdown("#### Sentiment Classifier Confusion Matrix (Phase 1)")
        cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix: Actual vs Predicted Sentiment (85.71% Accuracy)", use_container_width=True)

    with col_bench2:
        st.markdown("#### Chunking Training Size Accuracy Curve (Phase 7)")
        chunk_curve = os.path.join(OUTPUTS_DIR, "chunking_training_size_curve.png")
        if os.path.exists(chunk_curve):
            st.image(chunk_curve, caption="Impact of Training Size on Chunking Performance (60%, 70%, 80%)", use_container_width=True)

    st.divider()

    st.markdown("#### Curated Domain Datasets Preview:")
    d_tab1, d_tab2, d_tab3 = st.tabs(["Customer Reviews CSV", "Restaurant Catalog CSV", "WSD Training Dataset CSV"])

    with d_tab1:
        if os.path.exists(rev_file):
            df_rev_view = pd.read_csv(rev_file)
            st.caption(f"Total reviews: {len(df_rev_view)}")
            st.dataframe(df_rev_view, height=220, use_container_width=True)

    with d_tab2:
        if os.path.exists(rest_file):
            df_rest_view = pd.read_csv(rest_file)
            st.caption(f"Total restaurants: {len(df_rest_view)}")
            st.dataframe(df_rest_view[['restaurant_name', 'location', 'cuisine', 'rating', 'popular_dishes']], height=220, use_container_width=True)

    with d_tab3:
        wsd_file = os.path.join(DATA_DIR, "wsd_dataset.csv")
        if os.path.exists(wsd_file):
            df_wsd_view = pd.read_csv(wsd_file)
            st.caption(f"Total contextual sentences: {len(df_wsd_view)}")
            st.dataframe(df_wsd_view, height=220, use_container_width=True)


# ==============================================================================
# TAB 4: FORMAL COLLEGE PROJECT REPORT
# ==============================================================================
with nav_report:
    st.markdown("### 📑 Formal College Project Report (24 Sections)")
    st.write("Comprehensive technical report aligned with university curriculum standards:")

    report_path = os.path.join(BASE_DIR, "FINAL_REPORT.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()

        st.download_button(
            label="📥 Download Full College Project Report (.MD)",
            data=report_text,
            file_name="Restaurant_NLP_Project_Report.md",
            mime="text/markdown",
            type="primary"
        )
        st.divider()
        st.markdown(report_text)
    else:
        st.error("Report file FINAL_REPORT.md not found.")
