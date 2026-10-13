"""
Restaurant Review & Food Query Analysis System Using Natural Language Processing
College NLP Mini Project integrating Phases 1 through 10.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 and project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.sentiment import run_experiment_1, get_sentiment_analyzer
from modules.preprocessing import run_experiment_2, run_experiment_3, validate_script
from modules.morphology import run_experiment_4, get_morphological_analyzer
from modules.ngram import run_experiment_5, get_ngram_model
from modules.pos_tagging import run_experiment_6
from modules.chunking import run_experiment_7, run_chunking_training_experiment
from modules.ner import run_experiment_8
from modules.similarity import run_experiment_9, get_similarity_engine
from modules.wsd import run_experiment_10, get_wsd_disambiguator
from modules.pipeline import run_complete_nlp_pipeline

# Streamlit Page Config
st.set_page_config(
    page_title="Restaurant NLP Intelligence System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        color: #FFFFFF;
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.2);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.92;
        margin-bottom: 0;
        line-height: 1.4;
    }
    
    .phase-badge {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 0.3rem 0.75rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    
    .restaurant-card {
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .restaurant-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .target-restaurant-hero {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 6px solid #10B981;
        border-radius: 10px;
        padding: 1.3rem;
        margin-bottom: 1.5rem;
    }
    
    .match-tag {
        background: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        font-size: 0.88rem;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid #BBF7D0;
    }
</style>
""", unsafe_allow_html=True)

# Load Catalog & Datasets
df_restaurants = pd.read_csv(os.path.join(DATA_DIR, "restaurant_data.csv"))
df_reviews = pd.read_csv(os.path.join(DATA_DIR, "restaurant_reviews.csv"))

# Sidebar Navigation
st.sidebar.markdown("### 🍽️ Restaurant NLP Portal")
st.sidebar.caption("Unified 10-Phase Natural Language Processing System")

app_mode = st.sidebar.radio(
    "Navigation Mode:",
    [
        "🍽️ Restaurant Predictor & 10-Phase NLP Analysis",
        "🔍 Custom Query / Review Search Pipeline",
        "🔬 NLP 10-Phase Interactive Laboratory",
        "📊 Restaurant Catalog & Datasets",
        "📑 Formal Project Report (24 Sections)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Syllabus Phases Covered:**
- **Phase 1:** Sentiment & Applications
- **Phase 2:** Tokenization & Script Validation
- **Phase 3:** Stopwords, Stemming & Lemmatization
- **Phase 4:** Morphology & Word Generation
- **Phase 5:** N-Gram Language Modeling
- **Phase 6:** POS Tagging & Syntactic Analysis
- **Phase 7:** Chunking & Feature Selection
- **Phase 8:** Named Entity Recognition (NER)
- **Phase 9:** Similarity & Recommendations
- **Phase 10:** Deep Learning WSD (Bi-LSTM)
""")


# ==============================================================================
# MODE 1: RESTAURANT PREDICTOR & 10-PHASE NLP PIPELINE ON ANY CATALOG VENUE
# ==============================================================================
if app_mode == "🍽️ Restaurant Predictor & 10-Phase NLP Analysis":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🍽️ Restaurant Predictor & 10-Phase NLP Pipeline</div>
        <div class="hero-subtitle">Select any restaurant from all 35 catalog venues or enter a custom name to execute sequential linguistic analysis and predict top matching recommendations.</div>
    </div>
    """, unsafe_allow_html=True)

    # Format restaurant display options for all 35 venues
    rest_options = []
    for idx, row in df_restaurants.iterrows():
        rest_options.append(f"{idx + 1}: {row['restaurant_name']} — {row['cuisine']} [{row['location']}]")

    col_sel, col_type = st.columns([3, 2])
    with col_sel:
        selected_option = st.selectbox(
            "Select any restaurant from the full catalog (All 35 Venues available):",
            rest_options,
            index=0
        )
        selected_idx = int(selected_option.split(":")[0]) - 1
        rest_row = df_restaurants.iloc[selected_idx]

    with col_type:
        custom_input = st.text_input(
            "Or search by restaurant name / dish / keyword / ID:",
            placeholder="e.g. 1, Tasty Bites, Little Italy, Biryani By Kilo, pizza..."
        )

    # Resolve target restaurant
    if custom_input.strip():
        q_clean = custom_input.strip()
        matched = pd.DataFrame()
        if q_clean.isdigit():
            val = int(q_clean)
            if 1 <= val <= len(df_restaurants):
                matched = df_restaurants.iloc[[val - 1]]
        if matched.empty:
            matched = df_restaurants[df_restaurants["restaurant_name"].str.lower().str.contains(q_clean.lower(), regex=False)]
        if matched.empty:
            matched = df_restaurants[df_restaurants["popular_dishes"].str.lower().str.contains(q_clean.lower(), regex=False)]
        if matched.empty:
            matched = df_restaurants[df_restaurants["cuisine"].str.lower().str.contains(q_clean.lower(), regex=False)]

        if not matched.empty:
            rest_row = matched.iloc[0]
            selected_idx = df_restaurants.index[df_restaurants['restaurant_name'] == rest_row['restaurant_name']][0]
        else:
            st.warning(f"No match found for '{custom_input}'. Using dropdown selection: '{rest_row['restaurant_name']}'.")

    target_name = rest_row["restaurant_name"]
    target_loc = rest_row["location"]
    target_cuisine = rest_row["cuisine"]
    target_rating = rest_row["rating"]
    target_dishes = rest_row["popular_dishes"]
    target_desc = rest_row["description"]
    target_id = selected_idx + 1

    # Target Restaurant Hero Card
    st.markdown(f"""
    <div class="target-restaurant-hero">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <span class="phase-badge">CATALOG RESTAURANT #{target_id}</span>
                <h3 style="margin:0.2rem 0; color:#0F172A;">{target_name}</h3>
                <p style="margin:0.3rem 0; color:#475569; font-size:0.95rem;">
                    <strong>Location:</strong> {target_loc} &nbsp;|&nbsp; 
                    <strong>Cuisine:</strong> {target_cuisine} &nbsp;|&nbsp; 
                    <strong>Rating:</strong> ⭐ {target_rating}/5.0 &nbsp;|&nbsp; 
                    <strong>Popular Dishes:</strong> {target_dishes}
                </p>
            </div>
        </div>
        <p style="margin:0.7rem 0 0 0; color:#334155; font-size:0.95rem; line-height:1.5;">
            <strong>Description:</strong> <em>"{target_desc}"</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Execute Sequential 10-Phase NLP Pipeline & Predict Similar Restaurants", type="primary"):
        target_full_text = f"{target_name} in {target_loc} serves {target_dishes}. {target_desc}"
        
        with st.spinner("Processing restaurant profile across all 10 NLP phases..."):
            # Phase 1: Sentiment Analysis
            p1_res = run_experiment_1(target_full_text)

            # Phase 2: Tokenization & Script Validation
            p2_res = run_experiment_2(target_full_text)

            # Phase 3: Stopwords, Stemming & Lemmatization
            p3_res = run_experiment_3(p2_res["filtered_tokens"])

            # Phase 4: Morphology
            p4_res = run_experiment_4(p3_res["tokens_without_stopwords"])

            # Phase 5: N-Gram Model
            p5_res = run_experiment_5(target_full_text)

            # Phase 6: POS Tagging
            p6_res = run_experiment_6(p2_res["filtered_tokens"])

            # Phase 7: Chunking
            p7_res = run_experiment_7(p6_res["tagged_tokens"])

            # Phase 8: Named Entity Recognition
            p8_res = run_experiment_8(target_full_text)

            # Phase 9: Text Similarity & Recommendations
            p9_res = run_experiment_9(target_full_text, top_k=5)

            # Phase 10: Deep Learning WSD
            target_poly = "course" if "course" in target_full_text.lower() else ("bill" if "bill" in target_full_text.lower() else "course")
            p10_res = run_experiment_10(target_full_text, target_word=target_poly)

        # Presentation of Results
        st.markdown("### 🎯 Recommended Restaurants (Phase 9 - TF-IDF Cosine Similarity)")
        recs = p9_res.get("restaurant_matches", [])
        if recs:
            for idx, r in enumerate(recs[:4], 1):
                st.markdown(f"""
                <div class="restaurant-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:#1E3A8A;">#{idx} {r['restaurant_name']}</h4>
                        <span class="match-tag">Cosine Match: {r['similarity_percentage']}</span>
                    </div>
                    <p style="margin:4px 0; color:#4B5563; font-size:0.9rem;">
                        <strong>Location:</strong> {r['location']} &nbsp;|&nbsp; 
                        <strong>Cuisine:</strong> {r['cuisine']} &nbsp;|&nbsp; 
                        <strong>Rating:</strong> ⭐ {r['rating']}/5.0
                    </p>
                    <p style="margin:4px 0; color:#374151; font-size:0.88rem;">
                        <strong>Popular Dishes:</strong> {r['popular_dishes']}
                    </p>
                    <p style="margin:6px 0 0 0; color:#1F2937; font-size:0.92rem; line-height:1.4;">
                        <em>"{r['description']}"</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No similarity results returned.")

        st.markdown("---")
        st.markdown("### 🧠 Detailed Linguistic Analysis (Phases 1 through 10)")

        tabs = st.tabs([
            "Phase 1: Sentiment",
            "Phase 2: Tokenization & Script",
            "Phase 3: Stem & Lemma",
            "Phase 4: Morphology",
            "Phase 5: N-Grams",
            "Phase 6: POS Tagging",
            "Phase 7: Phrase Chunking",
            "Phase 8: NER",
            "Phase 10: Bi-LSTM WSD"
        ])

        with tabs[0]:
            st.markdown('<span class="phase-badge">Phase 1</span> **Study various applications of NLP and Formulate the Problem Statement (Sentiment & Opinion Analysis)**', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Predicted Sentiment", p1_res["predicted_sentiment"])
            c1.metric("Confidence Score", f"{p1_res['confidence']*100:.1f}%")
            c1.metric("Polarity Score", p1_res["polarity_score"])
            with c2:
                if p1_res.get("probabilities"):
                    st.write("**Probability Distribution:**")
                    pdf = pd.DataFrame(list(p1_res["probabilities"].items()), columns=["Sentiment", "Probability"])
                    st.bar_chart(pdf.set_index("Sentiment"))

        with tabs[1]:
            st.markdown('<span class="phase-badge">Phase 2</span> **Apply various text preprocessing techniques for any given text: Tokenization and Filtration & Script Validation**', unsafe_allow_html=True)
            v = p2_res["script_validation"]
            if v["is_valid"]:
                st.success(f"✓ Script Validation: Valid English Script ({v['english_char_percentage']}% standard characters)")
            else:
                st.error(f"⚠ Script Validation: {v['status']}")
            st.write(f"**Total Sentences:** {len(p2_res['sentence_tokens'])} | **Raw Word Tokens:** {len(p2_res['word_tokens'])}")
            st.write("**Filtered Tokens (Punctuation Stripped):**", p2_res["filtered_tokens"][:25])

        with tabs[2]:
            st.markdown('<span class="phase-badge">Phase 3</span> **Apply various other text preprocessing techniques for any given text: Stop Word Removal, Lemmatization / Stemming**', unsafe_allow_html=True)
            st.write(f"Removed **{p3_res['removed_stopwords_count']}** stop words. Retained **{p3_res['retained_tokens_count']}** canonical linguistic tokens.")
            st.markdown("**Comparative Linguistic Transformation:**")
            st.dataframe(pd.DataFrame(p3_res["comparison_table"]), use_container_width=True)

        with tabs[3]:
            st.markdown('<span class="phase-badge">Phase 4</span> **Morphological Analysis and Word Generation for Search Query Expansion**', unsafe_allow_html=True)
            st.write(f"**Expanded Search Query:** `{p4_res['query_expansion']['expanded_query_string']}`")
            if p4_res["word_analyses"]:
                st.markdown("**Word Root & Affix Decomposition with Generated Word Forms:**")
                st.dataframe(pd.DataFrame(p4_res["word_analyses"])[["input_word", "root_word", "prefix", "suffix", "generated_forms"]], use_container_width=True)

        with tabs[4]:
            st.markdown('<span class="phase-badge">Phase 5</span> **N-Gram Language Model with Boundary Markers and Query Prediction**', unsafe_allow_html=True)
            col_u, col_b, col_t = st.columns(3)
            with col_u:
                st.write("**Top Unigrams:**")
                st.dataframe(pd.DataFrame(p5_res["unigrams"][:6]), use_container_width=True)
            with col_b:
                st.write("**Top Bigrams (with <s>, </s>):**")
                st.dataframe(pd.DataFrame(p5_res["bigrams"][:6]), use_container_width=True)
            with col_t:
                st.write("**Top Trigrams:**")
                st.dataframe(pd.DataFrame(p5_res["trigrams"][:5]), use_container_width=True)
            if p5_res.get("next_word_predictions"):
                pred_words = [p['predicted_next_word'] for p in p5_res["next_word_predictions"] if 'predicted_next_word' in p]
                st.success(f"Next-Word Autocomplete Suggestion: **{', '.join(pred_words)}**")

        with tabs[5]:
            st.markdown('<span class="phase-badge">Phase 6</span> **Part-of-Speech (POS) Tagging and Syntactic Analysis**', unsafe_allow_html=True)
            df_pos = pd.DataFrame(p6_res["detailed_tags"])
            st.dataframe(df_pos[["word", "tag", "category", "description"]], use_container_width=True)
            st.bar_chart(pd.DataFrame(list(p6_res["category_distribution"].items()), columns=["Category", "Count"]).set_index("Category"))

        with tabs[6]:
            st.markdown('<span class="phase-badge">Phase 7</span> **Chunking (Phrase Extraction) and Empirical Analysis on Feature Selection and Training Size (60%, 70%, 80%)**', unsafe_allow_html=True)
            if p7_res["rule_based_chunks"]:
                st.write("**Extracted Food & Service Noun Phrases:**")
                st.dataframe(pd.DataFrame(p7_res["rule_based_chunks"]), use_container_width=True)
            st.markdown("**Syllabus Benchmark: Training Size vs Model Accuracy:**")
            st.table(pd.DataFrame(p7_res["training_size_comparison"])[["training_size_pct", "train_samples", "test_samples", "accuracy", "f1_score"]])

        with tabs[7]:
            st.markdown('<span class="phase-badge">Phase 8</span> **Named Entity Recognition (NER) for Restaurant Domain**', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 1.15rem; line-height: 2.2;'>{p8_res['annotated_html']}</div>", unsafe_allow_html=True)
            if p8_res["entities"]:
                st.dataframe(pd.DataFrame(p8_res["entities"])[["entity", "type", "start", "end"]], use_container_width=True)

        with tabs[8]:
            st.markdown('<span class="phase-badge">Phase 10</span> **Word Sense Disambiguation (WSD) using Deep Learning (Bi-LSTM / GRU)**', unsafe_allow_html=True)
            st.write(f"**Disambiguation Target Word:** `{p10_res['target_word']}`")
            c_w1, c_w2 = st.columns(2)
            c_w1.metric("Predicted Sense", p10_res["predicted_sense"])
            c_w2.metric("Neural Confidence", f"{p10_res['confidence']}%")
            st.info(f"**Sense Meaning:** {p10_res['sense_meaning']}")
            st.bar_chart(pd.DataFrame(list(p10_res["probabilities"].items()), columns=["Sense", "Probability"]).set_index("Sense"))


# ==============================================================================
# MODE 2: CUSTOM SEARCH QUERY & REVIEW PIPELINE
# ==============================================================================
elif app_mode == "🔍 Custom Query / Review Search Pipeline":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔍 Custom Search Query & Review NLP Pipeline</div>
        <div class="hero-subtitle">Type any natural language food query, dining question, or customer review to process it through all 10 unified NLP stages.</div>
    </div>
    """, unsafe_allow_html=True)

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    if col_q1.button("🍕 Italian Pizza in Pune"):
        st.session_state["custom_query_val"] = "Find authentic Italian restaurants serving wood-fired thin-crust pizza in Pune."
    if col_q2.button("🍗 Spicy Mutton Biryani"):
        st.session_state["custom_query_val"] = "Looking for restaurants serving spicy mutton biryani with raita."
    if col_q3.button("⭐ Positive Dining Review"):
        st.session_state["custom_query_val"] = "This restaurant serves delicious food and the service is excellent."
    if col_q4.button("⚠️ Cold Food Complaint"):
        st.session_state["custom_query_val"] = "The pizza was cold and greasy, worst dining experience ever."

    default_q = st.session_state.get("custom_query_val", "Find authentic Italian restaurants serving wood-fired thin-crust pizza in Pune.")
    user_query = st.text_input("Enter natural language query or customer review:", value=default_q)

    if st.button("⚡ Run Full 10-Phase NLP Pipeline", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a query or review first!")
        else:
            with st.spinner("Executing all 10 NLP phases..."):
                p1_res = run_experiment_1(user_query)
                p2_res = run_experiment_2(user_query)
                p3_res = run_experiment_3(p2_res["filtered_tokens"])
                p4_res = run_experiment_4(p3_res["tokens_without_stopwords"])
                p5_res = run_experiment_5(user_query)
                p6_res = run_experiment_6(p2_res["filtered_tokens"])
                p7_res = run_experiment_7(p6_res["tagged_tokens"])
                p8_res = run_experiment_8(user_query)
                p9_res = run_experiment_9(user_query, top_k=5)
                target_w = "course" if "course" in user_query.lower() else ("bill" if "bill" in user_query.lower() else "order" if "order" in user_query.lower() else "course")
                p10_res = run_experiment_10(user_query, target_word=target_w)

            # Phase 9 Recommendations
            st.markdown("### 🎯 Recommended Restaurants (Phase 9 - Content-Based TF-IDF Cosine Similarity)")
            recs = p9_res.get("restaurant_matches", [])
            if recs:
                for idx, r in enumerate(recs[:4], 1):
                    st.markdown(f"""
                    <div class="restaurant-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="margin:0; color:#1E3A8A;">#{idx} {r['restaurant_name']}</h4>
                            <span class="match-tag">Match: {r['similarity_percentage']}</span>
                        </div>
                        <p style="margin:4px 0; color:#4B5563; font-size:0.9rem;">
                            <strong>Location:</strong> {r['location']} &nbsp;|&nbsp; 
                            <strong>Cuisine:</strong> {r['cuisine']} &nbsp;|&nbsp; 
                            <strong>Rating:</strong> ⭐ {r['rating']}/5.0
                        </p>
                        <p style="margin:4px 0; color:#374151; font-size:0.88rem;">
                            <strong>Popular Dishes:</strong> {r['popular_dishes']}
                        </p>
                        <p style="margin:6px 0 0 0; color:#1F2937; font-size:0.92rem; line-height:1.4;">
                            <em>"{r['description']}"</em>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No matching restaurants found for this query.")

            st.markdown("---")
            st.markdown("### 🧠 Linguistic Inspection Across Connected Phases")

            tabs = st.tabs([
                "Phase 1: Sentiment",
                "Phase 2: Tokenization & Script",
                "Phase 3: Stem & Lemma",
                "Phase 4: Morphology",
                "Phase 5: N-Grams",
                "Phase 6: POS Tags",
                "Phase 7: Chunking",
                "Phase 8: NER",
                "Phase 10: Deep WSD"
            ])

            with tabs[0]:
                st.markdown('<span class="phase-badge">Phase 1</span> **Sentiment & Review Classification**', unsafe_allow_html=True)
                s_col1, s_col2 = st.columns(2)
                s_col1.metric("Predicted Tone", p1_res["predicted_sentiment"])
                s_col1.metric("Confidence", f"{p1_res['confidence']*100:.1f}%")
                s_col1.metric("Polarity", p1_res["polarity_score"])
                with s_col2:
                    if p1_res.get("probabilities"):
                        pdf = pd.DataFrame(list(p1_res["probabilities"].items()), columns=["Sentiment", "Probability"])
                        st.bar_chart(pdf.set_index("Sentiment"))

            with tabs[1]:
                st.markdown('<span class="phase-badge">Phase 2</span> **Tokenization & Script Validation**', unsafe_allow_html=True)
                v = p2_res["script_validation"]
                if v["is_valid"]:
                    st.success(f"✓ Valid English Script: ({v['english_char_percentage']}% valid)")
                else:
                    st.error(f"⚠ Script Warning: {v['status']}")
                st.write("**Filtered Tokens:**", p2_res["filtered_tokens"][:25])

            with tabs[2]:
                st.markdown('<span class="phase-badge">Phase 3</span> **Stop Word Removal, Lemmatization / Stemming**', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(p3_res["comparison_table"]), use_container_width=True)

            with tabs[3]:
                st.markdown('<span class="phase-badge">Phase 4</span> **Morphology & Query Expansion**', unsafe_allow_html=True)
                st.write(f"**Expanded Query:** `{p4_res['query_expansion']['expanded_query_string']}`")

            with tabs[4]:
                st.markdown('<span class="phase-badge">Phase 5</span> **N-Gram Model & Next-Word Suggestions**', unsafe_allow_html=True)
                cu, cb = st.columns(2)
                cu.write("**Top Bigrams:**", p5_res["bigrams"][:5])
                cb.write("**Predictions:**", p5_res.get("next_word_predictions", []))

            with tabs[5]:
                st.markdown('<span class="phase-badge">Phase 6</span> **Part-of-Speech Tagging**', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(p6_res["detailed_tags"])[["word", "tag", "category", "description"]], use_container_width=True)

            with tabs[6]:
                st.markdown('<span class="phase-badge">Phase 7</span> **Phrase Chunking**', unsafe_allow_html=True)
                if p7_res["rule_based_chunks"]:
                    st.dataframe(pd.DataFrame(p7_res["rule_based_chunks"]), use_container_width=True)

            with tabs[7]:
                st.markdown('<span class="phase-badge">Phase 8</span> **Named Entity Recognition**', unsafe_allow_html=True)
                st.markdown(p8_res["annotated_html"], unsafe_allow_html=True)

            with tabs[8]:
                st.markdown('<span class="phase-badge">Phase 10</span> **Deep Learning WSD (Bi-LSTM)**', unsafe_allow_html=True)
                st.success(f"Target Word: **'{p10_res['target_word']}'** | Predicted Sense: **{p10_res['predicted_sense']}** ({p10_res['confidence']}%)")
                st.info(f"**Meaning:** {p10_res['sense_meaning']}")


# ==============================================================================
# MODE 3: NLP 10-PHASE INTERACTIVE LABORATORY
# ==============================================================================
elif app_mode == "🔬 NLP 10-Phase Interactive Laboratory":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔬 NLP 10-Phase Interactive Laboratory</div>
        <div class="hero-subtitle">Individual interactive workbench for demonstrating each syllabus phase independently with live controls, visualizations, and metrics.</div>
    </div>
    """, unsafe_allow_html=True)

    phase_choice = st.selectbox(
        "Select Phase to Explore:",
        [
            "Phase 1: Study various applications of NLP and Formulate the Problem Statement for Mini Project (Sentiment & Opinion Analysis)",
            "Phase 2: Apply various text preprocessing techniques for any given text: Tokenization and Filtration & Script Validation",
            "Phase 3: Apply various other text preprocessing techniques for any given text: Stop Word Removal, Lemmatization / Stemming",
            "Phase 4: Morphological Analysis and Word Generation for Search Query Expansion",
            "Phase 5: N-Gram Language Model and Query Autocomplete Prediction",
            "Phase 6: Part-of-Speech (POS) Tagging and Syntactic Analysis",
            "Phase 7: Chunking (Phrase Extraction) and Empirical Analysis on Feature Selection and Training Size (60%, 70%, 80%)",
            "Phase 8: Named Entity Recognition (NER)",
            "Phase 9: Text Similarity and Content-Based Restaurant Recommendation (TF-IDF Cosine Similarity)",
            "Phase 10: Word Sense Disambiguation (WSD) using Deep Learning (LSTM / GRU)"
        ]
    )

    # PHASE 1 LAB
    if "Phase 1:" in phase_choice:
        st.subheader("Phase 1: Sentiment Analysis of Customer Reviews")
        st.markdown("**Objective:** Train TF-IDF + Classifier to evaluate dining reviews as Positive, Negative, or Neutral.")
        
        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            analyzer = get_sentiment_analyzer()
            st.metric("Trained Classifier Accuracy", f"{analyzer.evaluation_results.get('accuracy', 0.857)*100:.2f}%")
            st.metric("Trained Model F1-Score", f"{analyzer.evaluation_results.get('f1_score', 0.858)*100:.2f}%")

        with m_col2:
            cm_img = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
            if os.path.exists(cm_img):
                st.image(cm_img, caption="Sentiment Classifier Confusion Matrix (Phase 1 Benchmark)", width=480)

        st.markdown("---")
        st.write("### Test Custom Review")
        rev_input = st.text_input("Enter a restaurant review to test:", value="The pasta was bland and completely tasteless, definitely not returning.")
        if st.button("Classify Review Sentiment"):
            pred = analyzer.predict(rev_input)
            st.success(f"Sentiment: **{pred['predicted_sentiment']}** (Confidence: {pred['confidence']*100:.1f}%)")
            st.json(pred["probabilities"])

    # PHASE 2 LAB
    elif "Phase 2:" in phase_choice:
        st.subheader("Phase 2: Tokenization and Filtration & Script Validation")
        st.markdown("**Objective:** Perform sentence tokenization, word tokenization, punctuation filtration, and English script validation.")
        
        test_text = st.text_area("Input Restaurant Query:", value="Tasty Bites in Pune serves authentic Italian thin-crust pizza! Order online now?")
        script_test = st.selectbox("Or choose a test query with unsupported scripts:", [
            "Default English Query",
            "Marathi / Hindi (Devanagari): हा एक चांगला रेस्टॉरंट आहे",
            "Russian (Cyrillic): Отличный ресторан",
            "Corrupted Unicode: Restaurant \u2603 \u2602 test"
        ])
        if script_test != "Default English Query":
            test_text = script_test.split(": ")[-1]

        if st.button("Run Tokenization & Validation"):
            res = run_experiment_2(test_text)
            c_val, c_tok = st.columns([1, 2])
            with c_val:
                st.write("#### Script Validation Result")
                v = res["script_validation"]
                if v["is_valid"]:
                    st.success(f"✅ {v['status']} ({v['english_char_percentage']}% standard characters)")
                else:
                    st.error(f"❌ {v['status']}")

            with c_tok:
                st.write("#### Tokenization Breakdown")
                st.write("**Sentence Tokens:**", res["sentence_tokens"])
                st.write("**Word Tokens:**", res["word_tokens"])
                st.write("**Filtered Clean Tokens:**", res["filtered_tokens"])

    # PHASE 3 LAB
    elif "Phase 3:" in phase_choice:
        st.subheader("Phase 3: Stop Word Removal, Lemmatization / Stemming")
        st.markdown("**Objective:** Remove grammatical stopwords (preserving negations), apply Porter Stemming, and compute WordNet Lemmatization.")
        
        query_e3 = st.text_input("Enter Restaurant Query:", value="I am looking for restaurants serving delicious foods and not cold drinks.")
        if st.button("Preprocess Text"):
            toks_p3 = run_experiment_2(query_e3)["filtered_tokens"]
            prep = run_experiment_3(toks_p3)
            st.write(f"Removed **{prep['removed_stopwords_count']}** stop words. Retained **{prep['retained_tokens_count']}** tokens.")
            st.write("### Comparative Table (Original vs Stemmed vs Lemmatized)")
            st.dataframe(pd.DataFrame(prep["comparison_table"]), use_container_width=True)

    # PHASE 4 LAB
    elif "Phase 4:" in phase_choice:
        st.subheader("Phase 4: Morphological Analysis and Word Generation")
        st.markdown("**Objective:** Decompose culinary words into root and affixes, and generate inflectional forms for query expansion.")
        
        m_word = st.text_input("Enter culinary word (e.g., serve, cook, recommend, bake):", value="serve")
        if st.button("Analyze Word Morphology"):
            m_res = get_morphological_analyzer().analyze_word(m_word)
            st.json(m_res)
            st.success(f"Generated Word Forms: **{', '.join(m_res['generated_forms'])}**")

    # PHASE 5 LAB
    elif "Phase 5:" in phase_choice:
        st.subheader("Phase 5: N-Gram Language Model and Query Autocomplete")
        st.markdown("**Objective:** Extract Unigrams, Bigrams (with `<s>`, `</s>`), and Trigrams, and predict next-word autocomplete suggestions.")
        
        corpus_input = st.text_area("Input corpus text:", value="delicious Italian pizza and spicy Indian food in Pune")
        if st.button("Generate N-Grams"):
            res5 = run_experiment_5(corpus_input)
            c1, c2, c3 = st.columns(3)
            c1.dataframe(pd.DataFrame(res5["unigrams"]), use_container_width=True)
            c2.dataframe(pd.DataFrame(res5["bigrams"]), use_container_width=True)
            c3.dataframe(pd.DataFrame(res5["trigrams"]), use_container_width=True)

    # PHASE 6 LAB
    elif "Phase 6:" in phase_choice:
        st.subheader("Phase 6: Part-of-Speech Tagging")
        st.markdown("**Objective:** Annotate tokens with Penn Treebank syntactic POS categories.")
        
        pos_in = st.text_input("Input sentence:", value="They serve authentic Italian pasta carbonara in Pune.")
        if st.button("Tag Parts of Speech"):
            toks_p6 = run_experiment_2(pos_in)["filtered_tokens"]
            res6 = run_experiment_6(toks_p6)
            st.dataframe(pd.DataFrame(res6["detailed_tags"]), use_container_width=True)

    # PHASE 7 LAB
    elif "Phase 7:" in phase_choice:
        st.subheader("Phase 7: Chunking & Training Size Evaluation")
        st.markdown("**Objective:** Extract descriptive food noun phrases and evaluate model accuracy across training sizes (60%, 70%, 80%).")
        
        chk_in = st.text_input("Sentence for phrase chunking:", value="The friendly waiter served very spicy Indian food and delicious pizza.")
        if st.button("Extract Chunks & Show Curve"):
            toks_p7 = run_experiment_2(chk_in)["filtered_tokens"]
            tags_p7 = run_experiment_6(toks_p7)["tagged_tokens"]
            res7 = run_experiment_7(tags_p7)
            st.dataframe(pd.DataFrame(res7["rule_based_chunks"]), use_container_width=True)
            
            chk_curve = os.path.join(OUTPUTS_DIR, "chunking_training_size_curve.png")
            if os.path.exists(chk_curve):
                st.image(chk_curve, caption="Training Size vs Accuracy Curve (60%, 70%, 80%)", width=600)

    # PHASE 8 LAB
    elif "Phase 8:" in phase_choice:
        st.subheader("Phase 8: Named Entity Recognition")
        st.markdown("**Objective:** Extract Restaurant names, food dishes, locations, and dates.")
        
        ner_in = st.text_area("Input sentence:", value="Tasty Bites in Pune serves delicious pizza on Sunday. Chef Sanjeev visited yesterday.")
        if st.button("Extract Entities"):
            res8 = run_experiment_8(ner_in)
            st.markdown(res8["annotated_html"], unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(res8["entities"]), use_container_width=True)

    # PHASE 9 LAB
    elif "Phase 9:" in phase_choice:
        st.subheader("Phase 9: Text Similarity & Recommendations")
        st.markdown("**Objective:** Compute TF-IDF Cosine Similarity against all 35 catalog restaurants.")
        
        sim_q = st.text_input("Search query:", value="spicy mutton biryani with raita")
        if st.button("Search Recommendations"):
            res9 = run_experiment_9(sim_q, top_k=5)
            st.dataframe(pd.DataFrame(res9["restaurant_matches"])[["rank", "restaurant_name", "location", "cuisine", "similarity_percentage", "rating"]], use_container_width=True)

    # PHASE 10 LAB
    elif "Phase 10:" in phase_choice:
        st.subheader("Phase 10: Deep Learning WSD (Bi-LSTM)")
        st.markdown("**Objective:** Resolve ambiguous culinary polysemy using PyTorch Bidirectional LSTM neural network.")
        
        wsd_sent = st.text_input("Context sentence:", value="The main course was delicious and hot.")
        target_w = st.selectbox("Ambiguous target word:", ["course", "bill", "order"])
        if st.button("Disambiguate Sense"):
            res10 = run_experiment_10(wsd_sent, target_word=target_w)
            st.success(f"Predicted Sense: **{res10['predicted_sense']}** ({res10['confidence']}% confidence)")
            st.info(f"Meaning: {res10['sense_meaning']}")
            st.json(res10["probabilities"])


# ==============================================================================
# MODE 4: RESTAURANT CATALOG & DATASETS
# ==============================================================================
elif app_mode == "📊 Restaurant Catalog & Datasets":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📊 Restaurant Catalog & Project Datasets</div>
        <div class="hero-subtitle">Explore the curated culinary catalog of 35 restaurants, 70 labeled customer reviews, and neural disambiguation datasets.</div>
    </div>
    """, unsafe_allow_html=True)

    d_tab1, d_tab2, d_tab3 = st.tabs(["🍽️ Restaurant Catalog (35 Venues)", "⭐ Customer Reviews (70 Labeled)", "🧠 WSD Training Contexts (60 Sentences)"])

    with d_tab1:
        st.dataframe(df_restaurants, use_container_width=True)

    with d_tab2:
        st.dataframe(df_reviews, use_container_width=True)
        st.bar_chart(df_reviews['sentiment'].value_counts())

    with d_tab3:
        wsd_path = os.path.join(DATA_DIR, "wsd_dataset.csv")
        if os.path.exists(wsd_path):
            df_wsd = pd.read_csv(wsd_path)
            st.dataframe(df_wsd, use_container_width=True)


# ==============================================================================
# MODE 5: FORMAL COLLEGE PROJECT REPORT
# ==============================================================================
elif app_mode == "📑 Formal Project Report (24 Sections)":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📑 Formal College Project Report (24 Sections)</div>
        <div class="hero-subtitle">Comprehensive technical report adhering to academic engineering curriculum guidelines.</div>
    </div>
    """, unsafe_allow_html=True)

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
