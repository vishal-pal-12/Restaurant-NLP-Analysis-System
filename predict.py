"""
Interactive VS Code Prediction Console for Restaurant NLP System.
Allows users to enter any restaurant name, food dish, customer review, or query
from the dataset or live on-time, and predicts all NLP attributes in real-time.
"""

import os
import sys
import io
import pandas as pd

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure modules can be imported
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from modules.pipeline import run_complete_nlp_pipeline
from modules.sentiment import run_experiment_1
from modules.preprocessing import run_experiment_2, run_experiment_3
from modules.morphology import run_experiment_4
from modules.ngram import run_experiment_5
from modules.pos_tagging import run_experiment_6
from modules.chunking import run_experiment_7
from modules.ner import run_experiment_8
from modules.similarity import run_experiment_9
from modules.wsd import run_experiment_10

DATA_DIR = os.path.join(CURRENT_DIR, "data")
REST_FILE = os.path.join(DATA_DIR, "restaurant_data.csv")
REV_FILE = os.path.join(DATA_DIR, "restaurant_reviews.csv")

def load_datasets():
    df_rest = pd.read_csv(REST_FILE) if os.path.exists(REST_FILE) else pd.DataFrame()
    df_rev = pd.read_csv(REV_FILE) if os.path.exists(REV_FILE) else pd.DataFrame()
    return df_rest, df_rev

def display_banner():
    print("\n" + "=" * 76)
    print(" [RESTAURANT NLP PREDICTION CONSOLE] - Real-Time VS Code Interactive Mode")
    print("=" * 76)
    print(" Enter any text from dataset or live input:")
    print("   * Restaurant Name (e.g., 'Tasty Bites', 'Little Italy', 'Bukhara')")
    print("   * Food / Dish     (e.g., 'biryani', 'pizza', 'butter chicken')")
    print("   * Customer Review (e.g., 'The food was cold but service was great')")
    print("   * Food Query      (e.g., 'Find authentic South Indian food in Pune')")
    print("\n Commands:")
    print("   * 'list'   -> Show all 35 restaurants in the dataset")
    print("   * 'random' -> Pick a random review or restaurant from dataset")
    print("   * 'exit'   -> Quit console")
    print("=" * 76 + "\n")

def list_catalog(df_rest):
    if df_rest.empty:
        print("[!] Restaurant catalog not found.")
        return
    print("\n--- AVAILABLE RESTAURANTS IN DATASET ---")
    for idx, row in df_rest.iterrows():
        print(f"[{idx+1:2d}] {row['restaurant_name']:<24} | {row['location']:<22} | Cuisine: {row['cuisine']}")
    print("-" * 76)
    print("Tip: Type any of these restaurant names or dishes above to run prediction!\n")

def find_restaurant_match(user_input, df_rest):
    if df_rest.empty:
        return None
    query_lower = user_input.strip().lower()
    for _, row in df_rest.iterrows():
        r_name = str(row['restaurant_name']).lower()
        if query_lower == r_name or query_lower in r_name or r_name in query_lower:
            return row
    return None

def analyze_and_predict(user_input: str, df_rest, df_rev):
    user_input = user_input.strip()
    if not user_input:
        return

    print("\n" + "-" * 76)
    print(f"[*] ANALYZING INPUT: \"{user_input}\"")
    print("-" * 76)

    # 1. Check if user typed a specific restaurant name from dataset
    matched_rest = find_restaurant_match(user_input, df_rest)
    if matched_rest is not None:
        print("\n[+] [DATASET MATCH] Found Registered Restaurant Profile:")
        print(f"    Name:           {matched_rest['restaurant_name']}")
        print(f"    Rating:         {matched_rest['rating']} / 5.0")
        print(f"    Location:       {matched_rest['location']}")
        print(f"    Cuisine:        {matched_rest['cuisine']}")
        print(f"    Popular Dishes: {matched_rest['popular_dishes']}")
        print(f"    Description:    {matched_rest['description']}")

        # Check if we have reviews for this restaurant
        if not df_rev.empty:
            matching_revs = df_rev[df_rev['review'].str.contains(matched_rest['restaurant_name'], case=False, na=False)]
            if not matching_revs.empty:
                print(f"\n    Found {len(matching_revs)} customer review(s) in dataset for {matched_rest['restaurant_name']}:")
                for _, r_row in matching_revs.head(2).iterrows():
                    print(f"      - \"{r_row['review']}\" [Sentiment: {r_row['sentiment']}]")

        print("\n" + "." * 76)
        print("[-] RUNNING FULL 10-MODULE NLP PIPELINE ON PROFILE DESCRIPTION:")
        text_to_process = f"{matched_rest['restaurant_name']} in {matched_rest['location']} serves {matched_rest['popular_dishes']}. {matched_rest['description']}"
    else:
        text_to_process = user_input

    # Run complete unified pipeline
    res = run_complete_nlp_pipeline(text_to_process)
    summary = res["summary"]

    # 1. Sentiment Prediction
    s_label = summary["sentiment"]
    s_conf = summary["sentiment_confidence"]
    print(f"\n[EXP 1] PREDICTED SENTIMENT : {s_label.upper()} ({s_conf} confidence)")

    # 2. Script Validation & Tokens
    s_valid = "VALID (English)" if summary["script_valid"] else "INVALID SCRIPT"
    print(f"[EXP 2] SCRIPT VALIDATION   : {s_valid} | Tokens: {summary['token_count']} (Filtered: {summary['filtered_count']})")

    # 3. Stemming & Lemmatization Root Sample
    exp3 = res["experiment_3_preprocessing"]
    sample_lemmas = [f"{t['original']}->{t['lemmatized']}" for t in exp3["comparison_table"][:4] if not t["is_stopword"]]
    if sample_lemmas:
        print(f"[EXP 3] LEMMATIZED ROOTS    : {', '.join(sample_lemmas)}")

    # 4. Morphological Expansion
    exp4 = res["experiment_4_morphology"]
    exp_query = exp4["query_expansion"]["expanded_query_string"]
    if exp_query and exp_query != text_to_process:
        preview = (exp_query[:75] + '...') if len(exp_query) > 75 else exp_query
        print(f"[EXP 4] MORPHOLOGY EXPANSION: {preview}")

    # 5. N-Gram Predictions
    exp5 = res["experiment_5_ngram"]
    top_bigrams = [f"'{b['ngram']}'" for b in exp5["bigrams"][:3]]
    if top_bigrams:
        print(f"[EXP 5] TOP BIGRAMS (N-GRAM): {', '.join(top_bigrams)}")
    if exp5.get("next_word_predictions"):
        next_w = [p.get("next_word", "") for p in exp5["next_word_predictions"][:3] if p.get("next_word")]
        if next_w:
            print(f"        NEXT-WORD PREDICT   : {', '.join(next_w)}")

    # 6. Part-of-Speech Summary
    exp6 = res["experiment_6_pos"]
    dist = exp6["category_distribution"]
    pos_str = ", ".join([f"{k}: {v}" for k, v in dist.items() if v > 0])
    print(f"[EXP 6] POS DISTRIBUTION    : {pos_str}")

    # 7. Extracted Phrases (Chunking)
    exp7 = res["experiment_7_chunking"]
    chunk_list = [c["chunk_text"] for c in exp7["rule_based_chunks"][:4]]
    if chunk_list:
        formatted_chunks = ", ".join(['"' + str(c) + '"' for c in chunk_list])
        print(f"[EXP 7] EXTRACTED CHUNKS    : {formatted_chunks}")

    # 8. Named Entities
    exp8 = res["experiment_8_ner"]
    ent_list = [f"{e['entity']} ({e['type']})" for e in exp8["entities"]]
    if ent_list:
        print(f"[EXP 8] NAMED ENTITIES (NER): {', '.join(ent_list)}")

    # 9. Top Matching Recommendations (Cosine Similarity)
    exp9 = res["experiment_9_similarity"]
    print(f"[EXP 9] TOP RECOMMENDATIONS :")
    for r in exp9["restaurant_matches"][:3]:
        print(f"        #{r['rank']} {r['restaurant_name']:<22} (Match: {r['similarity_percentage']:>6}) | {r['cuisine']} | {r['location']}")

    # 10. Word Sense Disambiguation
    exp10 = res["experiment_10_wsd"]
    print(f"[EXP 10] WSD NEURAL PREDICT : Target '{exp10['target_word']}' -> Sense: {exp10['predicted_sense']} ({exp10['confidence']}%)")
    print(f"         Context Meaning    : {exp10['sense_meaning']}")

    print("-" * 76 + "\n")

def main():
    df_rest, df_rev = load_datasets()
    display_banner()

    # If argument provided from CLI directly (e.g., python predict.py "Tasty Bites")
    if len(sys.argv) > 1:
        cli_arg = " ".join(sys.argv[1:])
        analyze_and_predict(cli_arg, df_rest, df_rev)
        return

    while True:
        try:
            user_choice = input("Enter Name / Dish / Review / Query (or 'list', 'random', 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Restaurant NLP Console. Goodbye!")
            break

        if not user_choice:
            continue

        cmd_lower = user_choice.lower()
        if cmd_lower in ["exit", "quit", "q"]:
            print("\nExiting Restaurant NLP Console. Goodbye!")
            break
        elif cmd_lower == "list":
            list_catalog(df_rest)
        elif cmd_lower == "random":
            if not df_rest.empty and not df_rev.empty:
                import random
                if random.choice([True, False]):
                    r_sample = df_rest.sample(1).iloc[0]
                    print(f"\n[Random Pick from Catalog]: {r_sample['restaurant_name']}")
                    analyze_and_predict(r_sample['restaurant_name'], df_rest, df_rev)
                else:
                    rev_sample = df_rev.sample(1).iloc[0]
                    print(f"\n[Random Pick from Reviews]: \"{rev_sample['review']}\"")
                    analyze_and_predict(rev_sample['review'], df_rest, df_rev)
            elif not df_rest.empty:
                r_sample = df_rest.sample(1).iloc[0]
                analyze_and_predict(r_sample['restaurant_name'], df_rest, df_rev)
            else:
                analyze_and_predict("Tasty Bites in Pune serves delicious pizza", df_rest, df_rev)
        else:
            analyze_and_predict(user_choice, df_rest, df_rev)

if __name__ == "__main__":
    main()
