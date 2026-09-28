"""
Automated Test Suite for Restaurant NLP Analysis System
Tests all 10 individual experiments and the full unified pipeline.
"""

import unittest
from modules.sentiment import run_experiment_1
from modules.preprocessing import run_experiment_2, run_experiment_3
from modules.morphology import run_experiment_4
from modules.ngram import run_experiment_5
from modules.pos_tagging import run_experiment_6
from modules.chunking import run_experiment_7
from modules.ner import run_experiment_8
from modules.similarity import run_experiment_9
from modules.wsd import run_experiment_10
from modules.pipeline import run_complete_nlp_pipeline


class TestRestaurantNLPSystem(unittest.TestCase):

    def test_experiment_1_sentiment(self):
        pos_res = run_experiment_1("The food was delicious and the service was excellent.")
        self.assertEqual(pos_res["predicted_sentiment"], "Positive")
        self.assertGreater(pos_res["confidence"], 0.40)

        neg_res = run_experiment_1("The pizza was cold and greasy, worst dining experience ever.")
        self.assertEqual(neg_res["predicted_sentiment"], "Negative")
        self.assertGreater(neg_res["confidence"], 0.40)

    def test_experiment_2_tokenization_script(self):
        text = "Find delicious Italian food in Pune!"
        res = run_experiment_2(text)
        self.assertTrue(res["script_validation"]["is_valid"])
        self.assertIn("delicious", res["filtered_tokens"])
        self.assertNotIn("!", res["filtered_tokens"])

        # Non-English / Unsupported script test
        foreign = "हा एक सुंदर रेस्टॉरंट आहे"
        foreign_res = run_experiment_2(foreign)
        self.assertFalse(foreign_res["script_validation"]["is_valid"])

    def test_experiment_3_preprocessing(self):
        tokens = ["I", "am", "looking", "for", "restaurants", "serving", "delicious", "foods"]
        res = run_experiment_3(tokens)
        self.assertNotIn("am", res["tokens_without_stopwords"])
        self.assertIn("restaur", res["stemmed_tokens"])
        self.assertIn("food", res["lemmatized_tokens"])

    def test_experiment_4_morphology(self):
        tokens = ["serve", "cook", "recommend"]
        res = run_experiment_4(tokens)
        self.assertGreaterEqual(len(res["word_analyses"]), 3)
        serve_info = next(w for w in res["word_analyses"] if w["root_word"] == "serve")
        self.assertIn("serves", serve_info["generated_forms"])
        self.assertIn("serving", serve_info["generated_forms"])

    def test_experiment_5_ngram(self):
        text = "delicious Italian food"
        res = run_experiment_5(text)
        unigrams = [u["ngram"] for u in res["unigrams"]]
        bigrams = [b["ngram"] for b in res["bigrams"]]
        self.assertIn("<s>", unigrams)
        self.assertIn("</s>", unigrams)
        self.assertTrue(any("delicious italian" in b for b in bigrams))

    def test_experiment_6_pos_tagging(self):
        tokens = ["They", "serve", "delicious", "Italian", "restaurants"]
        res = run_experiment_6(tokens)
        self.assertEqual(res["total_tokens"], 5)
        tags = [t["tag"] for t in res["detailed_tags"]]
        self.assertTrue(any(t.startswith("VB") for t in tags))
        self.assertTrue(any(t.startswith("JJ") for t in tags))

    def test_experiment_7_chunking(self):
        tagged = [("delicious", "JJ"), ("pizza", "NN"), ("in", "IN"), ("Pune", "NNP")]
        res = run_experiment_7(tagged)
        self.assertGreaterEqual(res["total_chunks"], 1)
        chunks = [c["chunk_text"] for c in res["rule_based_chunks"]]
        self.assertTrue(any("delicious pizza" in c for c in chunks))
        # Verify 60%, 70%, 80% benchmark presence
        size_comp = res["training_size_comparison"]
        self.assertEqual(len(size_comp), 3)

    def test_experiment_8_ner(self):
        text = "Tasty Bites in Pune serves excellent pizza on Sunday."
        res = run_experiment_8(text)
        entities = {e["entity"].lower(): e["type"] for e in res["entities"]}
        self.assertIn("tasty bites", entities)
        self.assertEqual(entities["tasty bites"], "RESTAURANT")
        self.assertIn("pune", entities)
        self.assertEqual(entities["pune"], "LOCATION")
        self.assertIn("pizza", entities)
        self.assertEqual(entities["pizza"], "FOOD")

    def test_experiment_9_similarity(self):
        query = "delicious spicy Indian food"
        res = run_experiment_9(query, top_k=3)
        self.assertEqual(len(res["restaurant_matches"]), 3)
        self.assertGreater(res["highest_similarity_score"], 0.0)

    def test_experiment_10_wsd(self):
        food_sent = "The restaurant served an excellent main course."
        food_res = run_experiment_10(food_sent, target_word="course")
        self.assertEqual(food_res["predicted_sense"], "FOOD")

        edu_sent = "I enrolled in an online computer science course."
        edu_res = run_experiment_10(edu_sent, target_word="course")
        self.assertEqual(edu_res["predicted_sense"], "EDUCATION")

    def test_unified_pipeline(self):
        test_input = "Tasty Bites in Pune serves delicious pizza and the main course was amazing!"
        result = run_complete_nlp_pipeline(test_input)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["sentiment"], "Positive")
        self.assertTrue(result["summary"]["script_valid"])
        self.assertIn("experiment_1_sentiment", result)
        self.assertIn("experiment_10_wsd", result)


if __name__ == "__main__":
    unittest.main()
