"""
Fast Smoke Test: Simulates Data Loading and Decision Stump Evaluation.
"""

import os
import unittest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from adapters.universal_adapter import UniversalFeatureAdapter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_DIR = os.path.join(REPO_ROOT, "data", "sample")


class TestSmokePipeline(unittest.TestCase):
    def test_smoke_end_to_end(self):
        adapter = UniversalFeatureAdapter()
        formats = [
            ("Word", "Word_Sample.csv"),
            ("Excel", "Excel_Sample.csv"),
            ("PDF", "PDF_Sample.csv"),
            ("HTML", "HTML_Sample.csv"),
        ]

        for fmt, fname in formats:
            path = os.path.join(SAMPLE_DIR, fname)
            df = pd.read_csv(path)
            self.assertIn("label", df.columns, f"Label column missing in {fname}")

            X = df.drop(columns=["label", "file_path"], errors="ignore")
            y = df["label"]

            # Universal feature transformation
            uni_X = adapter.extract_features(fmt, X, k=6)
            self.assertEqual(uni_X.shape, (len(df), 6))

            # Train quick decision tree classifier
            clf = DecisionTreeClassifier(max_depth=2, random_state=42)
            clf.fit(uni_X, y)
            preds = clf.predict(uni_X)
            acc = accuracy_score(y, preds)
            self.assertGreater(acc, 0.50, f"Smoke model accuracy too low for {fmt}: {acc}")


if __name__ == "__main__":
    unittest.main()
