"""
Validation Tests for Universal Feature Adapters.
"""

import os
import unittest
import pandas as pd
from adapters.universal_adapter import UniversalFeatureAdapter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_DIR = os.path.join(REPO_ROOT, "data", "sample")


class TestUniversalFeatureAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = UniversalFeatureAdapter()

    def test_adapter_all_formats_budgets(self):
        formats = [
            ("Word", "Word_Sample.csv"),
            ("Excel", "Excel_Sample.csv"),
            ("PDF", "PDF_Sample.csv"),
            ("HTML", "HTML_Sample.csv"),
        ]
        for fmt, fname in formats:
            path = os.path.join(SAMPLE_DIR, fname)
            df = pd.read_csv(path)
            X = df.drop(columns=["label", "file_path"], errors="ignore")

            for k in [4, 6, 8, 10]:
                uni = self.adapter.extract_features(fmt, X, k=k)
                self.assertEqual(uni.shape[0], len(df), f"Row count mismatch for {fmt} k={k}")
                self.assertEqual(uni.shape[1], k, f"Column budget mismatch for {fmt} k={k}")
                self.assertFalse(uni.isnull().all().any(), f"Completely NaN column in {fmt} k={k}")

    def test_shortcut_free_word_adapter(self):
        path = os.path.join(SAMPLE_DIR, "Word_Sample.csv")
        df = pd.read_csv(path)
        X = df.drop(columns=["label", "file_path"], errors="ignore")

        uni = self.adapter.extract_features("Word (No Shortcuts)", X, k=6)
        self.assertEqual(uni.shape, (len(df), 6))
        # Slots 1, 3, 4, 6, 8, 9 must be 0.0 constant in shortcut-free Word adapter
        self.assertTrue((uni["1_content_entropy"] == 0.0).all())
        self.assertTrue((uni["3_external_reference_count"] == 0.0).all())
        self.assertTrue((uni["4_log_file_size"] == 0.0).all())
        self.assertTrue((uni["6_obfuscation_indicator"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
