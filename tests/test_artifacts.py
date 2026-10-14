"""
Integrity and Schema Validation Tests for Research Artifacts.
"""

import os
import json
import unittest
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
LATEX_DIR = os.path.join(REPO_ROOT, "latex_tables")


class TestArtifacts(unittest.TestCase):
    TARGET_13_FILES = [
        "honest_leaderboard.csv",
        "audit_provenance.csv",
        "audit_shortcuts.csv",
        "l2_removed_features.csv",
        "twin_vs_nontwin.csv",
        "sensitivity_k.csv",
        "transfer_gap_significance.csv",
        "hypotheses.csv",
        "arm4_shap_share.csv",
        "lofo_matrix.csv",
        "ensemble_comparison.csv",
        "mapping_log.json",
        "html_error_analysis.md",
    ]

    def test_all_13_target_files_exist_and_nonempty(self):
        for fname in self.TARGET_13_FILES:
            path = os.path.join(RESULTS_DIR, fname)
            self.assertTrue(os.path.exists(path), f"Missing artifact: {fname}")
            self.assertGreater(os.path.getsize(path), 0, f"Empty artifact: {fname}")

    def test_csv_artifacts_parse_properly(self):
        csv_files = [f for f in self.TARGET_13_FILES if f.endswith(".csv")]
        for fname in csv_files:
            path = os.path.join(RESULTS_DIR, fname)
            df = pd.read_csv(path)
            self.assertFalse(df.empty, f"CSV is empty: {fname}")
            self.assertGreater(len(df.columns), 0, f"CSV has no columns: {fname}")

    def test_mapping_log_json(self):
        path = os.path.join(RESULTS_DIR, "mapping_log.json")
        with open(path, "r") as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)
        for fmt in ["Word", "Excel", "PDF", "HTML"]:
            self.assertIn(fmt, data, f"Missing format {fmt} in mapping_log.json")

    def test_html_error_analysis_markdown(self):
        path = os.path.join(RESULTS_DIR, "html_error_analysis.md")
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("HTML Specialist Model Error Analysis", content)
        self.assertIn("LIME", content)
        self.assertIn("SHAP", content)

    def test_figures_exist_png_and_pdf(self):
        fig_stems = [
            "fig1_duplicate_twin_leakage",
            "fig2_stump_auc_top_features",
            "fig3_word_missingness_separation",
            "fig4_shortcut_erosion_curves",
            "fig5_l0_l1_l2_grouped_bars",
            "fig6_confusion_matrices",
            "fig7_roc_pr_curves",
            "fig8_threshold_vs_f1_curves",
            "fig9_shap_summary",
            "fig10_arm4_shap_share",
            "fig11_transfer_gap_forest",
            "fig12_sensitivity_k",
            "fig13_leave_one_format_out_heatmap",
            "fig14_latency_bars",
            "fig15_html_error_clusters",
            "fig16_ensemble_comparison",
        ]
        for stem in fig_stems:
            png_p = os.path.join(FIGURES_DIR, f"{stem}.png")
            pdf_p = os.path.join(FIGURES_DIR, f"{stem}.pdf")
            self.assertTrue(os.path.exists(png_p), f"Missing figure PNG: {stem}.png")
            self.assertTrue(os.path.exists(pdf_p), f"Missing figure PDF: {stem}.pdf")
            self.assertGreaterEqual(os.path.getsize(png_p), 30 * 1024, f"PNG under 30KB: {stem}.png")

    def test_latex_tables_exist(self):
        tables = [
            "hypotheses.tex",
            "lofo_matrix.tex",
            "table2_leaderboard.tex",
            "table3_ablation.tex",
            "table4_transfer.tex",
        ]
        for tbl in tables:
            path = os.path.join(LATEX_DIR, tbl)
            self.assertTrue(os.path.exists(path), f"Missing LaTeX table: {tbl}")
            self.assertGreater(os.path.getsize(path), 0, f"Empty LaTeX table: {tbl}")


if __name__ == "__main__":
    unittest.main()
