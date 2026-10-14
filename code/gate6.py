import os
import sys
import pandas as pd
from stage6_figures import run_stage6, FIGURES_DIR, LATEX_DIR

def run_gate6():
    run_stage6()
    print("\n================ GATE 6 OUTPUT ================")

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
        "fig16_ensemble_comparison"
    ]

    print("Figure File Sizes Check (Must all be >= 30 KB):")
    for stem in fig_stems:
        png_path = os.path.join(FIGURES_DIR, f"{stem}.png")
        assert os.path.exists(png_path), f"Figure PNG missing: {png_path}"
        sz_kb = os.path.getsize(png_path) / 1024.0
        assert sz_kb >= 30.0, f"Figure {stem}.png is under 30 KB ({sz_kb:.2f} KB)!"
        print(f"  {stem:<36}.png | Size: {sz_kb:.2f} KB | Status: PASS")

    print("\nLaTeX Tables Generated:")
    for tex in ['table2_leaderboard.tex', 'table3_ablation.tex', 'table4_transfer.tex', 'hypotheses.tex']:
        path = os.path.join(LATEX_DIR, tex)
        assert os.path.exists(path) and os.path.getsize(path) > 0, f"LaTeX table missing/empty: {path}"
        print(f"  {tex:<28} | Size: {os.path.getsize(path)} bytes")

    print("\n================ GATE 6 PASSED ================\n")

if __name__ == "__main__":
    run_gate6()
