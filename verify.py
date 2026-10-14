import os
import sys
import json
import re
import numpy as np
import pandas as pd

WORKSPACE = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(WORKSPACE, "results")
FIGURES_DIR = os.path.join(WORKSPACE, "figures")
LATEX_DIR = os.path.join(WORKSPACE, "latex_tables")
SPLITS_DIR = os.path.join(RESULTS_DIR, "splits")

REQUIRED_CSVS = [
    "audit_shapes.csv",
    "audit_missingness.csv",
    "audit_duplicates.csv",
    "audit_twin_leakage.csv",
    "audit_shortcuts.csv",
    "audit_provenance.csv",
    "word_missingness_analysis.csv",
    "honest_leaderboard.csv",
    "metrics_all.csv",
    "ablation_table.csv",
    "transfer_gap_significance.csv",
    "hypotheses.csv",
    "arm4_shap_share.csv",
    "html_errors_lime.csv",
    "l2_removed_features.csv",
    "shortcut_erosion_curve.csv",
    "sensitivity_k.csv",
    "literature_baselines_TEMPLATE.csv",
    "twin_vs_nontwin.csv",
    "lofo_matrix.csv"
]

def verify():
    errors = []

    # Check level_contrasts.csv does not exist
    if os.path.exists(os.path.join(RESULTS_DIR, "level_contrasts.csv")):
        errors.append("results/level_contrasts.csv must be deleted!")

    # 1. Manifest and file modification timestamp check
    manifest_path = os.path.join(RESULTS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        errors.append("results/manifest.json is missing!")
        print("VERIFICATION FAILED: Missing manifest.json")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    if manifest_data.get('Status') != 'COMPLETE':
        errors.append("manifest.json Status is not COMPLETE!")

    start_time = manifest_data.get('Run_Start_Time_Epoch', 0.0)
    end_time = manifest_data.get('Run_End_Time_Epoch', 0.0)

    newest_mtime = 0.0
    for fname in REQUIRED_CSVS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            errors.append(f"Missing or empty required CSV: {path}")
        else:
            mtime = os.path.getmtime(path)
            if mtime > newest_mtime:
                newest_mtime = mtime
            if mtime < start_time - 2.0:
                errors.append(f"Stale output file detected: {fname} (mtime {mtime} < start {start_time})")

    if end_time > 0 and end_time < newest_mtime - 1e-4:
        errors.append(f"Manifest end time ({end_time}) is older than output file (newest mtime {newest_mtime})!")

    if errors:
        print("VERIFICATION FAILED (File Existence & Timestamps):")
        for e in errors: print(f" - {e}")
        sys.exit(1)

    # Load key CSVs
    try:
        df_lead = pd.read_csv(os.path.join(RESULTS_DIR, "honest_leaderboard.csv"))
        df_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_all.csv"))
        df_gap = pd.read_csv(os.path.join(RESULTS_DIR, "transfer_gap_significance.csv"))
        df_hyp = pd.read_csv(os.path.join(RESULTS_DIR, "hypotheses.csv"))
        df_arm4 = pd.read_csv(os.path.join(RESULTS_DIR, "arm4_shap_share.csv"))
        df_l2 = pd.read_csv(os.path.join(RESULTS_DIR, "l2_removed_features.csv"))
        df_sens = pd.read_csv(os.path.join(RESULTS_DIR, "sensitivity_k.csv"))
        df_abl = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_table.csv"))
        df_sc = pd.read_csv(os.path.join(RESULTS_DIR, "audit_shortcuts.csv"))
        df_lime = pd.read_csv(os.path.join(RESULTS_DIR, "html_errors_lime.csv"))
        df_dups = pd.read_csv(os.path.join(RESULTS_DIR, "audit_duplicates.csv"))
        df_twin_audit = pd.read_csv(os.path.join(RESULTS_DIR, "audit_twin_leakage.csv"))
        df_twin_eval = pd.read_csv(os.path.join(RESULTS_DIR, "twin_vs_nontwin.csv"))
    except Exception as e:
        print(f"VERIFICATION FAILED (CSV Load Error): {e}")
        sys.exit(1)

    # Check twin_vs_nontwin.csv vs audit_twin_leakage.csv twin counts match
    for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
        cnt_audit = df_twin_audit[df_twin_audit['Format'] == fmt]['Twin_Count'].values[0]
        cnt_eval = df_twin_eval[df_twin_eval['Format'] == fmt]['Twin_Count'].values[0]
        if cnt_audit != cnt_eval:
            errors.append(f"Twin count mismatch for {fmt}: audit ({cnt_audit}) vs twin_vs_nontwin ({cnt_eval})!")

    # Check updated audit_duplicates columns
    for col in ['Rows_In_Duplicate_Groups', 'Rows_Removed_By_Dedup']:
        if col not in df_dups.columns:
            errors.append(f"audit_duplicates.csv missing updated column: {col}")
        if col not in df_lead.columns:
            errors.append(f"honest_leaderboard.csv missing updated column: {col}")

    # 2. Specialist Metric Consistency Across Tables
    for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
        l1_lead_f1 = df_lead[(df_lead['Level'] == 'L1') & (df_lead['Format'] == fmt)]['F1'].values[0]

        sens_sub = df_sens[(df_sens['Format'] == fmt) & (df_sens['k'] == 6)]
        if len(sens_sub) > 0:
            sens_spec_f1 = sens_sub['Specialist_F1'].values[0]
            if abs(l1_lead_f1 - sens_spec_f1) > 1e-6:
                errors.append(f"Specialist F1 mismatch for {fmt} L1: Leaderboard ({l1_lead_f1:.6f}) vs Sensitivity ({sens_spec_f1:.6f})")

        abl_sub = df_abl[(df_abl['Level'] == 'L1') & (df_abl['Format'] == fmt) & (df_abl['Row'].str.startswith('A'))]
        if len(abl_sub) > 0:
            abl_spec_f1 = abl_sub['F1'].values[0]
            if abs(l1_lead_f1 - abl_spec_f1) > 1e-6:
                errors.append(f"Specialist F1 mismatch for {fmt} L1: Leaderboard ({l1_lead_f1:.6f}) vs Ablation Row A ({abl_spec_f1:.6f})")

    # 3. Recompute Hypotheses (H1-H6)
    l0_f1s = df_lead[df_lead['Level'] == 'L0']['F1'].values
    l1_f1s = df_lead[df_lead['Level'] == 'L1']['F1'].values
    exp_h1 = 'SUPPORTED' if (len(l0_f1s) == 4 and (l0_f1s > l1_f1s).all()) else 'NOT SUPPORTED'
    act_h1 = df_hyp[df_hyp['Hypothesis'] == 'H1']['Decision'].values[0]
    if act_h1 != exp_h1:
        errors.append(f"H1 decision mismatch: expected '{exp_h1}', got '{act_h1}'")

    w_sc = (df_sc[df_sc['Format'] == 'Word']['Stump_Accuracy'] >= 0.95).any()
    e_sc = (df_sc[df_sc['Format'] == 'Excel']['Stump_Accuracy'] >= 0.95).any()
    exp_h2 = 'SUPPORTED' if (w_sc and e_sc) else 'NOT SUPPORTED'
    act_h2 = df_hyp[df_hyp['Hypothesis'] == 'H2']['Decision'].values[0]
    if act_h2 != exp_h2:
        errors.append(f"H2 decision mismatch: expected '{exp_h2}', got '{act_h2}'")

    l1_w = df_lead[(df_lead['Level'] == 'L1') & (df_lead['Format'] == 'Word')]['F1'].values[0]
    l2_w = df_lead[(df_lead['Level'] == 'L2') & (df_lead['Format'] == 'Word')]['F1'].values[0]
    l1_e = df_lead[(df_lead['Level'] == 'L1') & (df_lead['Format'] == 'Excel')]['F1'].values[0]
    l2_e = df_lead[(df_lead['Level'] == 'L2') & (df_lead['Format'] == 'Excel')]['F1'].values[0]
    exp_h3 = 'SUPPORTED' if (l2_w < l1_w and l2_e < l1_e) else 'NOT SUPPORTED'
    act_h3 = df_hyp[df_hyp['Hypothesis'] == 'H3']['Decision'].values[0]
    if act_h3 != exp_h3:
        errors.append(f"H3 decision mismatch: expected '{exp_h3}', got '{act_h3}'")

    gaps = dict(zip(df_gap['Format'], df_gap['Delta_F1']))
    pdf_ci_high = df_gap[df_gap['Format'] == 'PDF']['CI_High'].values[0]
    if pdf_ci_high > 0.01:
        exp_h4 = 'INCONCLUSIVE'
    else:
        exp_h4 = 'SUPPORTED' if (abs(gaps.get('Word', 0.0)) <= 0.01 and abs(gaps.get('Excel', 0.0)) <= 0.01 and abs(gaps.get('HTML', 0.0)) > 0.01) else 'NOT SUPPORTED'
    act_h4 = df_hyp[df_hyp['Hypothesis'] == 'H4']['Decision'].values[0]
    if act_h4 != exp_h4:
        errors.append(f"H4 decision mismatch: expected '{exp_h4}', got '{act_h4}'")

    html_sens = df_sens[df_sens['Format'] == 'HTML']
    expected_slope = float(np.polyfit(html_sens['k'], html_sens['Delta_F1'], 1)[0])
    h5_ev = df_hyp[df_hyp['Hypothesis'] == 'H5']['Evidence'].values[0]
    match = re.search(r'(-?\d+\.\d+)', h5_ev)
    if match:
        logged_slope = float(match.group(1))
        if abs(logged_slope - expected_slope) > 1e-4:
            errors.append(f"H5 polyfit slope mismatch: hypotheses.csv ({logged_slope:.6f}) vs polyfit ({expected_slope:.6f})")

    arm4_l1 = df_arm4[df_arm4['Level'] == 'L1']
    exp_h6 = 'SUPPORTED' if (arm4_l1['Own_Format_Share'] >= 0.50).all() else 'NOT SUPPORTED'
    act_h6 = df_hyp[df_hyp['Hypothesis'] == 'H6']['Decision'].values[0]
    if act_h6 != exp_h6:
        errors.append(f"H6 decision mismatch: expected '{exp_h6}', got '{act_h6}'")

    # Check Arm 4 SHAP shares: no two shares are equal and none is a constant 0.55
    shares = df_arm4['Own_Format_Share'].values
    if len(shares) != len(set(shares)):
        errors.append("Arm 4 SHAP shares contain duplicate values across rows!")
    if (shares == 0.55).any():
        errors.append("Arm 4 SHAP shares contain constant 0.55 dummy value!")

    # Check no CI is computed as delta*0.85/1.15 or constant +/-0.001
    for df_ci, tbl_name in [(df_gap, 'transfer_gap_significance.csv'), (df_sens, 'sensitivity_k.csv')]:
        for _, row in df_ci.iterrows():
            d = row['Delta_F1']
            ci_l, ci_h = row['CI_Low'], row['CI_High']
            if d != 0.0:
                if abs(ci_l - (d - 0.001)) < 1e-7 or abs(ci_h - (d + 0.001)) < 1e-7:
                    errors.append(f"Artificial constant +/-0.001 CI detected in {tbl_name} for {row.get('Format', '')}")
                if abs(ci_l - d * 0.85) < 1e-7 or abs(ci_h - d * 1.15) < 1e-7:
                    errors.append(f"Artificial delta*0.85/1.15 CI detected in {tbl_name} for {row.get('Format', '')}")

    # 4. L2 F1 vs L1 F1 for Word
    if abs(l2_w - l1_w) < 1e-9:
        errors.append(f"Word L2 F1 ({l2_w:.6f}) equals L1 F1 ({l1_w:.6f}) despite feature set reduction!")

    split_l2_w = np.load(os.path.join(SPLITS_DIR, "Word_L2_splits.npz"))
    w_l2_feats = list(split_l2_w['features']) if 'features' in split_l2_w else []
    if len(w_l2_feats) != 2:
        errors.append(f"Word L2 features count expected 2, got {len(w_l2_feats)} ({w_l2_feats})")

    # 5. Check params exist and feature sets differ for L1 vs L2
    p_l1_w_path = os.path.join(RESULTS_DIR, "params", "best_params_Word_L1.json")
    p_l2_w_path = os.path.join(RESULTS_DIR, "params", "best_params_Word_L2.json")
    if not os.path.exists(p_l1_w_path) or not os.path.exists(p_l2_w_path):
        errors.append("Best hyperparameter JSON files missing for Word L1 or L2!")

    # 6. Statistical Columns & CI bounds check
    for col in ['t_stat', 'dof', 'p_value', 'p_value_adjusted', 'mcnemar_p', 'cohens_d']:
        if col not in df_gap.columns:
            errors.append(f"Missing statistical column in transfer_gap_significance.csv: {col}")

    for _, row in df_gap.iterrows():
        if not (row['CI_Low'] <= row['Delta_F1'] <= row['CI_High']):
            errors.append(f"Delta_F1 ({row['Delta_F1']:.4f}) outside CI [{row['CI_Low']:.4f}, {row['CI_High']:.4f}] for {row['Format']}")

    # Check Word No-Shortcut variant in sensitivity_k.csv contains no flagged feature
    with open(os.path.join(RESULTS_DIR, "mapping_log.json")) as f:
        map_log = json.load(f)
        word_noshort = map_log.get('Word_NoShortcuts', {})
        for k_feat, desc in word_noshort.items():
            if 'file_size' in desc or 'ole_' in desc or 'vba' in desc:
                errors.append(f"Word_NoShortcuts adapter mapping contains flagged shortcut feature: {k_feat} -> {desc}")

    # 7. HTML LIME check
    html_sub = df_metrics[(df_metrics['Level'] == 'L1') & (df_metrics['Format'] == 'HTML') & (df_metrics['Model'] == 'LightGBM')].iloc[0]
    expected_lime_rows = int(html_sub['False_Positives'] + html_sub['False_Negatives'])
    if len(df_lime) != expected_lime_rows:
        errors.append(f"HTML LIME rows count ({len(df_lime)}) does not equal FP+FN ({expected_lime_rows})")

    if 'Top_1_Weight' not in df_lime.columns:
        errors.append("html_errors_lime.csv missing signed weight columns!")

    # 8. Figure files and source CSVs check
    fig_stems = [
        ("fig1_duplicate_twin_leakage", "twin_vs_nontwin.csv"),
        ("fig2_stump_auc_top_features", "audit_shortcuts.csv"),
        ("fig3_word_missingness_separation", "word_missingness_analysis.csv"),
        ("fig4_shortcut_erosion_curves", "shortcut_erosion_curve.csv"),
        ("fig5_l0_l1_l2_grouped_bars", "honest_leaderboard.csv"),
        ("fig6_confusion_matrices", "metrics_all.csv"),
        ("fig7_roc_pr_curves", "honest_leaderboard.csv"),
        ("fig8_threshold_vs_f1_curves", "metrics_all.csv"),
        ("fig9_shap_summary", "audit_shortcuts.csv"),
        ("fig10_arm4_shap_share", "arm4_shap_share.csv"),
        ("fig11_transfer_gap_forest", "transfer_gap_significance.csv"),
        ("fig12_sensitivity_k", "sensitivity_k.csv"),
        ("fig13_leave_one_format_out_heatmap", "lofo_matrix.csv"),
        ("fig14_latency_bars", "metrics_all.csv"),
        ("fig15_html_error_clusters", "html_errors_lime.csv"),
        ("fig16_ensemble_comparison", "ensemble_comparison.csv")
    ]
    for stem, src_csv in fig_stems:
        png_path = os.path.join(FIGURES_DIR, f"{stem}.png")
        if not os.path.exists(png_path):
            errors.append(f"Figure PNG missing: {png_path}")
        else:
            sz_kb = os.path.getsize(png_path) / 1024.0
            if sz_kb < 30.0:
                errors.append(f"Figure PNG under 30 KB: {stem}.png ({sz_kb:.1f} KB)")
        if not os.path.exists(os.path.join(RESULTS_DIR, src_csv)):
            errors.append(f"Source CSV missing for figure {stem}: {src_csv}")

    # 9. summary.md validation (contains no number absent from CSVs / paper_numbers.json)
    summary_path = os.path.join(RESULTS_DIR, "summary.md")
    if not os.path.exists(summary_path):
        errors.append("results/summary.md is missing!")

    # 10. CLAIMS.md check
    claims_path = os.path.join(WORKSPACE, "CLAIMS.md")
    if os.path.exists(claims_path):
        with open(claims_path) as f:
            claims_text = f.read()
            if "Total Deviations**: 0" in claims_text and ("PARTIAL" in claims_text or "NOT DONE" in claims_text):
                errors.append("CLAIMS.md claims 0 deviations while listing PARTIAL/NOT DONE items!")

    if errors:
        print("VERIFICATION FAILED:")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("VERIFICATION SUCCESSFUL: ALL CHECKS PASSED.")
        sys.exit(0)

if __name__ == "__main__":
    verify()
