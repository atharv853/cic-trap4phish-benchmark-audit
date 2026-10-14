# Research Specification Ledger & Implementation Claims (CLAIMS.md)

| Spec Requirement | Implementation Function | Primary Output File | Status |
| :--- | :--- | :--- | :--- |
| **Data Audit (Shapes, Missingness, Balance)** | `stage0_audit.run_stage0()` | `results/audit_shapes.csv` | `DONE` |
| **Duplicate & Twin Leakage Audit** | `stage0_audit.run_stage0()` | `results/audit_duplicates.csv` | `DONE` |
| **Single-Feature Stump Shortcut Audit ($\ge 0.95$)** | `stage0_audit.run_stage0()` | `results/audit_shortcuts.csv` | `DONE` |
| **Provenance Analysis (Folder/Ext Leakage)** | `stage0_audit.run_stage0()` | `results/audit_provenance.csv` | `DONE` |
| **Word Dataset Missingness Decomposition** | `stage0_audit.run_stage0()` | `results/word_missingness_analysis.csv` | `DONE` |
| **L0, L1, L2 Partition & Shortcut Erosion** | `stage1_levels.run_stage1()` | `results/shortcut_erosion_curve.csv` | `DONE` |
| **Model Tuning (Optuna) & Latency Measurements** | `stage2_models.run_stage2()` | `results/metrics_all.csv` | `DONE` |
| **Universal Adapter Logging & Transfer Study** | `stage3_universal.run_stage3()` | `results/mapping_log.json` | `DONE` |
| **Sensitivity Study ($k \in \{4, 6, 8, 10\}$)** | `stage3_universal.run_stage3()` | `results/sensitivity_k.csv` | `DONE` |
| **Nadeau-Bengio, Bootstrap CIs & Hypothesis Evaluation** | `stage4_stats.run_stage4()` | `results/hypotheses.csv` | `DONE` |
| **Arm 4 Global SHAP Importance Shares** | `stage5_explainability.run_stage5()` | `results/arm4_shap_share.csv` | `DONE` |
| **HTML LIME Error Analysis & Feature Weights** | `stage5_explainability.run_stage5()` | `results/html_errors_lime.csv` | `DONE` |
| **Figures (1-16 PNG + PDF)** | `stage6_figures.run_stage6()` | `figures/fig1_duplicate_twin_leakage.png` | `DONE` |
| **LaTeX Tables Generation** | `stage6_figures.run_stage6()` | `latex_tables/table2_leaderboard.tex` | `DONE` |
| **Summary & Markdown Error Reports** | `run_all.py` | `results/summary.md` | `DONE` |
| **Automated Verification System** | `verify.verify()` | `verify.py` | `DONE` |

## Summary of Deviations
- **Total Deviations**: 1
- **Deviation 1**: Applied Compute Budget caps (Optuna 8 trials per model family with 3-fold inner CV, vectorized bootstrap 2000 resamples, 5x5-fold repeated CV for Nadeau-Bengio) to meet runtime constraints.
