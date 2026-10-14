import os
import sys
import json
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from lime import lime_tabular
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.inspection import permutation_importance

WORKSPACE = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
DATA_DIR = os.path.join(WORKSPACE, "data")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
SPLITS_DIR = os.path.join(RESULTS_DIR, "splits")

SEED = 42

def sanitize_columns(columns):
    return [re.sub(r'[^\w]', '_', c) for c in columns]

def run_stage5():
    print("=== EXECUTING STAGE 5: EXPLAINABILITY & HTML LIME ===", flush=True)

    def _resolve_data(fname, sample_name):
        for p in [os.path.join(DATA_DIR, fname), os.path.join(WORKSPACE, fname), os.path.join(DATA_DIR, "sample", sample_name)]:
            if os.path.exists(p): return p
        return os.path.join(DATA_DIR, fname)

    file_paths = {
        'Word': _resolve_data("Word_All_features.csv", "Word_Sample.csv"),
        'Excel': _resolve_data("Excel_All_Features.csv", "Excel_Sample.csv"),
        'PDF': _resolve_data("PDF_All_features.csv", "PDF_Sample.csv"),
        'HTML': _resolve_data("HTML_All_Features.csv", "HTML_Sample.csv")
    }

    raw_dfs = {}
    for fmt, path in file_paths.items():
        df = pd.read_csv(path)
        if 'file_path' in df.columns:
            df = df.drop(columns=['file_path'])
        df.columns = sanitize_columns(df.columns)
        raw_dfs[fmt] = df

    # --- HTML LIME Explanations ---
    html_df = raw_dfs['HTML'].fillna(0)
    X_html = html_df.drop(columns=['label'])
    y_html = html_df['label']

    split_data = np.load(os.path.join(SPLITS_DIR, "HTML_L1_splits.npz"))
    tr_idx = split_data['train_idx']
    te_idx = split_data['test_idx']

    X_tr_h = X_html.iloc[tr_idx].reset_index(drop=True)
    y_tr_h = y_html.iloc[tr_idx].reset_index(drop=True)
    X_te_h = X_html.iloc[te_idx].reset_index(drop=True)
    y_te_h = y_html.iloc[te_idx].reset_index(drop=True)

    with open(os.path.join(RESULTS_DIR, "params", "best_params_HTML_L1.json")) as f:
        best_p = json.load(f)

    clf_html = lgb.LGBMClassifier(**best_p)
    clf_html.fit(X_tr_h, y_tr_h)

    probs_h = clf_html.predict_proba(X_te_h)[:, 1]

    df_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_all.csv"))
    html_sub = df_metrics[(df_metrics['Level'] == 'L1') & (df_metrics['Format'] == 'HTML') & (df_metrics['Model'] == 'LightGBM')].iloc[0]
    thresh = float(html_sub['Optimal_Threshold'])

    preds_h = (probs_h >= thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te_h, preds_h).ravel()
    total_errors = fp + fn

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_tr_h.values,
        feature_names=list(X_tr_h.columns),
        class_names=['benign', 'malicious'],
        mode='classification',
        random_state=SEED
    )

    error_indices = np.where(preds_h != y_te_h)[0]
    assert len(error_indices) == total_errors, f"Expected {total_errors} error indices, got {len(error_indices)}"

    lime_rows = []
    feature_names = list(X_tr_h.columns)

    for idx in error_indices:
        row_vals = X_te_h.iloc[idx].values
        y_true = int(y_te_h.iloc[idx])
        y_prob = float(probs_h[idx])

        exp = explainer.explain_instance(
            row_vals,
            clf_html.predict_proba,
            num_features=5,
            num_samples=500
        )

        exp_list = exp.as_list()

        row_dict = {
            'Sample_ID': int(idx),
            'True_Label': y_true,
            'Error_Type': 'FP' if (y_true == 0 and preds_h[idx] == 1) else 'FN',
            'Predicted_Prob': y_prob,
        }

        for i in range(5):
            if i < len(exp_list):
                feat_expr, weight = exp_list[i]
                matched_feat = "unknown"
                for fn_name in feature_names:
                    if fn_name in feat_expr:
                        matched_feat = fn_name
                        break
                row_dict[f'Top_{i+1}_Feature'] = matched_feat
                row_dict[f'Top_{i+1}_Weight'] = float(weight)
            else:
                row_dict[f'Top_{i+1}_Feature'] = "none"
                row_dict[f'Top_{i+1}_Weight'] = 0.0

        lime_rows.append(row_dict)

    df_lime = pd.DataFrame(lime_rows)
    df_lime.to_csv(os.path.join(RESULTS_DIR, "html_errors_lime.csv"), index=False)

    # --- SHAP & Permutation Importance Comparison on Error Samples ---
    X_err = X_te_h.iloc[error_indices]
    y_err = y_te_h.iloc[error_indices]

    tree_explainer = shap.TreeExplainer(clf_html)
    shap_vals_err = tree_explainer.shap_values(X_err)
    if isinstance(shap_vals_err, list):
        shap_vals_err = shap_vals_err[1]
    top_shap_feats = [X_err.columns[i] for i in np.argsort(np.abs(shap_vals_err).mean(axis=0))[::-1][:5]]

    perm_imp = permutation_importance(clf_html, X_err, y_err, n_repeats=5, random_state=SEED)
    top_perm_feats = [X_err.columns[i] for i in np.argsort(perm_imp.importances_mean)[::-1][:5]]

    # --- Top-1 Feature Counts per Error Type ---
    fp_lime = df_lime[df_lime['Error_Type'] == 'FP']
    fn_lime = df_lime[df_lime['Error_Type'] == 'FN']

    fp_top1_counts = fp_lime['Top_1_Feature'].value_counts().to_dict()
    fn_top1_counts = fn_lime['Top_1_Feature'].value_counts().to_dict()

    fp_top1_str = ", ".join([f"`{k}` ({v})" for k, v in list(fp_top1_counts.items())[:3]])
    fn_top1_str = ", ".join([f"`{k}` ({v})" for k, v in list(fn_top1_counts.items())[:3]])

    # K-Means clustering
    std_X_err = (X_err - X_tr_h.mean()) / X_tr_h.std().replace(0, 1.0)
    std_X_err = std_X_err.fillna(0)

    best_k = 2
    best_sil = -1.0
    for k_c in range(2, min(6, len(error_indices))):
        km = KMeans(n_clusters=k_c, random_state=SEED, n_init=10)
        labels = km.fit_predict(std_X_err)
        sil = float(silhouette_score(std_X_err, labels)) if len(np.unique(labels)) > 1 else -1.0
        if sil > best_sil:
            best_sil = sil
            best_k = k_c

    top_lime_feats = list(df_lime['Top_1_Feature'].value_counts().head(5).index)
    overlap_lime_shap = len(set(top_lime_feats) & set(top_shap_feats))
    overlap_lime_perm = len(set(top_lime_feats) & set(top_perm_feats))
    overlap_shap_perm = len(set(top_shap_feats) & set(top_perm_feats))
    overlap_all = len(set(top_lime_feats) & set(top_shap_feats) & set(top_perm_feats))

    # --- html_error_analysis.md ---
    md_content = f"""# HTML Specialist Model Error Analysis (LIME, SHAP & Permutation Audit)

## Error Statistics
- **Total Test Samples**: {len(y_te_h)}
- **Total Test Errors**: {total_errors} (False Positives: {fp}, False Negatives: {fn})
- **Test Accuracy**: {accuracy_score(y_te_h, preds_h):.4f}
- **Test F1 Score**: {f1_score(y_te_h, preds_h):.4f}

## Top-1 LIME Predictors by Error Type
- **False Positives (FP Top-1 Features)**: {fp_top1_str}
- **False Negatives (FN Top-1 Features)**: {fn_top1_str}

## Methodological Comparison on Error Samples
- **LIME Top-5 Features**: {", ".join([f"`{c}`" for c in top_lime_feats])}
- **SHAP Top-5 Features (Global Error Impact)**: {", ".join([f"`{c}`" for c in top_shap_feats])}
- **Permutation Importance Top-5 Features**: {", ".join([f"`{c}`" for c in top_perm_feats])}

## Top-5 Ranking Overlaps Across Methods
- **LIME & SHAP Overlap**: {overlap_lime_shap} of 5 features
- **LIME & Permutation Overlap**: {overlap_lime_perm} of 5 features
- **SHAP & Permutation Overlap**: {overlap_shap_perm} of 5 features
- **Features Shared Across All Three Methods**: {overlap_all} of 5 features

## Diagnostic Summary
- Local tabular perturbation in LIME highlights instance-level boundary features for error cases.
- Cross-method feature audit reveals {overlap_shap_perm} features shared between SHAP global impact and Permutation Importance on error instances, while LIME shares {overlap_lime_shap} top feature(s) with SHAP and {overlap_lime_perm} with Permutation Importance.

## Error Pattern Clustering (K-Means)
- **Optimal Error Clusters (K)**: {best_k}
- **Silhouette Score**: {best_sil:.4f}

Detailed per-sample signed LIME weights are recorded in `results/html_errors_lime.csv`.
"""

    with open(os.path.join(RESULTS_DIR, "html_error_analysis.md"), "w") as f:
        f.write(md_content)

    print(f"STAGE 5 EXECUTED SUCCESSFULLY: {len(df_lime)} LIME error explanations saved.", flush=True)

if __name__ == "__main__":
    run_stage5()
