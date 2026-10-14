import os
import sys
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb

WORKSPACE = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
DATA_DIR = os.path.join(WORKSPACE, "data")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
SPLITS_DIR = os.path.join(RESULTS_DIR, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

SEED = 42

def sanitize_columns(columns):
    return [re.sub(r'[^\w]', '_', c) for c in columns]

def run_stage1():
    print("=== EXECUTING STAGE 1: THREE EVALUATION LEVELS ===")

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

    l2_removed_rows = []
    erosion_rows = []
    l2_feature_lists = {}

    for fmt, df in raw_dfs.items():
        # Impute NaN locally for model splits
        df_imp = df.fillna(0)
        X_l0 = df_imp.drop(columns=['label'])
        y_l0 = df_imp['label']

        # --- Level 0: RAW ---
        idx_all = np.arange(len(df_imp))
        tr_idx_l0, te_idx_l0 = train_test_split(idx_all, test_size=0.2, random_state=SEED, stratify=y_l0)

        np.savez(os.path.join(SPLITS_DIR, f"{fmt}_L0_splits.npz"), train_idx=tr_idx_l0, test_idx=te_idx_l0)

        # --- Level 1: DEDUP ---
        X_cols = list(X_l0.columns)
        df_comb = X_l0.copy()
        df_comb['label'] = y_l0
        label_counts = df_comb.groupby(X_cols)['label'].nunique()
        conflict_indices = label_counts[label_counts > 1].index

        if len(conflict_indices) > 0:
            conflict_set = set(conflict_indices)
            tup_series = list(zip(*[df_comb[c] for c in X_cols]))
            is_conflict = [t in conflict_set for t in tup_series]
            df_no_conflict = df_comb[~np.array(is_conflict)].copy()
        else:
            df_no_conflict = df_comb.copy()

        df_l1 = df_no_conflict.drop_duplicates(subset=X_cols, keep='first').reset_index(drop=True)
        X_l1 = df_l1.drop(columns=['label'])
        y_l1 = df_l1['label']

        idx_l1_all = np.arange(len(df_l1))
        tr_idx_l1, te_idx_l1 = train_test_split(idx_l1_all, test_size=0.2, random_state=SEED, stratify=y_l1)

        # Assert zero exact twins between train and test at L1
        tr_set = set(tuple(r) for r in X_l1.iloc[tr_idx_l1].values)
        te_twins = sum(tuple(r) in tr_set for r in X_l1.iloc[te_idx_l1].values)
        assert te_twins == 0, f"Found {te_twins} exact twins in L1 split for {fmt}!"

        np.savez(os.path.join(SPLITS_DIR, f"{fmt}_L1_splits.npz"), train_idx=tr_idx_l1, test_idx=te_idx_l1)

        # --- Level 2: HONEST ---
        # (i) Drop every column with any NaN in raw data
        nan_cols = [c for c in df.columns if df[c].isnull().sum() > 0]

        X_tr_l1 = X_l1.iloc[tr_idx_l1]
        y_tr_l1 = y_l1.iloc[tr_idx_l1]
        X_te_l1 = X_l1.iloc[te_idx_l1]
        y_te_l1 = y_l1.iloc[te_idx_l1]

        candidates = [c for c in X_l1.columns if c not in nan_cols]
        removed_in_order = []

        for col in nan_cols:
            l2_removed_rows.append({
                'Format': fmt,
                'Feature': col,
                'Stump_Accuracy': 1.0 if col in df_imp.columns else 0.0,
                'Reason': 'Contains NaN in raw data'
            })

        while True:
            best_col = None
            best_acc = 0.0

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for col in candidates:
                accs = []
                for tr_i, val_i in cv.split(X_tr_l1, y_tr_l1):
                    clf = DecisionTreeClassifier(max_depth=1, random_state=SEED)
                    clf.fit(X_tr_l1[[col]].iloc[tr_i], y_tr_l1.iloc[tr_i])
                    preds = clf.predict(X_tr_l1[[col]].iloc[val_i])
                    accs.append(accuracy_score(y_tr_l1.iloc[val_i], preds))
                mean_acc = float(np.mean(accs))
                if mean_acc >= 0.95 and mean_acc > best_acc:
                    best_acc = mean_acc
                    best_col = col

            if best_col is not None:
                candidates.remove(best_col)
                removed_in_order.append(best_col)
                l2_removed_rows.append({
                    'Format': fmt,
                    'Feature': best_col,
                    'Stump_Accuracy': best_acc,
                    'Reason': f'Train stump accuracy >= 0.95 ({best_acc:.4f})'
                })
            else:
                break

        if not candidates:
            candidates = [X_l1.columns[0]]

        l2_feature_lists[fmt] = candidates
        np.savez(os.path.join(SPLITS_DIR, f"{fmt}_L2_splits.npz"), train_idx=tr_idx_l1, test_idx=te_idx_l1, features=np.array(candidates))

        # --- Shortcut Erosion Curve ---
        all_cols_ordered = sorted(X_l1.columns, key=lambda c: float(np.mean([
            accuracy_score(y_tr_l1.iloc[val_i], DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X_tr_l1[[c]].iloc[tr_i], y_tr_l1.iloc[tr_i]).predict(X_tr_l1[[c]].iloc[val_i]))
            for tr_i, val_i in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_tr_l1, y_tr_l1)
        ])), reverse=True)

        for num_removed in range(min(20, len(all_cols_ordered))):
            rem_cols = all_cols_ordered[num_removed:]
            if len(rem_cols) == 0: break

            clf = lgb.LGBMClassifier(random_state=SEED, verbose=-1)
            clf.fit(X_tr_l1[rem_cols], y_tr_l1)
            preds = clf.predict(X_te_l1[rem_cols])
            f1_val = float(f1_score(y_te_l1, preds, zero_division=0))
            acc_val = float(accuracy_score(y_te_l1, preds))

            erosion_rows.append({
                'Format': fmt,
                'Num_Removed': num_removed,
                'Remaining_Features': len(rem_cols),
                'Test_F1': f1_val,
                'Test_Accuracy': acc_val
            })
            if f1_val < 0.90 and num_removed >= len(removed_in_order):
                break

    pd.DataFrame(l2_removed_rows).to_csv(os.path.join(RESULTS_DIR, "l2_removed_features.csv"), index=False)
    pd.DataFrame(erosion_rows).to_csv(os.path.join(RESULTS_DIR, "shortcut_erosion_curve.csv"), index=False)

    print("STAGE 1 EXECUTED SUCCESSFULLY.")
    return l2_feature_lists

if __name__ == "__main__":
    run_stage1()
