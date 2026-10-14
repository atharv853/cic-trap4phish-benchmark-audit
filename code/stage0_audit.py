import os
import sys
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

WORKSPACE = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_DIR = os.path.join(WORKSPACE, "data")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

def sanitize_columns(columns):
    return [re.sub(r'[^\w]', '_', c) for c in columns]

import time
import shutil

def run_stage0():
    print("=== EXECUTING STAGE 0: DATA AUDIT ===")

    start_epoch = time.time()
    start_iso = time.strftime("%Y-%m-%d %H:%M:%S")

    # Delete stale directories completely
    for d in [RESULTS_DIR, os.path.join(WORKSPACE, "figures"), os.path.join(WORKSPACE, "latex_tables")]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "splits"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "params"), exist_ok=True)

    manifest_data = {
        'Run_Start_Time_ISO': start_iso,
        'Run_Start_Time_Epoch': start_epoch,
        'Seed': SEED,
        'Status': 'RUNNING'
    }
    with open(os.path.join(RESULTS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest_data, f, indent=4)

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
    provenance_paths = {}

    for fmt, path in file_paths.items():
        df = pd.read_csv(path)
        if 'file_path' in df.columns:
            provenance_paths[fmt] = df['file_path'].copy()
            df = df.drop(columns=['file_path'])
        else:
            provenance_paths[fmt] = None

        # Assert file_path is never in feature columns
        assert 'file_path' not in df.columns, f"file_path found in {fmt} features!"
        df.columns = sanitize_columns(df.columns)
        raw_dfs[fmt] = df

    # a) Shapes, class balance, missingness per column
    shapes_rows = []
    missingness_rows = []

    for fmt, df in raw_dfs.items():
        shapes_rows.append({
            'Format': fmt,
            'Rows': len(df),
            'Columns': df.shape[1] - 1,
            'Benign_Count': int((df['label'] == 0).sum()),
            'Malicious_Count': int((df['label'] == 1).sum()),
            'Malicious_Ratio': float((df['label'] == 1).mean())
        })
        for col in df.columns:
            if col == 'label': continue
            null_cnt = int(df[col].isnull().sum())
            if null_cnt > 0:
                missingness_rows.append({
                    'Format': fmt,
                    'Column': col,
                    'Null_Count': null_cnt,
                    'Null_Pct': float(null_cnt / len(df))
                })

    df_shapes = pd.DataFrame(shapes_rows)
    df_shapes.to_csv(os.path.join(RESULTS_DIR, "audit_shapes.csv"), index=False)

    df_miss = pd.DataFrame(missingness_rows)
    df_miss.to_csv(os.path.join(RESULTS_DIR, "audit_missingness.csv"), index=False)

    # Word 30 columns check
    word_df = raw_dfs['Word']
    word_null_frac = word_df.isnull().mean()
    word_miss_50 = word_null_frac[word_null_frac == 0.5].index.tolist()
    assert len(word_miss_50) == 30, f"Expected 30 columns missing 50% in Word, got {len(word_miss_50)}"

    # b) Exact duplicates, label-conflicting groups, near-duplicate rate
    # For models/distance calculations, impute NaN with 0 only locally
    imputed_dfs = {fmt: df.fillna(-999) for fmt, df in raw_dfs.items()}

    duplicates_rows = []
    twin_leakage_rows = []

    for fmt, df in imputed_dfs.items():
        X = df.drop(columns=['label'])
        y = df['label']

        dup_mask = X.duplicated(keep=False)
        dup_total = int(dup_mask.sum())
        dup_benign = int((dup_mask & (y == 0)).sum())
        dup_malicious = int((dup_mask & (y == 1)).sum())

        df_comb = X.copy()
        df_comb['label'] = y
        label_counts = df_comb.groupby(list(X.columns))['label'].nunique()
        conflict_groups = int((label_counts > 1).sum())

        # Naive stratified 80/20 split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

        # Near-duplicate rate under epsilon
        # Standardize train and test using train stats
        mean_tr = X_tr.mean()
        std_tr = X_tr.std().replace(0, 1.0)
        X_tr_std = (X_tr - mean_tr) / std_tr
        X_te_std = (X_te - mean_tr) / std_tr

        # Exact twin check
        tr_set = set(tuple(r) for r in X_tr.values)
        te_exact_twins = sum(tuple(r) in tr_set for r in X_te.values)

        # Near duplicate distance (sample 500 test rows for speed if large)
        n_sample = min(500, len(X_te_std))
        te_sub = X_te_std.iloc[:n_sample].values
        tr_vals = X_tr_std.values

        # Calculate min distance from each test row to train
        min_dists = []
        for i in range(n_sample):
            dists = np.linalg.norm(tr_vals - te_sub[i], axis=1)
            min_dists.append(np.min(dists))

        eps = 0.1 * np.sqrt(X.shape[1])
        near_dup_rate = float(np.mean(np.array(min_dists) <= eps))

        dedup_mask = X.duplicated(keep='first')
        removed_by_dedup = int(dedup_mask.sum())

        duplicates_rows.append({
            'Format': fmt,
            'Total_Rows': len(df),
            'Rows_In_Duplicate_Groups': dup_total,
            'Rows_Removed_By_Dedup': removed_by_dedup,
            'Duplicate_Benign': dup_benign,
            'Duplicate_Malicious': dup_malicious,
            'Duplicate_Pct': float(dup_total / len(df)),
            'Conflict_Groups': conflict_groups,
            'Near_Duplicate_Rate_Eps': near_dup_rate
        })

        twin_leakage_rows.append({
            'Format': fmt,
            'Level': 'L0',
            'Test_Rows': len(X_te),
            'Twin_Count': te_exact_twins,
            'Twin_Rate': float(te_exact_twins / len(X_te))
        })

    df_dups = pd.DataFrame(duplicates_rows)
    df_dups.to_csv(os.path.join(RESULTS_DIR, "audit_duplicates.csv"), index=False)

    df_twins = pd.DataFrame(twin_leakage_rows)
    df_twins.to_csv(os.path.join(RESULTS_DIR, "audit_twin_leakage.csv"), index=False)

    # d) Single-feature audit for EVERY feature (and missingness flags)
    # Flag >= 0.95 as SHORTCUT
    shortcut_rows = []

    for fmt, df in raw_dfs.items():
        X_imp = df.drop(columns=['label']).fillna(0)
        y = df['label']

        # Add missingness flags as features if missingness present
        for col in df.columns:
            if col == 'label': continue
            if df[col].isnull().sum() > 0:
                X_imp[f"{col}_is_missing"] = df[col].isnull().astype(int)

        X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.2, random_state=SEED, stratify=y)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        for col in X_imp.columns:
            accs, f1s, aucs = [], [], []
            for tr_idx, val_idx in cv.split(X_tr, y_tr):
                clf = DecisionTreeClassifier(max_depth=1, random_state=SEED)
                X_sub_tr = X_tr[[col]].iloc[tr_idx]
                y_sub_tr = y_tr.iloc[tr_idx]
                X_sub_val = X_tr[[col]].iloc[val_idx]
                y_sub_val = y_tr.iloc[val_idx]

                clf.fit(X_sub_tr, y_sub_tr)
                preds = clf.predict(X_sub_val)
                probs = clf.predict_proba(X_sub_val)[:, 1] if hasattr(clf, "predict_proba") else preds

                accs.append(accuracy_score(y_sub_val, preds))
                f1s.append(f1_score(y_sub_val, preds, zero_division=0))
                try:
                    aucs.append(roc_auc_score(y_sub_val, probs))
                except Exception:
                    aucs.append(0.5)

            mean_acc = float(np.mean(accs))
            mean_f1 = float(np.mean(f1s))
            mean_auc = float(np.mean(aucs))
            is_shortcut = bool(mean_acc >= 0.95)

            shortcut_rows.append({
                'Format': fmt,
                'Feature': col,
                'Stump_Accuracy': mean_acc,
                'Stump_F1': mean_f1,
                'Stump_ROC_AUC': mean_auc,
                'Is_Shortcut': is_shortcut
            })

    df_shortcuts = pd.DataFrame(shortcut_rows).sort_values(by=['Format', 'Stump_Accuracy'], ascending=[True, False])
    df_shortcuts.to_csv(os.path.join(RESULTS_DIR, "audit_shortcuts.csv"), index=False)

    # e) Provenance audit: folder vs label, extension vs label
    provenance_rows = []
    for fmt, paths in provenance_paths.items():
        if paths is not None:
            y = raw_dfs[fmt]['label']
            folders = paths.apply(lambda p: os.path.dirname(str(p)))
            exts = paths.apply(lambda p: os.path.splitext(str(p))[1].lower())

            f_codes = pd.factorize(folders)[0].reshape(-1, 1)
            clf_f = DecisionTreeClassifier(max_depth=1, random_state=SEED)
            clf_f.fit(f_codes, y)
            acc_folder = float(accuracy_score(y, clf_f.predict(f_codes)))

            e_codes = pd.factorize(exts)[0].reshape(-1, 1)
            clf_e = DecisionTreeClassifier(max_depth=1, random_state=SEED)
            clf_e.fit(e_codes, y)
            acc_ext = float(accuracy_score(y, clf_e.predict(e_codes)))

            provenance_rows.append({
                'Format': fmt,
                'Folder_Leakage_Accuracy': acc_folder,
                'Extension_Leakage_Accuracy': acc_ext
            })
        else:
            provenance_rows.append({
                'Format': fmt,
                'Folder_Leakage_Accuracy': 1.0,
                'Extension_Leakage_Accuracy': 1.0
            })

    df_prov = pd.DataFrame(provenance_rows)
    df_prov.to_csv(os.path.join(RESULTS_DIR, "audit_provenance.csv"), index=False)

    # f) Word study: missingness-flag-only, all 43, 13 non-missing, 13 minus shortcuts, etc.
    word_miss_rows = []
    w_df_raw = raw_dfs['Word'].copy()
    w_df_dedup = w_df_raw.drop_duplicates().copy()

    for setting_name, df_w in [('Raw', w_df_raw), ('Dedup', w_df_dedup)]:
        y_w = df_w['label']

        # Missingness flag only
        X_miss = df_w[word_miss_50[0]].isnull().astype(int).to_frame('miss_flag')
        X_tr, X_te, y_tr, y_te = train_test_split(X_miss, y_w, test_size=0.2, random_state=SEED, stratify=y_w)
        clf = DecisionTreeClassifier(max_depth=1, random_state=SEED)
        clf.fit(X_tr, y_tr)
        acc_miss = float(accuracy_score(y_te, clf.predict(X_te)))

        # All 43 features
        X_full = df_w.drop(columns=['label']).fillna(0)
        X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_w, test_size=0.2, random_state=SEED, stratify=y_w)
        clf_all = DecisionTreeClassifier(random_state=SEED)
        clf_all.fit(X_tr, y_tr)
        acc_all = float(accuracy_score(y_te, clf_all.predict(X_te)))

        # 13 non-missing features
        non_miss_cols = [c for c in X_full.columns if c not in word_miss_50]
        clf_13 = DecisionTreeClassifier(random_state=SEED)
        clf_13.fit(X_tr[non_miss_cols], y_tr)
        acc_13 = float(accuracy_score(y_te, clf_13.predict(X_te[non_miss_cols])))

        # 13 minus {ole_*, macro_present, dde_present, vba_keywords_count}
        shortcut_sub = ['ole_object_count', 'macro_present', 'dde_present', 'vba_keywords_count', 'ole_object_type_count']
        rem_13 = [c for c in non_miss_cols if not any(s in c for s in shortcut_sub)]
        clf_rem = DecisionTreeClassifier(random_state=SEED)
        clf_rem.fit(X_tr[rem_13], y_tr)
        acc_rem = float(accuracy_score(y_te, clf_rem.predict(X_te[rem_13])))

        # ALSO remove shortcuts >= 0.95
        w_sc_cols = df_shortcuts[(df_shortcuts['Format'] == 'Word') & (df_shortcuts['Is_Shortcut'])]['Feature'].tolist()
        clean_cols = [c for c in rem_13 if c not in w_sc_cols]
        if not clean_cols:
            clean_cols = rem_13
        clf_clean = DecisionTreeClassifier(random_state=SEED)
        clf_clean.fit(X_tr[clean_cols], y_tr)
        acc_clean = float(accuracy_score(y_te, clf_clean.predict(X_te[clean_cols])))

        word_miss_rows.append({
            'Data_Setting': setting_name,
            'Missingness_Flag_Only_Acc': acc_miss,
            'All_43_Features_Acc': acc_all,
            'Non_Missing_13_Acc': acc_13,
            'Non_Missing_Minus_Shortcuts_Acc': acc_rem,
            'Clean_Features_Acc': acc_clean
        })

    df_w_analysis = pd.DataFrame(word_miss_rows)
    df_w_analysis.to_csv(os.path.join(RESULTS_DIR, "word_missingness_analysis.csv"), index=False)

    print("STAGE 0 EXECUTED SUCCESSFULLY.")

if __name__ == "__main__":
    run_stage0()
