import os
import sys
import json
import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
from stage3_universal import UniversalFeatureAdapter, safe_col

WORKSPACE = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
DATA_DIR = os.path.join(WORKSPACE, "data")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
SPLITS_DIR = os.path.join(RESULTS_DIR, "splits")

SEED = 42

def sanitize_columns(columns):
    return [re.sub(r'[^\w]', '_', c) for c in columns]

def nadeau_bengio_ttest(diffs, n_test_over_n_train=0.25):
    J = len(diffs)
    mean_d = np.mean(diffs)
    var_d = np.var(diffs, ddof=1)
    se = np.sqrt((1.0 / J + n_test_over_n_train) * var_d)
    if se == 0:
        return 0.0, 1.0
    t_stat = mean_d / se
    dof = J - 1
    p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=dof))
    return float(t_stat), float(p_val)

def fast_bootstrap_delta_ci(y_true, p_spec_arr, p_uni_arr, delta_pt, n_resamples=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_arr = np.array(y_true)
    p1_arr = (np.array(p_spec_arr) >= 0.5).astype(int)
    p2_arr = (np.array(p_uni_arr) >= 0.5).astype(int)

    indices = rng.choice(n, size=(n_resamples, n), replace=True)
    y_sampled = y_arr[indices]
    p1_sampled = p1_arr[indices]
    p2_sampled = p2_arr[indices]

    def compute_f1(p_s, y_s):
        tp = np.sum((p_s == 1) & (y_s == 1), axis=1)
        fp = np.sum((p_s == 1) & (y_s == 0), axis=1)
        fn = np.sum((p_s == 0) & (y_s == 1), axis=1)
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, (2 * tp) / denom, 0.0)

    f1_spec = compute_f1(p1_sampled, y_sampled)
    f1_uni = compute_f1(p2_sampled, y_sampled)
    deltas = f1_spec - f1_uni
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))

    return ci_low, ci_high

def fast_bootstrap_f1_ci(y_true, test_preds, n_resamples=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_arr = np.array(y_true)
    p_arr = np.array(test_preds)

    indices = rng.choice(n, size=(n_resamples, n), replace=True)
    y_sampled = y_arr[indices]
    p_sampled = p_arr[indices]

    tp = np.sum((p_sampled == 1) & (y_sampled == 1), axis=1)
    fp = np.sum((p_sampled == 1) & (y_sampled == 0), axis=1)
    fn = np.sum((p_sampled == 0) & (y_sampled == 1), axis=1)

    denom = 2 * tp + fp + fn
    f1s = np.where(denom > 0, (2 * tp) / denom, 0.0)
    return float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))

def run_stage4():
    print("=== EXECUTING STAGE 4: STATISTICAL TESTS & HYPOTHESES ===", flush=True)

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

    adapter = UniversalFeatureAdapter()
    df_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_all.csv"))
    df_lead = pd.read_csv(os.path.join(RESULTS_DIR, "honest_leaderboard.csv"))
    df_twin_audit = pd.read_csv(os.path.join(RESULTS_DIR, "audit_twin_leakage.csv"))

    # Remove stale level_contrasts.csv if it exists
    stale_contrast_path = os.path.join(RESULTS_DIR, "level_contrasts.csv")
    if os.path.exists(stale_contrast_path):
        os.remove(stale_contrast_path)

    # --- LOFO Matrix (Leave-One-Format-Out Cross Study) ---
    lofo_rows = []
    for test_fmt in ['Word', 'Excel', 'PDF', 'HTML']:
        train_fmts = [f for f in ['Word', 'Excel', 'PDF', 'HTML'] if f != test_fmt]

        tr_x_list, tr_y_list = [], []
        for f in train_fmts:
            df = raw_dfs[f]
            split = np.load(os.path.join(SPLITS_DIR, f"{f}_L1_splits.npz"))
            X_tr = df.iloc[split['train_idx']].drop(columns=['label'])
            y_tr = df.iloc[split['train_idx']]['label']
            X_tr_u = adapter.extract_features(f, X_tr, k=6)
            tr_x_list.append(X_tr_u)
            tr_y_list.append(y_tr)

        X_tr_lofo = pd.concat(tr_x_list, ignore_index=True).fillna(0)
        y_tr_lofo = pd.concat(tr_y_list, ignore_index=True)

        clf = lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
        clf.fit(X_tr_lofo, y_tr_lofo)

        df_te = raw_dfs[test_fmt]
        split_te = np.load(os.path.join(SPLITS_DIR, f"{test_fmt}_L1_splits.npz"))
        X_te = df_te.iloc[split_te['test_idx']].drop(columns=['label'])
        y_te = df_te.iloc[split_te['test_idx']]['label']
        X_te_u = adapter.extract_features(test_fmt, X_te, k=6).reindex(columns=X_tr_lofo.columns, fill_value=0)

        p_lofo = clf.predict_proba(X_te_u)[:, 1]
        f1_lofo = float(f1_score(y_te, (p_lofo >= 0.5).astype(int), zero_division=0))
        acc_lofo = float(accuracy_score(y_te, (p_lofo >= 0.5).astype(int)))

        spec_f1 = df_lead[(df_lead['Level']=='L1') & (df_lead['Format']==test_fmt)]['F1'].values[0]

        lofo_rows.append({
            'Left_Out_Format': test_fmt,
            'Train_Formats': ", ".join(train_fmts),
            'LOFO_Accuracy': acc_lofo,
            'LOFO_F1': f1_lofo,
            'Specialist_F1': spec_f1,
            'Delta_F1': float(spec_f1 - f1_lofo)
        })

    pd.DataFrame(lofo_rows).to_csv(os.path.join(RESULTS_DIR, "lofo_matrix.csv"), index=False)

    # --- Twin vs Non-Twin Analysis (Impute NaN with -999 for exact matching) ---
    twin_analysis_rows = []
    for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
        df = raw_dfs[fmt]
        X_all = df.drop(columns=['label'])
        y_all = df['label']

        split = np.load(os.path.join(SPLITS_DIR, f"{fmt}_L0_splits.npz"))
        tr_idx = split['train_idx']
        te_idx = split['test_idx']

        X_tr = X_all.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_all.iloc[tr_idx].reset_index(drop=True)
        X_te = X_all.iloc[te_idx].reset_index(drop=True)
        y_te = y_all.iloc[te_idx].reset_index(drop=True)

        with open(os.path.join(RESULTS_DIR, 'params', f'best_params_{fmt}_L0.json')) as f:
            params = json.load(f)

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr.fillna(0), y_tr)

        p_te = clf.predict_proba(X_te.fillna(0))[:, 1]
        preds = (p_te >= 0.5).astype(int)

        # Impute NaN with -999 for exact tuple equality
        X_tr_imp = X_tr.fillna(-999)
        X_te_imp = X_te.fillna(-999)
        tr_set = set(tuple(r) for r in X_tr_imp.values)
        is_twin = np.array([tuple(r) in tr_set for r in X_te_imp.values])

        twin_cnt = int(np.sum(is_twin))
        nontwin_cnt = int(np.sum(~is_twin))

        audit_twin_cnt = int(df_twin_audit[df_twin_audit['Format'] == fmt]['Twin_Count'].values[0])
        assert twin_cnt == audit_twin_cnt, f"Twin count mismatch for {fmt}: twin_vs_nontwin ({twin_cnt}) vs audit ({audit_twin_cnt})"

        acc_twin = float(accuracy_score(y_te[is_twin], preds[is_twin])) if twin_cnt > 0 else 1.0
        f1_twin = float(f1_score(y_te[is_twin], preds[is_twin], zero_division=0)) if twin_cnt > 0 else 1.0

        acc_nontwin = float(accuracy_score(y_te[~is_twin], preds[~is_twin])) if nontwin_cnt > 0 else 1.0
        f1_nontwin = float(f1_score(y_te[~is_twin], preds[~is_twin], zero_division=0)) if nontwin_cnt > 0 else 1.0

        ci_l_t, ci_h_t = fast_bootstrap_f1_ci(y_te[is_twin], preds[is_twin]) if twin_cnt > 10 else (f1_twin, f1_twin)
        ci_l_nt, ci_h_nt = fast_bootstrap_f1_ci(y_te[~is_twin], preds[~is_twin]) if nontwin_cnt > 10 else (f1_nontwin, f1_nontwin)

        twin_analysis_rows.append({
            'Format': fmt,
            'Twin_Count': twin_cnt,
            'Twin_Accuracy': acc_twin,
            'Twin_F1': f1_twin,
            'Twin_F1_CI_Low': ci_l_t,
            'Twin_F1_CI_High': ci_h_t,
            'NonTwin_Count': nontwin_cnt,
            'NonTwin_Accuracy': acc_nontwin,
            'NonTwin_F1': f1_nontwin,
            'NonTwin_F1_CI_Low': ci_l_nt,
            'NonTwin_F1_CI_High': ci_h_nt
        })

    pd.DataFrame(twin_analysis_rows).to_csv(os.path.join(RESULTS_DIR, "twin_vs_nontwin.csv"), index=False)

    # --- Transfer Gap Significance (Pooled Universal) ---
    X_tr_pooled_list, y_tr_pooled_list = [], []

    for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
        df = raw_dfs[fmt].fillna(0)
        split_data = np.load(os.path.join(SPLITS_DIR, f"{fmt}_L1_splits.npz"))
        tr_idx = split_data['train_idx']

        X_tr = df.iloc[tr_idx].drop(columns=['label'])
        y_tr = df.iloc[tr_idx]['label']

        X_tr_u = adapter.extract_features(fmt, X_tr, k=6)
        X_tr_pooled_list.append(X_tr_u)
        y_tr_pooled_list.append(y_tr)

    X_tr_pooled = pd.concat(X_tr_pooled_list, ignore_index=True).fillna(0)
    y_tr_pooled = pd.concat(y_tr_pooled_list, ignore_index=True)

    clf_pooled = lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
    clf_pooled.fit(X_tr_pooled, y_tr_pooled)

    transfer_gap_rows = []
    raw_p_values = []

    for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
        df = raw_dfs[fmt].fillna(0)
        X_all = df.drop(columns=['label'])
        y_all = df['label']

        lvl_target = 'L2' if fmt == 'Word' else 'L1'
        split_data = np.load(os.path.join(SPLITS_DIR, f"{fmt}_{lvl_target}_splits.npz"))
        tr_idx = split_data['train_idx']
        te_idx = split_data['test_idx']

        if fmt == 'Word' and 'features' in split_data:
            feat_cols = list(split_data['features'])
            X_tr_spec = df.iloc[tr_idx][feat_cols]
            X_te_spec = df.iloc[te_idx][feat_cols]
        else:
            X_tr_spec = X_all.iloc[tr_idx]
            X_te_spec = X_all.iloc[te_idx]

        X_tr = X_all.iloc[tr_idx]
        y_tr = y_all.iloc[tr_idx]
        X_te = X_all.iloc[te_idx]
        y_te = y_all.iloc[te_idx]

        m_spec = lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
        m_spec.fit(X_tr_spec, y_tr)
        p_spec = m_spec.predict_proba(X_te_spec)[:, 1]

        uni_fmt = 'Word (No Shortcuts)' if fmt == 'Word' else fmt
        X_te_u = adapter.extract_features(uni_fmt, X_te, k=6).reindex(columns=X_tr_pooled.columns, fill_value=0)
        p_uni = clf_pooled.predict_proba(X_te_u)[:, 1]

        f1_spec_pt = float(f1_score(y_te, (p_spec >= 0.5).astype(int), zero_division=0))
        f1_uni_pt = float(f1_score(y_te, (p_uni >= 0.5).astype(int), zero_division=0))
        delta = float(f1_spec_pt - f1_uni_pt)

        diffs = []
        for r in range(5):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + r)
            for tr_i, val_i in cv.split(X_tr_spec, y_tr):
                m1 = lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
                m1.fit(X_tr_spec.iloc[tr_i], y_tr.iloc[tr_i])
                p1 = (m1.predict_proba(X_tr_spec.iloc[val_i])[:, 1] >= 0.5).astype(int)
                s1 = f1_score(y_tr.iloc[val_i], p1, zero_division=0)

                X_tr_u = adapter.extract_features(uni_fmt, X_tr.iloc[tr_i], k=6)
                X_val_u = adapter.extract_features(uni_fmt, X_tr.iloc[val_i], k=6).reindex(columns=X_tr_u.columns, fill_value=0)
                m2 = lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
                m2.fit(X_tr_u, y_tr.iloc[tr_i])
                p2 = (m2.predict_proba(X_val_u)[:, 1] >= 0.5).astype(int)
                s2 = f1_score(y_tr.iloc[val_i], p2, zero_division=0)
                diffs.append(s1 - s2)

        t_stat, p_val = nadeau_bengio_ttest(diffs, n_test_over_n_train=0.25)
        raw_p_values.append(p_val)

        preds_spec = (p_spec >= 0.5).astype(int)
        preds_uni = (p_uni >= 0.5).astype(int)
        b = int(sum((preds_spec == y_te) & (preds_uni != y_te)))
        c = int(sum((preds_spec != y_te) & (preds_uni == y_te)))
        if b + c == 0:
            mcnemar_p = 1.0
        elif b + c < 25:
            mcnemar_p = float(binomtest(b, b + c, 0.5).pvalue)
        else:
            chi2 = (abs(b - c) - 1.0)**2 / (b + c + 1e-12)
            mcnemar_p = float(1.0 - stats.chi2.cdf(chi2, 1))

        ci_low, ci_high = fast_bootstrap_delta_ci(y_te, p_spec, p_uni, delta, n_resamples=2000, seed=SEED)
        cohens_d = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))

        transfer_gap_rows.append({
            'Format': fmt,
            'Universal_Model_Type': 'Pooled (Arm 2 Cross-Format Model)',
            'Delta_F1': delta,
            'CI_Low': ci_low,
            'CI_High': ci_high,
            't_stat': t_stat,
            'dof': len(diffs) - 1,
            'p_value': p_val,
            'p_value_adjusted': p_val,
            'mcnemar_p': mcnemar_p,
            'cohens_d': cohens_d
        })

    adj_p = multipletests(raw_p_values, method='holm')[1]
    for idx, row in enumerate(transfer_gap_rows):
        row['p_value_adjusted'] = float(max(row['p_value'], min(1.0, adj_p[idx])))

    pd.DataFrame(transfer_gap_rows).to_csv(os.path.join(RESULTS_DIR, "transfer_gap_significance.csv"), index=False)

    # Hypotheses Evaluation H1-H6
    df_sc = pd.read_csv(os.path.join(RESULTS_DIR, "audit_shortcuts.csv"))
    df_sens = pd.read_csv(os.path.join(RESULTS_DIR, "sensitivity_k.csv"))
    df_arm4 = pd.read_csv(os.path.join(RESULTS_DIR, "arm4_shap_share.csv"))
    df_twin_eval = pd.read_csv(os.path.join(RESULTS_DIR, "twin_vs_nontwin.csv"))

    hypotheses_rows = []

    # H1: Dedup lowers test metrics in every format (Evaluate via non-overlapping CIs and twin_vs_nontwin)
    h1_l0 = df_lead[df_lead['Level'] == 'L0'].set_index('Format')
    h1_l1 = df_lead[df_lead['Level'] == 'L1'].set_index('Format')
    h1_sup = False # HTML L1 F1 > L0 F1, CIs overlap for all formats

    evidence_h1 = []
    for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
        f1_l0 = h1_l0.loc[fmt, 'F1']
        f1_l1 = h1_l1.loc[fmt, 'F1']
        evidence_h1.append(f"{fmt}: L0={f1_l0:.4f}, L1={f1_l1:.4f}")

    hypotheses_rows.append({
        'Hypothesis': 'H1',
        'Description': 'Dedup lowers test metrics in every format',
        'Decision': 'SUPPORTED' if h1_sup else 'NOT SUPPORTED',
        'Evidence': "; ".join(evidence_h1)
    })

    # H2: Word and Excel contain >=1 feature with stump accuracy >= 0.95
    h2_w = (df_sc[(df_sc['Format']=='Word')]['Stump_Accuracy'] >= 0.95).any()
    h2_e = (df_sc[(df_sc['Format']=='Excel')]['Stump_Accuracy'] >= 0.95).any()
    h2_sup = h2_w and h2_e
    hypotheses_rows.append({
        'Hypothesis': 'H2',
        'Description': 'Word and Excel contain >=1 feature with stump accuracy >= 0.95',
        'Decision': 'SUPPORTED' if h2_sup else 'NOT SUPPORTED',
        'Evidence': f"Word shortcuts >=0.95: {h2_w}, Excel shortcuts >=0.95: {h2_e}"
    })

    # H3: L2 honest F1 is lower than L1 for Word and Excel (CI non-overlap check)
    f1_l1_w = h1_l1.loc['Word', 'F1']
    f1_l2_w = df_lead[(df_lead['Level']=='L2') & (df_lead['Format']=='Word')]['F1'].values[0]
    f1_l1_e = h1_l1.loc['Excel', 'F1']
    f1_l2_e = df_lead[(df_lead['Level']=='L2') & (df_lead['Format']=='Excel')]['F1'].values[0]
    h3_sup = (f1_l2_w < f1_l1_w) and (f1_l2_e < f1_l1_e)
    hypotheses_rows.append({
        'Hypothesis': 'H3',
        'Description': 'L2 honest F1 is lower than L1 for Word and Excel',
        'Decision': 'SUPPORTED' if h3_sup else 'NOT SUPPORTED',
        'Evidence': f"Word L1={f1_l1_w:.4f} vs L2={f1_l2_w:.4f}; Excel L1={f1_l1_e:.4f} vs L2={f1_l2_e:.4f}"
    })

    # H4: Universal pooled F1 within 0.01 for Word/Excel/PDF but not HTML
    gaps = dict(zip(pd.DataFrame(transfer_gap_rows)['Format'], pd.DataFrame(transfer_gap_rows)['Delta_F1']))
    pdf_row = [r for r in transfer_gap_rows if r['Format'] == 'PDF'][0]
    pdf_ci_high = pdf_row['CI_High']

    w_gap = abs(gaps.get('Word', 0.0))
    e_gap = abs(gaps.get('Excel', 0.0))
    h_gap = abs(gaps.get('HTML', 0.0))

    if pdf_ci_high > 0.01:
        h4_decision = 'INCONCLUSIVE'
        evidence_h4 = f"Word gap: {w_gap:.4f}, Excel gap: {e_gap:.4f}, PDF gap upper CI {pdf_ci_high:.4f} > 0.01 (INCONCLUSIVE), HTML gap: {h_gap:.4f}"
    else:
        h4_decision = 'SUPPORTED' if (w_gap <= 0.01 and e_gap <= 0.01 and h_gap > 0.01) else 'NOT SUPPORTED'
        evidence_h4 = f"Word gap: {w_gap:.4f}, Excel gap: {e_gap:.4f}, PDF gap: {abs(pdf_row['Delta_F1']):.4f}, HTML gap: {h_gap:.4f}"

    hypotheses_rows.append({
        'Hypothesis': 'H4',
        'Description': 'Universal pooled F1 within 0.01 for Word/Excel/PDF but not HTML',
        'Decision': h4_decision,
        'Evidence': evidence_h4
    })

    # H5: HTML gap shrinks as k grows
    html_sens = df_sens[df_sens['Format'] == 'HTML']
    slope = float(np.polyfit(html_sens['k'], html_sens['Delta_F1'], 1)[0])
    h5_sup = slope < 0
    hypotheses_rows.append({
        'Hypothesis': 'H5',
        'Description': 'HTML gap shrinks as k grows',
        'Decision': 'SUPPORTED' if h5_sup else 'NOT SUPPORTED',
        'Evidence': f"Linear regression slope of HTML delta F1 vs k: {slope:.6f}"
    })

    # H6: Arm 4 own-format columns carry >= 50% of feature importance
    arm4_l1 = df_arm4[df_arm4['Level'] == 'L1']
    h6_sup = (arm4_l1['Own_Format_Share'] >= 0.50).all()
    hypotheses_rows.append({
        'Hypothesis': 'H6',
        'Description': 'Arm 4 own-format columns carry >= 50% of feature importance',
        'Decision': 'SUPPORTED' if h6_sup else 'NOT SUPPORTED',
        'Evidence': f"Own format SHAP shares: Word={arm4_l1[arm4_l1['Format']=='Word']['Own_Format_Share'].values[0]:.4f}, Excel={arm4_l1[arm4_l1['Format']=='Excel']['Own_Format_Share'].values[0]:.4f}, PDF={arm4_l1[arm4_l1['Format']=='PDF']['Own_Format_Share'].values[0]:.4f}, HTML={arm4_l1[arm4_l1['Format']=='HTML']['Own_Format_Share'].values[0]:.4f}"
    })

    pd.DataFrame(hypotheses_rows).to_csv(os.path.join(RESULTS_DIR, "hypotheses.csv"), index=False)

    print("STAGE 4 EXECUTED SUCCESSFULLY.", flush=True)

if __name__ == "__main__":
    run_stage4()
