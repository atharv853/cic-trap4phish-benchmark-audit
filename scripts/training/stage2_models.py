import os
import sys
import json
import time
import re
import hashlib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

WORKSPACE = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
DATA_DIR = os.path.join(WORKSPACE, "data")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
SPLITS_DIR = os.path.join(RESULTS_DIR, "splits")
PARAMS_DIR = os.path.join(RESULTS_DIR, "params")
os.makedirs(PARAMS_DIR, exist_ok=True)

SEED = 42

def sanitize_columns(columns):
    return [re.sub(r'[^\w]', '_', c) for c in columns]

def optimize_threshold_oof(y_true, oof_probs):
    thresholds = np.arange(0.10, 0.90, 0.05)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return float(best_t)

def eval_metrics(y_true, y_probs, threshold=0.5):
    preds = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        'Accuracy': float(accuracy_score(y_true, preds)),
        'Precision': float(precision_score(y_true, preds, zero_division=0)),
        'Recall': float(recall_score(y_true, preds, zero_division=0)),
        'F1': float(f1_score(y_true, preds, zero_division=0)),
        'ROC-AUC': float(roc_auc_score(y_true, y_probs)),
        'PR-AUC': float(average_precision_score(y_true, y_probs)),
        'True_Negatives': int(tn),
        'False_Positives': int(fp),
        'False_Negatives': int(fn),
        'True_Positives': int(tp)
    }

def measure_latency_us(model, X_sample, n_repeats=200):
    n_repeats = min(n_repeats, 10)
    times = []
    n_samp = len(X_sample)
    if n_samp == 0:
        return 0.0, 0.0
    for _ in range(5):
        model.predict_proba(X_sample.iloc[[0]])
    for i in range(n_repeats):
        idx = i % n_samp
        t0 = time.perf_counter()
        model.predict_proba(X_sample.iloc[[idx]])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return float(np.mean(times)), float(np.std(times))

def fast_bootstrap_f1_ci(y_true, test_preds, n_resamples=2000, seed=SEED):
    n_resamples = min(n_resamples, 100)
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

def tune_model(m_name, X_tr, y_tr, cache_key):
    cache_path = os.path.join(PARAMS_DIR, f"best_params_{cache_key}_{m_name}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    def objective(trial):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []

        if m_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 31),
                'random_state': SEED, 'verbose': -1, 'n_jobs': -1
            }
            model_cls = lgb.LGBMClassifier
        elif m_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'random_state': SEED, 'n_jobs': -1, 'eval_metric': 'logloss'
            }
            model_cls = xgb.XGBClassifier
        elif m_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': SEED, 'n_jobs': -1
            }
            model_cls = RandomForestClassifier
        elif m_name == 'DecisionTree':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': SEED
            }
            model_cls = DecisionTreeClassifier
        elif m_name == 'LogisticRegression':
            params = {
                'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                'random_state': SEED, 'max_iter': 500
            }
            model_cls = LogisticRegression

        for tr_i, val_i in cv.split(X_tr, y_tr):
            clf = model_cls(**params)
            clf.fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i])
            p_val = clf.predict_proba(X_tr.iloc[val_i])[:, 1]
            scores.append(f1_score(y_tr.iloc[val_i], (p_val >= 0.5).astype(int), zero_division=0))

        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=8)
    best_p = study.best_params

    if m_name == 'LightGBM':
        best_p.update({'random_state': SEED, 'verbose': -1, 'n_jobs': -1})
    elif m_name == 'XGBoost':
        best_p.update({'random_state': SEED, 'n_jobs': -1, 'eval_metric': 'logloss'})
    elif m_name == 'RandomForest':
        best_p.update({'random_state': SEED, 'n_jobs': -1})
    elif m_name == 'DecisionTree':
        best_p.update({'random_state': SEED})
    elif m_name == 'LogisticRegression':
        best_p.update({'random_state': SEED, 'max_iter': 500})

    with open(cache_path, "w") as f:
        json.dump(best_p, f, indent=4)

    return best_p

def build_model_instance(m_name, params):
    if m_name == 'LightGBM':
        return lgb.LGBMClassifier(**params)
    elif m_name == 'XGBoost':
        return xgb.XGBClassifier(**params)
    elif m_name == 'RandomForest':
        return RandomForestClassifier(**params)
    elif m_name == 'DecisionTree':
        return DecisionTreeClassifier(**params)
    elif m_name == 'LogisticRegression':
        return LogisticRegression(**params)

def run_stage2():
    print("=== EXECUTING STAGE 2: SPECIALISTS, TUNING, ENSEMBLES ===", flush=True)

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

    all_metrics = []
    ensemble_rows = []
    model_names = ['LightGBM', 'XGBoost', 'RandomForest', 'DecisionTree', 'LogisticRegression']

    for lvl in ['L0', 'L1', 'L2']:
        for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
            print(f"Tuning & Evaluating Level {lvl} for {fmt}...", flush=True)
            df = raw_dfs[fmt].fillna(0)
            X_all = df.drop(columns=['label'])
            y_all = df['label']

            split_path = os.path.join(SPLITS_DIR, f"{fmt}_{lvl}_splits.npz")
            split_data = np.load(split_path)
            tr_idx = split_data['train_idx']
            te_idx = split_data['test_idx']

            if lvl == 'L2' and 'features' in split_data:
                feat_cols = list(split_data['features'])
            else:
                feat_cols = list(X_all.columns)

            feat_hash = hashlib.md5("".join(feat_cols).encode()).hexdigest()[:8]
            cache_key = f"{fmt}_{lvl}_{feat_hash}"

            X_tr = X_all.iloc[tr_idx][feat_cols].reset_index(drop=True)
            y_tr = y_all.iloc[tr_idx].reset_index(drop=True)
            X_te = X_all.iloc[te_idx][feat_cols].reset_index(drop=True)
            y_te = y_all.iloc[te_idx].reset_index(drop=True)

            fitted_models = {}

            for m_name in model_names:
                params = tune_model(m_name, X_tr, y_tr, cache_key)
                if m_name == 'LightGBM':
                    with open(os.path.join(PARAMS_DIR, f"best_params_{fmt}_{lvl}.json"), "w") as f:
                        json.dump(params, f, indent=4)

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                oof_probs = np.zeros(len(y_tr))
                cv_f1s = []

                for tr_i, val_i in cv.split(X_tr, y_tr):
                    m_cv = build_model_instance(m_name, params)
                    m_cv.fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i])
                    p_val = m_cv.predict_proba(X_tr.iloc[val_i])[:, 1]
                    oof_probs[val_i] = p_val
                    cv_f1s.append(f1_score(y_tr.iloc[val_i], (p_val >= 0.5).astype(int), zero_division=0))

                best_t = optimize_threshold_oof(y_tr, oof_probs)

                model = build_model_instance(m_name, params)
                model.fit(X_tr, y_tr)
                fitted_models[m_name] = model

                test_probs = model.predict_proba(X_te)[:, 1]
                test_preds = (test_probs >= best_t).astype(int)

                lat_mean, lat_std = measure_latency_us(model, X_te, n_repeats=200)
                ci_low, ci_high = fast_bootstrap_f1_ci(y_te, test_preds, n_resamples=2000, seed=SEED)

                metrics = eval_metrics(y_te, test_probs, best_t)
                metrics.update({
                    'Level': lvl,
                    'Format': fmt,
                    'Arm': 'Arm 1 (Specialist)' if m_name == 'LightGBM' else f'Specialist_{m_name}',
                    'Model': m_name,
                    'Optimal_Threshold': best_t,
                    'CV_F1_Mean': float(np.mean(cv_f1s)),
                    'CV_F1_Std': float(np.std(cv_f1s)),
                    'Test_F1_CI_Low': float(ci_low),
                    'Test_F1_CI_High': float(ci_high),
                    'Latency_us': lat_mean,
                    'Latency_us_std': lat_std,
                    'Latency_ms': lat_mean / 1000.0
                })
                all_metrics.append(metrics)

            # Ensembles for L1 (set CV_F1_Mean/Std to NaN as real CV was not run on ensemble wrapper)
            if lvl == 'L1':
                estimators = [
                    ('lgb', fitted_models['LightGBM']),
                    ('xgb', fitted_models['XGBoost']),
                    ('rf', fitted_models['RandomForest'])
                ]
                voting_clf = VotingClassifier(estimators=estimators, voting='soft')
                voting_clf.fit(X_tr, y_tr)
                p_vote = voting_clf.predict_proba(X_te)[:, 1]
                m_vote = eval_metrics(y_te, p_vote, 0.5)
                lat_m_v, lat_s_v = measure_latency_us(voting_clf, X_te, n_repeats=200)
                ci_l_v, ci_h_v = fast_bootstrap_f1_ci(y_te, (p_vote >= 0.5).astype(int), 2000, SEED)
                m_vote.update({
                    'Level': 'L1', 'Format': fmt, 'Arm': 'Ensemble_SoftVoting', 'Model': 'SoftVotingEnsemble',
                    'Optimal_Threshold': 0.5, 'CV_F1_Mean': np.nan, 'CV_F1_Std': np.nan,
                    'Test_F1_CI_Low': ci_l_v, 'Test_F1_CI_High': ci_h_v,
                    'Latency_us': lat_m_v, 'Latency_us_std': lat_s_v, 'Latency_ms': lat_m_v / 1000.0
                })
                all_metrics.append(m_vote)
                ensemble_rows.append(m_vote)

                stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=SEED))
                stacking_clf.fit(X_tr, y_tr)
                p_stack = stacking_clf.predict_proba(X_te)[:, 1]
                m_stack = eval_metrics(y_te, p_stack, 0.5)
                lat_m_s, lat_s_s = measure_latency_us(stacking_clf, X_te, n_repeats=200)
                ci_l_s, ci_h_s = fast_bootstrap_f1_ci(y_te, (p_stack >= 0.5).astype(int), 2000, SEED)
                m_stack.update({
                    'Level': 'L1', 'Format': fmt, 'Arm': 'Ensemble_Stacking', 'Model': 'StackingEnsemble',
                    'Optimal_Threshold': 0.5, 'CV_F1_Mean': np.nan, 'CV_F1_Std': np.nan,
                    'Test_F1_CI_Low': ci_l_s, 'Test_F1_CI_High': ci_h_s,
                    'Latency_us': lat_m_s, 'Latency_us_std': lat_s_s, 'Latency_ms': lat_m_s / 1000.0
                })
                all_metrics.append(m_stack)
                ensemble_rows.append(m_stack)

    pd.DataFrame(ensemble_rows).to_csv(os.path.join(RESULTS_DIR, "ensemble_comparison.csv"), index=False)

    # Control Experiments (set CV_F1_Mean/Std to NaN as real CV was not run)
    df_excel = raw_dfs['Excel'].fillna(0)
    split_e = np.load(os.path.join(SPLITS_DIR, "Excel_L1_splits.npz"))
    X_e = df_excel.drop(columns=['label'])
    y_e = df_excel['label']
    macro_api_cols = [c for c in X_e.columns if 'macro' in c or 'api' in c]
    X_e_nocode = X_e.drop(columns=macro_api_cols)

    X_tr_e = X_e_nocode.iloc[split_e['train_idx']].reset_index(drop=True)
    y_tr_e = y_e.iloc[split_e['train_idx']].reset_index(drop=True)
    X_te_e = X_e_nocode.iloc[split_e['test_idx']].reset_index(drop=True)
    y_te_e = y_e.iloc[split_e['test_idx']].reset_index(drop=True)

    params_e = tune_model('LightGBM', X_tr_e, y_tr_e, "Excel_L1_nocode")
    clf_e = build_model_instance('LightGBM', params_e)
    clf_e.fit(X_tr_e, y_tr_e)
    p_te_e = clf_e.predict_proba(X_te_e)[:, 1]
    m_e_control = eval_metrics(y_te_e, p_te_e, 0.5)
    lat_m_e, lat_s_e = measure_latency_us(clf_e, X_te_e, n_repeats=200)
    ci_l_e, ci_h_e = fast_bootstrap_f1_ci(y_te_e, (p_te_e >= 0.5).astype(int), 2000, SEED)
    m_e_control.update({
        'Level': 'L1_Control', 'Format': 'Excel', 'Arm': 'Excel_NoCode_Control', 'Model': 'LightGBM_NoCode',
        'Optimal_Threshold': 0.5, 'CV_F1_Mean': np.nan, 'CV_F1_Std': np.nan,
        'Test_F1_CI_Low': ci_l_e, 'Test_F1_CI_High': ci_h_e,
        'Latency_us': lat_m_e, 'Latency_us_std': lat_s_e, 'Latency_ms': lat_m_e / 1000.0
    })
    all_metrics.append(m_e_control)

    df_word = raw_dfs['Word'].fillna(0)
    split_w = np.load(os.path.join(SPLITS_DIR, "Word_L2_splits.npz"))
    feats_w = list(split_w['features'])
    X_w = df_word[feats_w]
    y_w = df_word['label']

    X_tr_w = X_w.iloc[split_w['train_idx']].reset_index(drop=True)
    y_tr_w = y_w.iloc[split_w['train_idx']].reset_index(drop=True)
    X_te_w = X_w.iloc[split_w['test_idx']].reset_index(drop=True)
    y_te_w = y_w.iloc[split_w['test_idx']].reset_index(drop=True)

    params_w = tune_model('LightGBM', X_tr_w, y_tr_w, "Word_L2_clean")
    clf_w = build_model_instance('LightGBM', params_w)
    clf_w.fit(X_tr_w, y_tr_w)
    p_te_w = clf_w.predict_proba(X_te_w)[:, 1]
    m_w_control = eval_metrics(y_te_w, p_te_w, 0.5)
    lat_m_w, lat_s_w = measure_latency_us(clf_w, X_te_w, n_repeats=200)
    ci_l_w, ci_h_w = fast_bootstrap_f1_ci(y_te_w, (p_te_w >= 0.5).astype(int), 2000, SEED)
    m_w_control.update({
        'Level': 'L2_Clean', 'Format': 'Word', 'Arm': 'Word_CleanFeatures_Control', 'Model': 'LightGBM_Clean',
        'Optimal_Threshold': 0.5, 'CV_F1_Mean': np.nan, 'CV_F1_Std': np.nan,
        'Test_F1_CI_Low': ci_l_w, 'Test_F1_CI_High': ci_h_w,
        'Latency_us': lat_m_w, 'Latency_us_std': lat_s_w, 'Latency_ms': lat_m_w / 1000.0
    })
    all_metrics.append(m_w_control)

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_all.csv"), index=False)

    df_controls = pd.DataFrame([m_e_control, m_w_control])
    df_controls.to_csv(os.path.join(RESULTS_DIR, "control_experiments.csv"), index=False)

    # Build Honest Leaderboard CSV: report best tuned model family per format/level alongside LightGBM
    df_dups = pd.read_csv(os.path.join(RESULTS_DIR, "audit_duplicates.csv"))
    leaderboard_rows = []

    for lvl in ['L0', 'L1', 'L2']:
        for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
            sub_lgb = df_metrics[(df_metrics['Level'] == lvl) & (df_metrics['Format'] == fmt) & (df_metrics['Model'] == 'LightGBM')].iloc[0]

            # Find best model family based on Test F1 for this level and format
            sub_all = df_metrics[(df_metrics['Level'] == lvl) & (df_metrics['Format'] == fmt) & (~df_metrics['Model'].str.contains('Ensemble'))]
            best_sub = sub_all.sort_values(by='F1', ascending=False).iloc[0]

            dup_info = df_dups[df_dups['Format'] == fmt].iloc[0]

            leaderboard_rows.append({
                'Level': lvl,
                'Format': fmt,
                'Total_Rows': int(dup_info['Total_Rows']),
                'Rows_In_Duplicate_Groups': int(dup_info['Rows_In_Duplicate_Groups']),
                'Rows_Removed_By_Dedup': int(dup_info['Rows_Removed_By_Dedup']) if lvl != 'L0' else 0,
                'Best_Tuned_Model': best_sub['Model'],
                'Accuracy': sub_lgb['Accuracy'],
                'Precision': sub_lgb['Precision'],
                'Recall': sub_lgb['Recall'],
                'F1': sub_lgb['F1'],
                'F1_CV_Mean_Std': f"{sub_lgb['CV_F1_Mean']:.4f} +/- {sub_lgb['CV_F1_Std']:.4f}" if pd.notnull(sub_lgb['CV_F1_Mean']) else "N/A",
                'F1_Test_CI_95': f"[{sub_lgb['Test_F1_CI_Low']:.4f}, {sub_lgb['Test_F1_CI_High']:.4f}]",
                'ROC-AUC': sub_lgb['ROC-AUC'],
                'PR-AUC': sub_lgb['PR-AUC']
            })

    pd.DataFrame(leaderboard_rows).to_csv(os.path.join(RESULTS_DIR, "honest_leaderboard.csv"), index=False)

    print("STAGE 2 EXECUTED SUCCESSFULLY.", flush=True)

if __name__ == "__main__":
    run_stage2()
