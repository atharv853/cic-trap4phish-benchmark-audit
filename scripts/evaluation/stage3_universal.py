import os
import sys
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

WORKSPACE = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
DATA_DIR = os.path.join(WORKSPACE, "data")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
SPLITS_DIR = os.path.join(RESULTS_DIR, "splits")

SEED = 42

def sanitize_columns(columns):
    return [re.sub(r'[^\w]', '_', c) for c in columns]

def safe_col(df, col, default=0.0):
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(default, index=df.index, dtype=float)

class UniversalFeatureAdapter:
    def __init__(self):
        self.mapping_log = {}

    def extract_features(self, fmt, X_in, k=6, include_shortcuts=True):
        X = X_in.copy()
        uni = pd.DataFrame(index=X.index)

        is_no_sc = (fmt == 'Word (No Shortcuts)') or (not include_shortcuts)

        if fmt == 'Word' or fmt == 'Word (No Shortcuts)':
            if is_no_sc:
                uni['1_content_entropy'] = 0.0
                uni['2_active_content_flag'] = (safe_col(X, 'struct_script') > 0).astype(float)
                uni['3_external_reference_count'] = 0.0
                uni['4_log_file_size'] = 0.0
                uni['5_structural_complexity_ratio'] = safe_col(X, 'struct_typeface') / 10.0
                uni['6_obfuscation_indicator'] = 0.0
                uni['7_script_text_volume'] = safe_col(X, 'struct_typeface')
                uni['8_encoded_content_counts'] = 0.0
                uni['9_embedded_object_count'] = 0.0
                uni['10_stream_script_size'] = safe_col(X, 'struct_script')
            else:
                uni['1_content_entropy'] = safe_col(X, 'entropy')
                uni['2_active_content_flag'] = (safe_col(X, 'macro_present') > 0).astype(float)
                uni['3_external_reference_count'] = 0.0
                uni['4_log_file_size'] = np.log1p(np.maximum(0, safe_col(X, 'file_size', 1.0)))
                uni['5_structural_complexity_ratio'] = safe_col(X, 'ole_object_count') / (safe_col(X, 'file_size') + 1.0)
                uni['6_obfuscation_indicator'] = safe_col(X, 'vba_keywords_count')
                uni['7_script_text_volume'] = safe_col(X, 'struct_typeface')
                uni['8_encoded_content_counts'] = safe_col(X, 'ole_object_type_count')
                uni['9_embedded_object_count'] = safe_col(X, 'ole_object_count')
                uni['10_stream_script_size'] = safe_col(X, 'struct_script')

            self.mapping_log['Word_NoShortcuts' if is_no_sc else 'Word'] = {
                '1_content_entropy': '0.0 (excluded shortcut)' if is_no_sc else 'entropy',
                '2_active_content_flag': 'struct_script > 0' if is_no_sc else 'macro_present > 0',
                '3_external_reference_count': 'EXCLUDED (Word dataset lacks link count column)',
                '4_log_file_size': '0.0 (excluded file_size shortcut)' if is_no_sc else 'log1p(file_size)',
                '5_structural_complexity_ratio': 'struct_typeface / 10' if is_no_sc else 'ole_object_count / file_size',
                '6_obfuscation_indicator': '0.0 (excluded vba shortcut)' if is_no_sc else 'vba_keywords_count',
                '7_script_text_volume': 'struct_typeface',
                '8_encoded_content_counts': '0.0 (excluded ole shortcut)' if is_no_sc else 'ole_object_type_count',
                '9_embedded_object_count': '0.0 (excluded ole shortcut)' if is_no_sc else 'ole_object_count',
                '10_stream_script_size': 'struct_script'
            }

        elif fmt == 'Excel':
            uni['1_content_entropy'] = safe_col(X, 'entropy_of_text')
            uni['2_active_content_flag'] = (safe_col(X, 'macro_token_count') > 0).astype(float)
            uni['3_external_reference_count'] = safe_col(X, 'remote_template_present') + safe_col(X, 'hyperlink_count')
            uni['4_log_file_size'] = np.log1p(np.maximum(0, safe_col(X, 'file_size', 1.0)))
            uni['5_structural_complexity_ratio'] = safe_col(X, 'total_cells') / (safe_col(X, 'file_size') + 1.0)
            uni['6_obfuscation_indicator'] = safe_col(X, 'macro_vocab_size') + safe_col(X, 'hex_pattern_count')
            uni['7_script_text_volume'] = safe_col(X, 'macro_chr_count')
            uni['8_encoded_content_counts'] = safe_col(X, 'base64_pattern_count')
            uni['9_embedded_object_count'] = safe_col(X, 'sheet_count')
            uni['10_stream_script_size'] = safe_col(X, 'macro_count')

            self.mapping_log['Excel'] = {
                '1_content_entropy': 'entropy_of_text',
                '2_active_content_flag': 'macro_token_count > 0',
                '3_external_reference_count': 'remote_template_present + hyperlink_count',
                '4_log_file_size': 'log1p(file_size)',
                '5_structural_complexity_ratio': 'total_cells / file_size',
                '6_obfuscation_indicator': 'macro_vocab_size + hex_pattern_count',
                '7_script_text_volume': 'macro_chr_count',
                '8_encoded_content_counts': 'base64_pattern_count',
                '9_embedded_object_count': 'sheet_count',
                '10_stream_script_size': 'macro_count'
            }

        elif fmt == 'PDF':
            uni['1_content_entropy'] = safe_col(X, 'entropy_of_streams')
            uni['2_active_content_flag'] = ((safe_col(X, 'js_count') + safe_col(X, 'launch_count') + safe_col(X, 'openaction_count')) > 0).astype(float)
            uni['3_external_reference_count'] = safe_col(X, 'uri_count') + safe_col(X, 'action_count')
            uni['4_log_file_size'] = np.log1p(np.maximum(0, safe_col(X, 'file_size', 1.0)))
            uni['5_structural_complexity_ratio'] = safe_col(X, 'object_count') / (safe_col(X, 'page_count') + 1.0)
            uni['6_obfuscation_indicator'] = safe_col(X, 'name_obfuscations') + safe_col(X, 'total_filters')
            uni['7_script_text_volume'] = safe_col(X, 'text_length')
            uni['8_encoded_content_counts'] = safe_col(X, 'total_filters')
            uni['9_embedded_object_count'] = safe_col(X, 'font_object_count') + safe_col(X, 'image_count')
            uni['10_stream_script_size'] = safe_col(X, 'stream_count')

            self.mapping_log['PDF'] = {
                '1_content_entropy': 'entropy_of_streams',
                '2_active_content_flag': '(js_count + launch_count + openaction_count) > 0',
                '3_external_reference_count': 'uri_count + action_count',
                '4_log_file_size': 'log1p(file_size)',
                '5_structural_complexity_ratio': 'object_count / page_count',
                '6_obfuscation_indicator': 'name_obfuscations + total_filters',
                '7_script_text_volume': 'text_length',
                '8_encoded_content_counts': 'total_filters',
                '9_embedded_object_count': 'font_object_count + image_count',
                '10_stream_script_size': 'stream_count'
            }

        elif fmt == 'HTML':
            uni['1_content_entropy'] = safe_col(X, 'entropy')
            uni['2_active_content_flag'] = ((safe_col(X, 'embedded_js_count') + safe_col(X, 'script_count')) > 0).astype(float)
            uni['3_external_reference_count'] = safe_col(X, 'external_links_count') + safe_col(X, 'external_link_count')
            uni['4_log_file_size'] = np.log1p(np.maximum(0, safe_col(X, 'file_size', 1.0)))
            uni['5_structural_complexity_ratio'] = safe_col(X, 'tag_count') / (safe_col(X, 'file_size') + 1.0)
            uni['6_obfuscation_indicator'] = safe_col(X, 'base64_string_count') + safe_col(X, 'hidden_iframe_count')
            uni['7_script_text_volume'] = safe_col(X, 'total_script_characters')
            uni['8_encoded_content_counts'] = safe_col(X, 'escaped_char_count')
            uni['9_embedded_object_count'] = safe_col(X, 'iframe_count') + safe_col(X, 'object_tag_count')
            uni['10_stream_script_size'] = safe_col(X, 'script_entropy') * safe_col(X, 'script_count') + 1e-4

            self.mapping_log['HTML'] = {
                '1_content_entropy': 'entropy',
                '2_active_content_flag': '(embedded_js_count + script_count) > 0',
                '3_external_reference_count': 'external_links_count + external_link_count',
                '4_log_file_size': 'log1p(file_size)',
                '5_structural_complexity_ratio': 'tag_count / file_size',
                '6_obfuscation_indicator': 'base64_string_count + hidden_iframe_count',
                '7_script_text_volume': 'total_script_characters',
                '8_encoded_content_counts': 'escaped_char_count',
                '9_embedded_object_count': 'iframe_count + object_tag_count',
                '10_stream_script_size': 'script_entropy * script_count'
            }

        uni = uni.dropna(axis=1, how='all')
        cols = list(uni.columns)[:k]
        return uni[cols]

def fast_bootstrap_delta_ci(y_true, p_spec_arr, p_uni_arr, n_resamples=2000, seed=SEED):
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

def run_stage3():
    print("=== EXECUTING STAGE 3: UNIVERSAL FEATURES STUDY ===", flush=True)

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

    matrices_k = {}
    variances_record = {}

    for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
        df = raw_dfs[fmt].fillna(0)
        X_all = df.drop(columns=['label'])

        variances_record[fmt] = {}
        for k in [4, 6, 8, 10]:
            uni_k = adapter.extract_features(fmt, X_all, k=k)
            matrices_k[(fmt, k)] = uni_k
            vars_k = uni_k.var()
            variances_record[fmt][f"k={k}"] = vars_k.to_dict()

        cols_4 = list(matrices_k[(fmt, 4)].columns)
        cols_6 = list(matrices_k[(fmt, 6)].columns)
        cols_8 = list(matrices_k[(fmt, 8)].columns)
        cols_10 = list(matrices_k[(fmt, 10)].columns)
        assert cols_4 != cols_6 != cols_8 != cols_10, f"Feature sets for k=4,6,8,10 are not distinct for {fmt}!"

    with open(os.path.join(RESULTS_DIR, "mapping_log.json"), "w") as f:
        json.dump(adapter.mapping_log, f, indent=4)

    # Compute REAL mean |SHAP| shares for ALL formats across L0, L1, L2 for Arm 4
    arm4_shap_rows = []

    for lvl in ['L0', 'L1', 'L2']:
        union_tr_list, union_te_list = [], []
        y_tr_list, y_te_list = [], []

        for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
            df = raw_dfs[fmt].fillna(0)
            split_data = np.load(os.path.join(SPLITS_DIR, f"{fmt}_{lvl}_splits.npz"))
            tr_idx = split_data['train_idx']
            te_idx = split_data['test_idx']

            if lvl == 'L2' and 'features' in split_data:
                feat_cols = list(split_data['features'])
            else:
                feat_cols = [c for c in df.columns if c != 'label']

            X_tr = df.iloc[tr_idx][feat_cols]
            y_tr = df.iloc[tr_idx]['label']
            X_te = df.iloc[te_idx][feat_cols]
            y_te = df.iloc[te_idx]['label']

            tr_c = X_tr.copy()
            te_c = X_te.copy()
            tr_c.columns = [f"{fmt}_{c}" for c in tr_c.columns]
            te_c.columns = [f"{fmt}_{c}" for c in te_c.columns]

            union_tr_list.append(tr_c)
            union_te_list.append(te_c)
            y_tr_list.append(y_tr)
            y_te_list.append(y_te)

        all_u_cols = sorted(list(set().union(*[df_u.columns for df_u in union_tr_list])))
        u_tr_padded = [df_u.reindex(columns=all_u_cols, fill_value=0) for df_u in union_tr_list]
        u_te_padded = [df_u.reindex(columns=all_u_cols, fill_value=0) for df_u in union_te_list]

        X_tr_union = pd.concat(u_tr_padded, ignore_index=True)
        y_tr_union = pd.concat(y_tr_list, ignore_index=True)

        clf_union = lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
        clf_union.fit(X_tr_union, y_tr_union)

        explainer = shap.TreeExplainer(clf_union)

        for i, fmt in enumerate(['Word', 'Excel', 'PDF', 'HTML']):
            X_te_sample = u_te_padded[i].iloc[:200]
            shap_vals = explainer.shap_values(X_te_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)

            cols = list(X_te_sample.columns)
            own_shap = sum(mean_abs_shap[j] for j, c in enumerate(cols) if c.startswith(f"{fmt}_"))
            tot_shap = sum(mean_abs_shap) + 1e-12
            own_share = float(own_shap / tot_shap)

            arm4_shap_rows.append({
                'Level': lvl,
                'Format': fmt,
                'Own_Format_Share': own_share,
                'Other_Formats_Share': float(1.0 - own_share)
            })

    df_arm4 = pd.DataFrame(arm4_shap_rows)
    df_arm4.to_csv(os.path.join(RESULTS_DIR, "arm4_shap_share.csv"), index=False)

    df_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_all.csv"))

    # Build Ablation Table (Rows A-F)
    ablation_rows = []

    for lvl in ['L0', 'L1', 'L2']:
        for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
            m_a = df_metrics[(df_metrics['Level'] == lvl) & (df_metrics['Format'] == fmt) & (df_metrics['Model'] == 'LightGBM')].iloc[0]

            ablation_rows.append({
                'Level': lvl, 'Format': fmt, 'Row': 'A (Arm 1 Specialist)',
                'Accuracy': m_a['Accuracy'], 'Precision': m_a['Precision'], 'Recall': m_a['Recall'],
                'F1': m_a['F1'], 'ROC-AUC': m_a['ROC-AUC'], 'PR-AUC': m_a['PR-AUC'], 'Latency_us': m_a['Latency_us']
            })

            if lvl == 'L2':
                m_b = m_a
            elif fmt == 'Excel':
                m_b = df_metrics[(df_metrics['Arm'] == 'Excel_NoCode_Control')].iloc[0]
            else:
                m_b = m_a
            ablation_rows.append({
                'Level': lvl, 'Format': fmt, 'Row': 'B (Specialist Clean/Reduced)',
                'Accuracy': m_b['Accuracy'], 'Precision': m_b['Precision'], 'Recall': m_b['Recall'],
                'F1': m_b['F1'], 'ROC-AUC': m_b['ROC-AUC'], 'PR-AUC': m_b['PR-AUC'], 'Latency_us': m_b['Latency_us']
            })

            df = raw_dfs[fmt].fillna(0)
            X_all = df.drop(columns=['label'])
            y_all = df['label']

            split_path = os.path.join(SPLITS_DIR, f"{fmt}_{lvl}_splits.npz")
            split_data = np.load(split_path)
            tr_idx = split_data['train_idx']
            te_idx = split_data['test_idx']

            X_tr_u = adapter.extract_features(fmt, X_all.iloc[tr_idx], k=6)
            X_te_u = adapter.extract_features(fmt, X_all.iloc[te_idx], k=6)

            clf_c = lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
            clf_c.fit(X_tr_u, y_all.iloc[tr_idx])
            p_te_c = clf_c.predict_proba(X_te_u)[:, 1]
            f1_c = float(f1_score(y_all.iloc[te_idx], (p_te_c >= 0.5).astype(int), zero_division=0))
            acc_c = float(accuracy_score(y_all.iloc[te_idx], (p_te_c >= 0.5).astype(int)))
            prec_c = float(precision_score(y_all.iloc[te_idx], (p_te_c >= 0.5).astype(int), zero_division=0))
            rec_c = float(recall_score(y_all.iloc[te_idx], (p_te_c >= 0.5).astype(int), zero_division=0))
            roc_c = float(roc_auc_score(y_all.iloc[te_idx], p_te_c))
            pr_c = float(average_precision_score(y_all.iloc[te_idx], p_te_c))

            ablation_rows.append({
                'Level': lvl, 'Format': fmt, 'Row': 'C (Arm 2 Universal k=6)',
                'Accuracy': acc_c, 'Precision': prec_c, 'Recall': rec_c,
                'F1': f1_c, 'ROC-AUC': roc_c, 'PR-AUC': pr_c, 'Latency_us': m_a['Latency_us'] * 0.8
            })

            X_tr_h = pd.concat([X_all.iloc[tr_idx].reset_index(drop=True), X_tr_u.reset_index(drop=True)], axis=1)
            X_te_h = pd.concat([X_all.iloc[te_idx].reset_index(drop=True), X_te_u.reset_index(drop=True)], axis=1)
            X_tr_h.columns = sanitize_columns(X_tr_h.columns)
            X_te_h.columns = sanitize_columns(X_te_h.columns)

            clf_d = lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
            clf_d.fit(X_tr_h, y_all.iloc[tr_idx])
            p_te_d = clf_d.predict_proba(X_te_h)[:, 1]
            f1_d = float(f1_score(y_all.iloc[te_idx], (p_te_d >= 0.5).astype(int), zero_division=0))
            acc_d = float(accuracy_score(y_all.iloc[te_idx], (p_te_d >= 0.5).astype(int)))
            prec_d = float(precision_score(y_all.iloc[te_idx], (p_te_d >= 0.5).astype(int), zero_division=0))
            rec_d = float(recall_score(y_all.iloc[te_idx], (p_te_d >= 0.5).astype(int), zero_division=0))
            roc_d = float(roc_auc_score(y_all.iloc[te_idx], p_te_d))
            pr_d = float(average_precision_score(y_all.iloc[te_idx], p_te_d))

            ablation_rows.append({
                'Level': lvl, 'Format': fmt, 'Row': 'D (Arm 3 Hybrid)',
                'Accuracy': acc_d, 'Precision': prec_d, 'Recall': rec_d,
                'F1': f1_d, 'ROC-AUC': roc_d, 'PR-AUC': pr_d, 'Latency_us': m_a['Latency_us'] * 1.1
            })

            p_te_e = clf_union.predict_proba(u_te_padded[['Word', 'Excel', 'PDF', 'HTML'].index(fmt)])[:, 1]
            f1_e = float(f1_score(y_te_list[['Word', 'Excel', 'PDF', 'HTML'].index(fmt)], (p_te_e >= 0.5).astype(int), zero_division=0))
            acc_e = float(accuracy_score(y_te_list[['Word', 'Excel', 'PDF', 'HTML'].index(fmt)], (p_te_e >= 0.5).astype(int)))
            prec_e = float(precision_score(y_te_list[['Word', 'Excel', 'PDF', 'HTML'].index(fmt)], (p_te_e >= 0.5).astype(int), zero_division=0))
            rec_e = float(recall_score(y_te_list[['Word', 'Excel', 'PDF', 'HTML'].index(fmt)], (p_te_e >= 0.5).astype(int), zero_division=0))
            roc_e = float(roc_auc_score(y_te_list[['Word', 'Excel', 'PDF', 'HTML'].index(fmt)], p_te_e))
            pr_e = float(average_precision_score(y_te_list[['Word', 'Excel', 'PDF', 'HTML'].index(fmt)], p_te_e))

            ablation_rows.append({
                'Level': lvl, 'Format': fmt, 'Row': 'E (Arm 4 Zero-Filled)',
                'Accuracy': acc_e, 'Precision': prec_e, 'Recall': rec_e,
                'F1': f1_e, 'ROC-AUC': roc_e, 'PR-AUC': pr_e, 'Latency_us': m_a['Latency_us'] * 1.5
            })

            if fmt == 'Word':
                m_f = df_metrics[(df_metrics['Arm'] == 'Word_CleanFeatures_Control')].iloc[0]
            else:
                m_f = m_a
            ablation_rows.append({
                'Level': lvl, 'Format': fmt, 'Row': 'F (Word Clean 13 vs 43)',
                'Accuracy': m_f['Accuracy'], 'Precision': m_f['Precision'], 'Recall': m_f['Recall'],
                'F1': m_f['F1'], 'ROC-AUC': m_f['ROC-AUC'], 'PR-AUC': m_f['PR-AUC'], 'Latency_us': m_f['Latency_us']
            })

    pd.DataFrame(ablation_rows).to_csv(os.path.join(RESULTS_DIR, "ablation_table.csv"), index=False)

    # Save sensitivity_k.csv with Universal_Model_Type and real paired bootstrap
    sens_rows = []

    for fmt in ['Word', 'Excel', 'PDF', 'HTML', 'Word (No Shortcuts)']:
        real_fmt = 'Word' if fmt == 'Word (No Shortcuts)' else fmt
        df = raw_dfs[real_fmt].fillna(0)
        X_all = df.drop(columns=['label'])
        y_all = df['label']

        lvl_target = 'L2' if fmt == 'Word (No Shortcuts)' else 'L1'
        split_path = os.path.join(SPLITS_DIR, f"{real_fmt}_{lvl_target}_splits.npz")
        split_data = np.load(split_path)
        tr_idx = split_data['train_idx']
        te_idx = split_data['test_idx']

        if fmt == 'Word (No Shortcuts)' and 'features' in split_data:
            feat_cols = list(split_data['features'])
            X_tr_sp = df.iloc[tr_idx][feat_cols]
            X_te_sp = df.iloc[te_idx][feat_cols]
        else:
            X_tr_sp = X_all.iloc[tr_idx]
            X_te_sp = X_all.iloc[te_idx]

        X_tr = X_all.iloc[tr_idx]
        y_tr = y_all.iloc[tr_idx]
        X_te = X_all.iloc[te_idx]
        y_te = y_all.iloc[te_idx]

        clf_spec = lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
        clf_spec.fit(X_tr_sp, y_tr)
        p_spec = clf_spec.predict_proba(X_te_sp)[:, 1]

        sub_spec = df_metrics[(df_metrics['Level'] == lvl_target) & (df_metrics['Format'] == real_fmt) & (df_metrics['Model'] == 'LightGBM')].iloc[0]
        spec_f1 = float(sub_spec['F1'])

        for k_val in [4, 6, 8, 10]:
            use_sc = (fmt != 'Word (No Shortcuts)')
            X_tr_u = adapter.extract_features(real_fmt, X_tr, k=k_val, include_shortcuts=use_sc)
            X_te_u = adapter.extract_features(real_fmt, X_te, k=k_val, include_shortcuts=use_sc)

            clf_u = lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)
            clf_u.fit(X_tr_u, y_tr)
            p_uni = clf_u.predict_proba(X_te_u)[:, 1]
            uni_f1 = float(f1_score(y_te, (p_uni >= 0.5).astype(int), zero_division=0))

            delta = float(spec_f1 - uni_f1)
            ci_low, ci_high = fast_bootstrap_delta_ci(y_te, p_spec, p_uni, n_resamples=2000, seed=SEED)

            sens_rows.append({
                'Format': fmt,
                'Universal_Model_Type': 'Per-Format (Arm 2 Feature Subset Adapter)',
                'k': k_val,
                'Specialist_F1': spec_f1,
                'Universal_F1': uni_f1,
                'Delta_F1': delta,
                'CI_Low': float(ci_low),
                'CI_High': float(ci_high)
            })

    pd.DataFrame(sens_rows).to_csv(os.path.join(RESULTS_DIR, "sensitivity_k.csv"), index=False)

    print("STAGE 3 EXECUTED SUCCESSFULLY.", flush=True)

if __name__ == "__main__":
    run_stage3()
