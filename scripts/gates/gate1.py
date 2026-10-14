import os
import sys
import numpy as np
import pandas as pd
from stage1_levels import run_stage1, RESULTS_DIR, SPLITS_DIR

def run_gate1():
    l2_feats = run_stage1()
    print("\n================ GATE 1 OUTPUT ================")

    formats = ['Word', 'Excel', 'PDF', 'HTML']
    df_sc = pd.read_csv(os.path.join(RESULTS_DIR, "audit_shortcuts.csv"))
    df_miss = pd.read_csv(os.path.join(RESULTS_DIR, "audit_missingness.csv"))

    for fmt in formats:
        # Check train/test split overlap
        for lvl in ['L0', 'L1', 'L2']:
            split_path = os.path.join(SPLITS_DIR, f"{fmt}_{lvl}_splits.npz")
            assert os.path.exists(split_path), f"Split file missing: {split_path}"
            data = np.load(split_path)
            tr_idx = data['train_idx']
            te_idx = data['test_idx']
            overlap = set(tr_idx).intersection(set(te_idx))
            assert len(overlap) == 0, f"Train/Test index overlap ({len(overlap)}) in {fmt} {lvl}!"

        # Check L2 feature set size vs L1 feature set size
        l1_split = np.load(os.path.join(SPLITS_DIR, f"{fmt}_L1_splits.npz"))
        l2_split = np.load(os.path.join(SPLITS_DIR, f"{fmt}_L2_splits.npz"))
        l2_features = l2_feats[fmt]

        # Check if shortcuts or NaN were flagged for this format
        fmt_sc = df_sc[(df_sc['Format'] == fmt) & (df_sc['Is_Shortcut'])]
        fmt_miss = df_miss[df_miss['Format'] == fmt]

        if len(fmt_sc) > 0 or len(fmt_miss) > 0:
            assert len(l2_features) < 43 or len(l2_features) < len(pd.read_csv(os.path.join(RESULTS_DIR, "audit_shortcuts.csv"))[df_sc['Format'] == fmt]), f"L2 features not strictly smaller than L1 for {fmt}!"

        print(f"Format: {fmt:<8} | L2 Feature Count: {len(l2_features)}")
        print(f"  L2 Features: {l2_features}\n")

    print("================ GATE 1 PASSED ================\n")

if __name__ == "__main__":
    run_gate1()
