import os
import sys
import pandas as pd
from stage4_stats import run_stage4, RESULTS_DIR

def run_gate4():
    run_stage4()
    print("\n================ GATE 4 OUTPUT ================")

    df_sig = pd.read_csv(os.path.join(RESULTS_DIR, "transfer_gap_significance.csv"))
    print("Full Significance Table (Specialist vs Universal Pooled at Level L1):")
    print(df_sig.to_string(index=False))

    # Assert Nadeau-Bengio, Holm, McNemar and bootstrap columns exist
    required_cols = ['Format', 'Delta_F1', 'CI_Low', 'CI_High', 't_stat', 'dof', 'p_value', 'p_value_adjusted', 'mcnemar_p', 'cohens_d']
    for col in required_cols:
        assert col in df_sig.columns, f"Column {col} missing in transfer_gap_significance.csv!"

    # Assert Delta_F1 is inside its CI
    for _, row in df_sig.iterrows():
        assert row['CI_Low'] <= row['Delta_F1'] <= row['CI_High'], f"Delta_F1 ({row['Delta_F1']}) outside CI [{row['CI_Low']}, {row['CI_High']}] for {row['Format']}!"
        assert row['p_value_adjusted'] >= row['p_value'], f"Adjusted p-value ({row['p_value_adjusted']}) less than raw p-value ({row['p_value']}) for {row['Format']}!"

    print("\nHypotheses Re-evaluation:")
    df_hyp = pd.read_csv(os.path.join(RESULTS_DIR, "hypotheses.csv"))
    print(df_hyp.to_string(index=False))

    print("\n================ GATE 4 PASSED ================\n")

if __name__ == "__main__":
    run_gate4()
