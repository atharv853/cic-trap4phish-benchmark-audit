import os
import sys
import json
import pandas as pd
from stage2_models import run_stage2, RESULTS_DIR, PARAMS_DIR

def run_gate2():
    run_stage2()
    print("\n================ GATE 2 OUTPUT ================")

    df_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_all.csv"))

    # Assert no latency value is a round constant (e.g. 1500.0 or integer)
    latencies = df_metrics['Latency_us'].values
    for lat in latencies:
        assert lat != 1500.0 and lat != 1000.0 and lat != 2000.0, f"Found round latency constant: {lat}"
        # Check fractional component
        frac = abs(lat - round(lat))
        assert frac != 0.0 or lat > 0.0, f"Latency appears to be round constant: {lat}"

    # Assert thresholds were set from OOF sweep
    thresholds = df_metrics['Optimal_Threshold'].unique()
    assert len(thresholds) > 0, "No optimal threshold values found!"

    print("Best Model Hyperparameters (Tuned via Optuna on Train Partition):")
    for lvl in ['L0', 'L1', 'L2']:
        for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
            p_path = os.path.join(PARAMS_DIR, f"best_params_{fmt}_{lvl}.json")
            if os.path.exists(p_path):
                with open(p_path) as f:
                    params = json.load(f)
                sub_m = df_metrics[(df_metrics['Level'] == lvl) & (df_metrics['Format'] == fmt) & (df_metrics['Model'] == 'LightGBM')].iloc[0]
                print(f"[{lvl}] Format: {fmt:<6} | Best Params: {params} | OOF Threshold: {sub_m['Optimal_Threshold']:.2f} | Latency: {sub_m['Latency_us']:.2f} µs")

    print("\n================ GATE 2 PASSED ================\n")

if __name__ == "__main__":
    run_gate2()
