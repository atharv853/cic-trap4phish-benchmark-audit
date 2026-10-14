import os
import sys
import json
import pandas as pd
from stage3_universal import run_stage3, RESULTS_DIR

def run_gate3():
    run_stage3()
    print("\n================ GATE 3 OUTPUT ================")

    mapping_log_path = os.path.join(RESULTS_DIR, "mapping_log.json")
    assert os.path.exists(mapping_log_path), "mapping_log.json missing!"

    with open(mapping_log_path) as f:
        mapping_log = json.load(f)

    assert len(mapping_log) > 0, "mapping_log.json is empty!"
    for fmt in ['Word', 'Excel', 'PDF', 'HTML']:
        assert fmt in mapping_log, f"Missing mapping log for {fmt}!"

    print("mapping_log.json content:")
    print(json.dumps(mapping_log, indent=2))

    df_sens = pd.read_csv(os.path.join(RESULTS_DIR, "sensitivity_k.csv"))
    print("\nSensitivity study across k in {4, 6, 8, 10}:")
    print(df_sens[['Format', 'k', 'Specialist_F1', 'Universal_F1', 'Delta_F1', 'CI_Low', 'CI_High']].to_string(index=False))

    print("\n================ GATE 3 PASSED ================\n")

if __name__ == "__main__":
    run_gate3()
