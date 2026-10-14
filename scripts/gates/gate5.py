import os
import sys
import pandas as pd
from stage5_explainability import run_stage5, RESULTS_DIR

def run_gate5():
    run_stage5()
    print("\n================ GATE 5 OUTPUT ================")

    lime_path = os.path.join(RESULTS_DIR, "html_errors_lime.csv")
    assert os.path.exists(lime_path), "html_errors_lime.csv missing!"

    df_lime = pd.read_csv(lime_path)
    print(f"LIME Rows Count: {len(df_lime)}")

    # Assert signed LIME weight columns exist
    for i in range(1, 6):
        assert f"Top_{i}_Feature" in df_lime.columns, f"Top_{i}_Feature missing!"
        assert f"Top_{i}_Weight" in df_lime.columns, f"Top_{i}_Weight missing!"

    print("\nTop 5 rows of html_errors_lime.csv:")
    print(df_lime.head(5).to_string(index=False))

    print("\n================ GATE 5 PASSED ================\n")

if __name__ == "__main__":
    run_gate5()
