import os
import json
import argparse
import numpy as np
import pandas as pd

def forgetting_metric(acc_matrix):
    T = len(acc_matrix)
    if T <= 1:
        return 0.0
    arr = np.full((T, T), np.nan)
    for t, row in enumerate(acc_matrix):
        for i, val in enumerate(row):
            arr[t, i] = val
    F_list = []
    for i in range(T-1):
        max_k = np.nanmax(arr[:T-1, i]) 
        a_T_i = arr[T-1, i]
        if np.isnan(a_T_i) or np.isnan(max_k):
            continue
        F_list.append(max_k - a_T_i)
    if len(F_list) == 0:
        return 0.0
    F = (1.0 / (T - 1)) * sum(F_list)
    return F

def load_metrics_from_folder(folder):
    path = os.path.join(folder, "metrics_all.json")
    if not os.path.exists(path):
        matrices = []
        files = sorted([f for f in os.listdir(folder) if f.startswith("metrics_task") and f.endswith(".json")])
        for f in files:
            with open(os.path.join(folder, f), 'r') as fh:
                j = json.load(fh)
                matrices.append(j.get('accs', []))
        return matrices
    with open(path, 'r') as fh:
        j = json.load(fh)
        return j.get('acc_matrix', [])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results', help='top-level results dir that contains method_seedX folders')
    parser.add_argument('--out', type=str, default='results/summary_all.csv')
    args = parser.parse_args()

    rows = []
    for entry in sorted(os.listdir(args.results_dir)):
        p = os.path.join(args.results_dir, entry)
        if os.path.isdir(p) and ("ewc" in entry or "vcrr" in entry):
            acc_matrix = load_metrics_from_folder(p)
            F = forgetting_metric(acc_matrix)
            rows.append({
                'run': entry,
                'F': float(F)
            })
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Wrote summary to", args.out)
    print(df)