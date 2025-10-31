"""
analysis.py
Run after the demo to summarize results and compute forgetting reduction.

Usage:
python notebooks/analysis.py --results_dir results --output results/analysis_report.txt
"""

import argparse
import pandas as pd
import numpy as np

def run_analysis(results_dir="results", output="results/analysis_report.txt"):
    summary_csv = f"{results_dir}/summary_all.csv"
    df = pd.read_csv(summary_csv)
    # pivot: compute mean F per method across seeds
    df['method'] = df['run'].apply(lambda s: 'ewc' if 'ewc' in s else ('cas' if 'cas' in s else 'other'))
    grouped = df.groupby('method')['F'].agg(['mean','std','count']).reset_index()
    # forgetting reduction = 1 - (F_cas / F_ewc)
    ewc_mean = grouped.loc[grouped['method']=='ewc','mean'].values
    cas_mean = grouped.loc[grouped['method']=='cas','mean'].values
    if len(ewc_mean)==0 or len(cas_mean)==0:
        fr = None
    else:
        fr = 1.0 - (cas_mean[0] / ewc_mean[0]) if ewc_mean[0] != 0 else None

    with open(output, 'w') as f:
        f.write("Summary of demo runs (N seeds per method = 3 by default)\n")
        f.write(grouped.to_string(index=False))
        f.write("\n\nForgetting reduction (1 - F_cas / F_ewc) = {}\n".format(fr))
    print("Wrote analysis to", output)
    print(grouped)
    print("Forgetting reduction:", fr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output', type=str, default='results/analysis_report.txt')
    args = parser.parse_args()
    run_analysis(args.results_dir, args.output)