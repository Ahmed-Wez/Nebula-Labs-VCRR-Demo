import argparse
import pandas as pd
import numpy as np

def run_analysis(results_dir="results", output="results/analysis_report.txt"):
    summary_csv = f"{results_dir}/summary_all.csv"
    df = pd.read_csv(summary_csv)
    df['method'] = df['run'].apply(lambda s: 'ewc' if 'ewc' in s else ('vcrr' if 'vcrr' in s else 'other'))
    grouped = df.groupby('method')['F'].agg(['mean','std','count']).reset_index()
    ewc_mean = grouped.loc[grouped['method']=='ewc','mean'].values
    vcrr_mean = grouped.loc[grouped['method']=='vcrr','mean'].values
    if len(ewc_mean)==0 or len(vcrr_mean)==0:
        fr = None
    else:
        fr = 1.0 - (vcrr_mean[0] / ewc_mean[0]) if ewc_mean[0] != 0 else None

    with open(output, 'w') as f:
        f.write("Summary of demo runs (N seeds per method = 3 by default)\n")
        f.write(grouped.to_string(index=False))
        f.write("\n\nForgetting reduction (1 - F_vcrr / F_ewc) = {}\n".format(fr))
    print("Wrote analysis to", output)
    print(grouped)
    print("Forgetting reduction:", fr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output', type=str, default='results/analysis_report.txt')
    args = parser.parse_args()
    run_analysis(args.results_dir, args.output)