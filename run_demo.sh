#!/usr/bin/env bash
set -euo pipefail

# Config
SEEDS=(0 1 2)
METHODS=("ewc" "cas")
OUTDIR="results"
CONFIG="configs/demo.yaml"

mkdir -p ${OUTDIR}

echo "Running demo: methods=${METHODS[*]}, seeds=${SEEDS[*]}"

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== Running method=${method} seed=${seed} ==="
    python src/train.py --config ${CONFIG} --method ${method} --seed ${seed} --out ${OUTDIR}/${method}_seed${seed}
  done
done

echo "All runs finished. Now computing summaries..."
python src/eval.py --results_dir ${OUTDIR} --out ${OUTDIR}/summary_all.csv

echo "Demo complete. See results/ for per-run outputs and summary CSV."