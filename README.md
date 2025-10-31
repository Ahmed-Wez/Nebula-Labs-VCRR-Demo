# Nebula Labs — CAS reproducible demo (Split-CIFAR-100) — N=3 seeds

Purpose
-------
Minimal reproducible demo showing a baseline (EWC) and a toy CAS prototype on a Split-CIFAR-100 setup.
This is a fast proof-of-concept: reproduce the runs, view results, and compute the "forgetting reduction" metric.

What you will get
-----------------
- reproducible code (Dockerfile + run scripts)
- results recorded in `results/` (per-method, per-seed)
- `notebooks/analysis.py` (script-style notebook) to compute forgetting metric and produce summary
- 1-page pilot offer `pilot_offer.md` to send to partners

Quick commands (local, no Docker)
--------------------------------
1. Create a python venv:
   python3 -m venv venv
   source venv/bin/activate
2. Install:
   pip install -r requirements.txt
3. Run demo:
   bash run_demo.sh

Docker (recommended if you want isolation)
-----------------------------------------
# build
docker build -t nebula-cas-demo .
# run (this will create results/)
docker run --gpus all --rm -v $(pwd)/results:/app/results nebula-cas-demo

What the demo does
------------------
- Trains a small CNN sequentially on a Split-CIFAR-100 (5 tasks by default).
- Methods: 'ewc' (elastic weight consolidation) and 'cas' (toy surgical reconfiguration).
- Runs N seeds (default 3) for each method and logs per-task accuracies.
- `src/eval.py` computes the forgetting metric F per Chaudhry et al. and produces `results/summary_*` CSVs.

Reproducibility
---------------
- Each run seeds Python / numpy / torch RNGs. See `src/utils.py`.
- To reproduce, run `run_demo.sh` or run `python src/train.py --method ewc --seed 0` etc.

Notes
-----
This demo emphasizes reproducible pipeline & analysis. The CAS here is a prototype placeholder to demonstrate surgical reconfiguration mechanics; the research implementation should follow after Phase-1 validation.

Contact
-------
Ahmed (Founder) — add email/LinkedIn