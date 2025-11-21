import os, sys, subprocess
from pathlib import Path

REPO = Path.cwd() / "Nebula-Labs-VCRR-Demo"
SRC = REPO / "src"
if not REPO.exists():
    raise FileNotFoundError(f"{REPO} missing; run setup cells first.")

env = os.environ.copy()
env["PYTHONPATH"] = f"{str(SRC)}:{env.get('PYTHONPATH','')}"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"

# Set seeds
N_SEEDS = 10
seeds = range(N_SEEDS)

METHOD = {
    "vcrr": "vcrr",
    "ewc": "ewc",
    "gdumb": "hybrid",
    "hope": "hope",
    "icarl": "hybrid",
    "mir": "hybrid",
    "scr": "hybrid"
}

CONFIG = {
    "vcrr": "configs/parity_boosted_vcrr.yaml",
    "ewc": "configs/parity_boosted.yaml",
    "gdumb": "configs/parity_boosted_gdumb.yaml",
    "hope": "configs/parity_boosted.yaml",
    "icarl": "configs/parity_boosted_icarl.yaml",
    "mir": "configs/parity_boosted_mir.yaml",
    "scr": "configs/parity_boosted_scr.yaml"
}

RUN_ORDER = ["vcrr", "ewc", "gdumb", "hope", "icarl", "mir", "scr"]

print("Starting BOOSTED parity verification (N={}) for: {}".format(N_SEEDS, ", ".join(RUN_ORDER)))
for label in RUN_ORDER:
    cfg = REPO / CONFIG.get(label, "")
    method = METHOD.get(label)
    if not cfg.exists():
        print(f"[SKIP] {label} -> config missing: {cfg}")
        continue
    for seed in seeds:
        outdir = REPO / "results" / f"{label}_seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)
        if (outdir / "metrics_all.json").exists():
            print(f"[SKIP] {outdir} exists (metrics_all.json)")
            continue
        print(f">>> RUN {label} (method={method}) seed={seed} -> {outdir}")
        cmd = [sys.executable, str(SRC / "train.py"),
               "--config", str(cfg), "--method", method, "--seed", str(seed), "--out", str(outdir)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        (outdir / "run.log").write_text(proc.stdout or "")
        print((proc.stdout or "")[:2000])
        if proc.returncode != 0:
            print(f"[ERROR] {label} seed {seed} failed (rc={proc.returncode}) — see {outdir/'run.log'}")
        else:
            print(f"[DONE] {label} seed {seed}")
        print("----")

print("All requested parity runs attempted. Inspect results/*_seed*/run.log and metrics_all.json for details.")