import os, sys, subprocess, time
from pathlib import Path

REPO = Path.cwd() / "Nebula-Labs-VCRR-Demo"
SRC = REPO / "src"
CFG_DIR = REPO / "configs"
OUT_ROOT = REPO / "results"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

if not (SRC / "train.py").exists():
    raise FileNotFoundError(f"train.py not found at {SRC}/train.py!")

N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

vcrr_cfgs = [CFG_DIR / f"parity_tuned_vcrr_exp{i}.yaml" for i in range(1,10)]
vcrr_labels = [f"vcrr_exp{i}" for i in range(1,10)]

benchmarks = {
    "ewc": {"method_arg": "ewc", "config": CFG_DIR / "parity_boosted.yaml", "label": "ewc"},
    "hope": {"method_arg": "hope", "config": CFG_DIR / "parity_boosted.yaml", "label": "hope"},
    "icarl": {"method_arg": "hybrid", "config": CFG_DIR / "parity_boosted_icarl.yaml", "label": "icarl"},
    "gdumb": {"method_arg": "hybrid", "config": CFG_DIR / "parity_boosted_gdumb.yaml", "label": "gdumb"},
    "mir": {"method_arg": "hybrid", "config": CFG_DIR / "parity_boosted_mir.yaml", "label": "mir"},
    "scr": {"method_arg": "hybrid", "config": CFG_DIR / "parity_boosted_scr.yaml", "label": "scr"},
}

runs = []
for cfg, lab in zip(vcrr_cfgs, vcrr_labels):
    runs.append({"label": lab, "method_arg": "vcrr", "config": cfg})

for k,v in benchmarks.items():
    runs.append({"label": v["label"], "method_arg": v["method_arg"], "config": v["config"]})

missing_cfgs = [r for r in runs if not r["config"].exists()]
if missing_cfgs:
    print("WARNING: Some config files are missing. Missing list:")
    for r in missing_cfgs:
        print(" -", r["label"], "->", r["config"])
    print("The launcher will skip runs whose config files are missing.")

env = os.environ.copy()
env["PYTHONPATH"] = f"{str(SRC)}:{env.get('PYTHONPATH','')}"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"

def is_done(outdir: Path):
    if (outdir / "metrics_all.json").exists() or (outdir / "run_info.json").exists():
        return True
    return False

print(f"Launching experiments: {len(runs)} run types × {N_SEEDS} seeds = up to {len(runs)*N_SEEDS} jobs")
time.sleep(0.5)

for r in runs:
    label = r["label"]
    cfg = r["config"]
    method_arg = r["method_arg"]
    if not cfg.exists():
        print(f"[SKIP] {label}: config missing {cfg}")
        continue
    print("=== RUN TYPE:", label, "method_arg:", method_arg, "config:", cfg.name)
    for seed in SEEDS:
        outdir = OUT_ROOT / f"{label}_seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)
        if is_done(outdir):
            print(f"  [SKIP] {outdir} already done")
            continue
        cmd = [sys.executable, str(SRC / "train.py"),
               "--config", str(cfg),
               "--method", method_arg,
               "--seed", str(seed),
               "--out", str(outdir)]
        print("  >>>", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
            (outdir / "run.log").write_text(proc.stdout or "")
            if proc.returncode != 0:
                print(f"    [ERROR] seed {seed} rc={proc.returncode} (see {outdir/'run.log'})")
            else:
                print(f"    [DONE] {outdir.name}")
        except Exception as e:
            print(f"    [EXCEPTION] on seed {seed}: {e}")
    print("----")

print("Launcher completed. Re-run the master-table builder to refresh statistics.")