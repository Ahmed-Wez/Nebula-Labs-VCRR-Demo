import os, sys, subprocess
from pathlib import Path

REPO = Path.cwd() / "Nebula-Labs-VCRR-Demo"
SRC = REPO / "src"
CFG_DIR = REPO / "configs"
RESULTS_DIR = REPO / "results"

if not REPO.exists() or not SRC.exists() or not (SRC / "train.py").exists():
    raise FileNotFoundError(f"Repo or train.py not found under {REPO}!")

env = os.environ.copy()
env["PYTHONPATH"] = f"{str(SRC)}:{env.get('PYTHONPATH','')}"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"

MODE = os.environ.get("MODE", "max_quality")
DEFAULT_CONFIG_CANDIDATES = []
if MODE == "max_quality":
    DEFAULT_CONFIG_CANDIDATES = ["configs/parity_final_long.yaml", "configs/parity_boosted.yaml", "configs/parity_tuned_vcrr_exp1.yaml"]
else:
    DEFAULT_CONFIG_CANDIDATES = ["configs/parity_final_robust.yaml", "configs/parity_boosted.yaml", "configs/parity_tuned_vcrr_exp1.yaml"]

# Seeds
N_SEEDS = int(os.environ.get("N_SEEDS", "10"))
SEEDS = list(range(N_SEEDS))

METHOD_ARG = {
    "vcrr": "vcrr",
    "ewc": "ewc",
    "hope": "hope",
    "gdumb": "hybrid",
    "icarl": "hybrid",
    "mir": "hybrid",
    "scr": "hybrid"
}

RUN_ORDER = ["vcrr", "ewc", "hope", "gdumb", "icarl", "mir", "scr"]

def choose_config_for(label):
    candidates = []
    candidates += [f"configs/parity_final_{label}.yaml", f"configs/parity_final_{label}.yml"]
    candidates += [f"configs/parity_tuned_{label}.yaml", f"configs/parity_boosted_{label}.yaml"]
    if label == "vcrr":
        candidates += [f"configs/parity_tuned_vcrr_exp1.yaml", f"configs/parity_tuned_vcrr_exp2.yaml", "configs/parity_boosted.yaml"]
    candidates += DEFAULT_CONFIG_CANDIDATES
    for c in candidates:
        p = REPO / c
        if p.exists():
            return p
    return None

print(f"Launcher MODE={MODE} N_SEEDS={N_SEEDS} - using REPO={REPO}")
for label in RUN_ORDER:
    method_arg = METHOD_ARG.get(label)
    if method_arg is None:
        print(f"[SKIP] unknown method mapping for {label}")
        continue
    cfg_path = choose_config_for(label)
    if cfg_path is None:
        print(f"[SKIP] {label} -> no config found (looked at candidates).")
        continue
    print(f"\n=== Launching label={label} using config={cfg_path} (method arg={method_arg}) ===")
    for seed in SEEDS:
        outdir = RESULTS_DIR / f"{label}_seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)
        if (outdir / "metrics_all.json").exists() or (outdir / "run_info.json").exists():
            print(f"[SKIP] {outdir} exists (already finished)")
            continue
        print(f">>> RUN {label} seed={seed} -> {outdir}")
        cmd = [sys.executable, str(SRC / "train.py"),
               "--config", str(cfg_path),
               "--method", method_arg,
               "--seed", str(seed),
               "--out", str(outdir)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
            (outdir / "run.log").write_text(proc.stdout or "")
            lines = (proc.stdout or "").splitlines()
            if lines:
                head = "\n".join(lines[:10])
                tail = "\n".join(lines[-5:]) if len(lines) > 10 else ""
                print("----- run.log preview (head) -----")
                print(head)
                if tail:
                    print("----- run.log preview (tail) -----")
                    print(tail)
            else:
                print("[no stdout captured]")
            if proc.returncode != 0:
                print(f"[ERROR] {label} seed {seed} failed (rc={proc.returncode}) — see {outdir/'run.log'}")
            else:
                print(f"[DONE] {label} seed {seed}")
        except Exception as e:
            err = f"[EXCEPTION] while running {label} seed {seed}: {e}"
            print(err)
            (outdir / "run.log").write_text(err + "\n")
        print("----")

print("\nAll requested experiments attempted. Inspect results/*_seed*/run.log, run_info.json and metrics_all.json for details.")