import os, sys, subprocess, textwrap
from pathlib import Path

REPO = Path.cwd() / "Nebula-Labs-VCRR-Demo"
SRC = REPO / "src"
CFG_DIR = REPO / "configs"
RESULTS_DIR = REPO / "results"
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

# Safety checks
if not REPO.exists():
    raise FileNotFoundError(f"Repo dir not found: {REPO}!")
if not SRC.exists():
    raise FileNotFoundError(f"Expected src/ in repo but not found: {SRC}!")
if not (SRC / "train.py").exists():
    raise FileNotFoundError(f"train.py not found in {SRC}!")

CFG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

run_order = [
    ("vcrr_exp1", "parity_tuned_vcrr_exp1.yaml"),
    ("vcrr_exp2", "parity_tuned_vcrr_exp2.yaml"),
    ("vcrr_exp3", "parity_tuned_vcrr_exp3.yaml"),
    ("vcrr_exp4", "parity_tuned_vcrr_exp4.yaml"),
    ("vcrr_exp5", "parity_tuned_vcrr_exp5.yaml"),
    ("vcrr_exp6", "parity_tuned_vcrr_exp6.yaml"),
    ("vcrr_exp7", "parity_tuned_vcrr_exp7.yaml"),
    ("vcrr_exp8", "parity_tuned_vcrr_exp8.yaml"),
    ("vcrr_exp9", "parity_tuned_vcrr_exp9.yaml"),
]

env = os.environ.copy()
env["PYTHONPATH"] = f"{str(SRC)}:{env.get('PYTHONPATH','')}"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"

summary = []
for label, cfgname in run_order:
    cfgpath = CFG_DIR / cfgname
    if not cfgpath.exists():
        print(f"[SKIP] {label} -> missing config {cfgpath}")
        continue
    print(f"\n=== RUN {label} (config: {cfgpath}) seeds={SEEDS} ===")
    for seed in SEEDS:
        outdir = RESULTS_DIR / f"{label}_seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)

        if (outdir / "metrics_all.json").exists() or (outdir / "run_info.json").exists():
            print(f"[SKIP] {outdir} already finished (metrics present).")
            summary.append((label, seed, "skipped"))
            continue

        cmd = [sys.executable, str(SRC / "train.py"),
               "--config", str(cfgpath), "--method", "vcrr", "--seed", str(seed), "--out", str(outdir)]
        print(f"-> running seed {seed} ... (writing {outdir/'run.log'})")
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
            log_text = proc.stdout or ""
            (outdir / "run.log").write_text(log_text)
            lines = log_text.splitlines()
            if lines:
                head = "\n".join(lines[:20])
                tail = "\n".join(lines[-10:]) if len(lines) > 10 else ""
                print("----- run.log head -----")
                print(head)
                if tail:
                    print("----- run.log tail -----")
                    print(tail)
            else:
                print("[no stdout captured]")
            rc = proc.returncode
            if rc != 0:
                print(f"[ERROR] {label} seed {seed} returned rc={rc}; see {outdir/'run.log'}")
                summary.append((label, seed, f"error(rc={rc})"))
            else:
                print(f"[DONE] {label} seed {seed}")
                summary.append((label, seed, "done"))
        except Exception as e:
            err = f"[EXCEPTION] running seed {seed}: {e}"
            print(err)
            (outdir / "run.log").write_text(err + "\n")
            summary.append((label, seed, "exception"))
    print(f"=== Finished {label} ===\n")

# summary
print("Summary (label, seed, status):")
for s in summary:
    print(" -", s)
print("\nAll runs attempted. Inspect results/*_seed*/run.log, run_info.json and metrics_all.json for details.")