import os, sys, subprocess, json
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

# Configuration
DATASET = os.environ.get("DATASET", "cifar100")  # cifar10, cifar100, permuted_mnist, tinyimagenet, core50
N_SEEDS = int(os.environ.get("N_SEEDS", "10"))
SEEDS = list(range(N_SEEDS))

# Method mapping
METHOD_ARG = {
    "baseline": "baseline",
    "ewc": "ewc",
    "lwf": "lwf",
    "er": "er",
    "agem": "agem",
    "der": "der",
    "packnet": "packnet",
    "prognn": "prognn",
    "hope": "hope",
    "icarl": "icarl",
    "gdumb": "gdumb",
    "mir": "mir",
    "scr": "scr",
    "vcrr": "vcrr_exp1"
}

# Run all methods for the specified dataset
RUN_ORDER = list(METHOD_ARG.keys())

def find_config(dataset, method):
    """Find config file for dataset + method combination"""
    candidates = [
        CFG_DIR / f"{dataset}_{method}.yaml",
        CFG_DIR / f"{dataset}_{method}.yml",
    ]
    
    # Special case for VCRR on CIFAR-100
    if method == "vcrr" and dataset == "cifar100":
        candidates.insert(0, CFG_DIR / "parity_tuned_vcrr_exp1.yaml")
        candidates.insert(1, CFG_DIR / "parity_boosted_vcrr.yaml")
    
    for c in candidates:
        if c.exists():
            return c
    return None

print(f"╔════════════════════════════════════════════════════╗")
print(f"║     ROBUST LAUNCHER - {DATASET.upper():^30} ║")
print(f"╚════════════════════════════════════════════════════╝")
print(f"Seeds: {N_SEEDS}")
print(f"Methods: {len(RUN_ORDER)}\n")

successful = 0
failed = 0
skipped = 0

for label in RUN_ORDER:
    method_arg = METHOD_ARG.get(label)
    if method_arg is None:
        print(f"[SKIP] {label} - unknown method")
        continue
    
    cfg_path = find_config(DATASET, label)
    if cfg_path is None:
        print(f"[SKIP] {label} - no config found for {DATASET}")
        skipped += 1
        continue
    
    print(f"\n{'='*60}")
    print(f"METHOD: {label} ({method_arg})")
    print(f"CONFIG: {cfg_path.name}")
    print(f"{'='*60}")
    
    for seed in SEEDS:
        outdir = RESULTS_DIR / f"{DATASET}_{label}_seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)
        
        if (outdir / "metrics_all.json").exists():
            print(f"  [SKIP] seed {seed} - already finished")
            skipped += 1
            continue
        
        print(f"  [RUN] seed {seed}...")
        
        cmd = [
            sys.executable, str(SRC / "train.py"),
            "--config", str(cfg_path),
            "--method", method_arg,
            "--seed", str(seed),
            "--output_dir", str(outdir)
        ]
        
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                timeout=7200
            )
            
            (outdir / "run.log").write_text(proc.stdout or "")
            
            if proc.returncode != 0:
                print(f"    [ERROR] rc={proc.returncode}")
                failed += 1
            else:
                print(f"    [DONE]")
                successful += 1
                
        except subprocess.TimeoutExpired:
            print(f"    [TIMEOUT]")
            (outdir / "run.log").write_text("TIMEOUT\n")
            failed += 1
        except Exception as e:
            print(f"    [EXCEPTION] {e}")
            (outdir / "run.log").write_text(f"EXCEPTION: {e}\n")
            failed += 1

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Skipped: {skipped}")
print(f"\nTo run different dataset: DATASET=cifar10 python experiments/robust_launcher.py")