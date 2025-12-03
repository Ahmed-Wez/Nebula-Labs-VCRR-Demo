import os, sys, subprocess, time, json
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

# Define all benchmarks and their corresponding configs
BENCHMARKS = {
    'cifar10': ['cifar10'],
    'cifar100': ['parity_boosted', 'parity_tuned_vcrr_exp1'],
    'permuted_mnist': ['permuted_mnist'],
    'tinyimagenet': ['tinyimagenet'],
    'core50': ['core50']
}

# Define all methods and their method_arg mapping
METHODS = {
    'baseline': 'baseline',
    'ewc': 'ewc',
    'lwf': 'lwf',
    'er': 'er',
    'agem': 'agem',
    'der': 'der',
    'packnet': 'packnet',
    'prognn': 'prognn',
    'hope': 'hope',
    'icarl': 'icarl',
    'gdumb': 'gdumb',
    'mir': 'mir',
    'scr': 'scr',
    'vcrr_exp1': 'vcrr_exp1'
}

# Build complete run list
runs = []

# Add all benchmark + method combinations
for dataset, cfg_prefixes in BENCHMARKS.items():
    for method, method_arg in METHODS.items():
        # Try to find config file
        config_found = None
        
        # Special handling for VCRR experiments on cifar100
        if method.startswith('vcrr_exp') and dataset == 'cifar100':
            for prefix in cfg_prefixes:
                if 'vcrr' in prefix:
                    cfg_path = CFG_DIR / f"{prefix}.yaml"
                    if cfg_path.exists():
                        config_found = cfg_path
                        break
        else:
            # Standard config search
            cfg_path = CFG_DIR / f"{dataset}_{method}.yaml"
            if cfg_path.exists():
                config_found = cfg_path
        
        if config_found:
            runs.append({
                "label": f"{dataset}_{method}",
                "method_arg": method_arg,
                "config": config_found,
                "dataset": dataset
            })

# Check for missing configs
missing_cfgs = [r for r in runs if not r["config"].exists()]
if missing_cfgs:
    print("WARNING: Some config files are missing:")
    for r in missing_cfgs:
        print(f"  - {r['label']} -> {r['config']}")
    print("These runs will be skipped.\n")

# Environment setup
env = os.environ.copy()
env["PYTHONPATH"] = f"{str(SRC)}:{env.get('PYTHONPATH','')}"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"

def is_done(outdir: Path):
    return (outdir / "metrics_all.json").exists() or (outdir / "run_info.json").exists()

# Summary statistics
total_runs = len([r for r in runs if r["config"].exists()]) * N_SEEDS
print(f"╔═══════════════════════════════════════════════════════╗")
print(f"║  COMPREHENSIVE CONTINUAL LEARNING BENCHMARK SUITE    ║")
print(f"╚═══════════════════════════════════════════════════════╝")
print(f"\nDatasets: {', '.join(BENCHMARKS.keys())}")
print(f"Methods: {', '.join(METHODS.keys())}")
print(f"Seeds per run: {N_SEEDS}")
print(f"Total experiments: {total_runs}\n")

time.sleep(1)

# Track statistics
completed = 0
skipped = 0
failed = 0

for r in runs:
    label = r["label"]
    cfg = r["config"]
    method_arg = r["method_arg"]
    dataset = r["dataset"]
    
    if not cfg.exists():
        print(f"[SKIP] {label}: config missing {cfg}")
        continue
    
    print(f"\n{'='*70}")
    print(f"RUN: {label}")
    print(f"  Dataset: {dataset}")
    print(f"  Method: {method_arg}")
    print(f"  Config: {cfg.name}")
    print(f"{'='*70}")
    
    for seed in SEEDS:
        outdir = OUT_ROOT / f"{label}_seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)
        
        if is_done(outdir):
            print(f"  [SKIP] seed {seed} - already done")
            skipped += 1
            continue
        
        cmd = [
            sys.executable, str(SRC / "train.py"),
            "--config", str(cfg),
            "--method", method_arg,
            "--seed", str(seed),
            "--output_dir", str(outdir)
        ]
        
        print(f"  [RUN] seed {seed} -> {outdir.name}")
        
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                timeout=7200  # 2 hour timeout per run
            )
            
            # Save log
            log_text = proc.stdout or ""
            (outdir / "run.log").write_text(log_text)
            
            if proc.returncode != 0:
                print(f"    [ERROR] Failed with return code {proc.returncode}")
                print(f"    Check log: {outdir/'run.log'}")
                failed += 1
                
                # Save error info
                error_info = {
                    "returncode": proc.returncode,
                    "label": label,
                    "seed": seed,
                    "log_excerpt": log_text[-1000:] if log_text else "No output"
                }
                (outdir / "error_info.json").write_text(json.dumps(error_info, indent=2))
            else:
                print(f"    [DONE] seed {seed}")
                completed += 1
                
        except subprocess.TimeoutExpired:
            print(f"    [TIMEOUT] seed {seed} - exceeded 2 hours")
            (outdir / "run.log").write_text(f"TIMEOUT after 2 hours\n")
            failed += 1
        except Exception as e:
            print(f"    [EXCEPTION] seed {seed}: {e}")
            (outdir / "run.log").write_text(f"EXCEPTION: {e}\n")
            failed += 1

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Completed: {completed}")
print(f"Skipped: {skipped}")
print(f"Failed: {failed}")
print(f"Total: {completed + skipped + failed}")
print(f"\nResults saved to: {OUT_ROOT}")
print(f"Run: python experiments/build_master_table.py to generate comparison table")