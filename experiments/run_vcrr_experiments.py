import subprocess
import sys
import os
import argparse
import textwrap
from pathlib import Path
from typing import List, Set


def parse_range_list(s: str, min_val: int = 1, max_val: int = 9) -> List[int]:
    s = (s or "").strip().lower()
    if not s:
        return []
    if s == "all":
        return list(range(min_val, max_val + 1))

    parts = [p.strip() for p in s.replace(",", " ").split()]
    nums: Set[int] = set()
    for p in parts:
        if "-" in p:
            try:
                a, b = p.split("-", 1)
                a_i = int(a); b_i = int(b)
                if a_i > b_i:
                    a_i, b_i = b_i, a_i
                for v in range(a_i, b_i + 1):
                    if min_val <= v <= max_val:
                        nums.add(v)
            except Exception:
                raise ValueError(f"Invalid range token: '{p}'")
        else:
            try:
                v = int(p)
                if min_val <= v <= max_val:
                    nums.add(v)
            except Exception:
                raise ValueError(f"Invalid token: '{p}'")
    return sorted(nums)


def find_config_file(cfg_dir: Path, exp_num: int) -> Path:
    candidates = [
        cfg_dir / f"parity_tuned_vcrr_exp{exp_num}.yaml",
        cfg_dir / f"parity_tuned_vcrr_exp{exp_num}.yml",
        cfg_dir / f"vcrr_exp{exp_num}.yaml",
        cfg_dir / f"vcrr_exp{exp_num}.yml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No config found for vcrr_exp{exp_num} in {cfg_dir} (checked {', '.join(str(x.name) for x in candidates)})")


def prompt_user(prompt: str, default: str = "") -> str:
    try:
        if default:
            return input(f"{prompt} [{default}]: ") or default
        else:
            return input(f"{prompt}: ")
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)


def run_job(python_exe: str, train_py: Path, cfg_path: Path, method: str, seed: int, outdir: Path, env_extra: dict = None):
    cmd = [python_exe, str(train_py), "--config", str(cfg_path), "--method", method, "--seed", str(seed), "--out", str(outdir)]
    print("RUN:", " ".join(cmd))
    env = os.environ.copy()
    repo_root = train_py.resolve().parents[1]
    env["PYTHONPATH"] = f"{str(repo_root / 'src')}:{env.get('PYTHONPATH','')}"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    if env_extra:
        env.update(env_extra)

    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, text=True)
    return proc.returncode


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Interactive launcher for VCRR experiments (vcrr_exp1..vcrr_exp9).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--n-seeds", type=int, default=None, help="Number of seeds to run (1..10). If omitted, will prompt interactively.")
    parser.add_argument("--vcrr", type=str, default=None,
                        help="VCRR experiments to run. Examples: '1 2 3', '1-3', 'all'. If omitted, will prompt.")
    parser.add_argument("--repo-root", type=str, default=None, help="Path to repo root (defaults to two levels above this script).")
    parser.add_argument("--method-arg", type=str, default="vcrr", help="Method argument passed to train.py (default: vcrr).")
    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python interpreter to use for subprocesses.")
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    inferred_repo_root = script_path.parents[1]
    repo_root = Path(args.repo_root) if args.repo_root else inferred_repo_root
    src_train = repo_root / "src" / "train.py"
    cfg_dir = repo_root / "configs"
    results_dir = repo_root / "results"

    if not src_train.exists():
        print(f"ERROR: could not find train.py at expected path: {src_train}")
        print("Make sure you run this script from inside your repository and that src/train.py exists.")
        sys.exit(2)
    if not cfg_dir.exists():
        print(f"ERROR: configs directory not found at: {cfg_dir}")
        sys.exit(2)
    results_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = args.n_seeds
    if n_seeds is None:
        while True:
            val = prompt_user("Enter number of seeds to run (1..10)", default="3")
            try:
                n_seeds = int(val)
                if 1 <= n_seeds <= 10:
                    break
                else:
                    print("Please enter an integer between 1 and 10.")
            except Exception:
                print("Invalid integer, try again.")
    else:
        if not (1 <= n_seeds <= 10):
            print("Argument --n-seeds must be in [1,10].")
            sys.exit(2)

    vcrr_input = args.vcrr
    if vcrr_input is None:
        default_hint = "all"
        vcrr_input = prompt_user(textwrap.dedent(
            "Which VCRR experiments to run?\n"
            " - Enter numbers (e.g. '1 2 3')\n"
            " - Ranges allowed (e.g. '1-4')\n"
            " - Comma-separated ok (e.g. '1,3,5-7')\n"
            " - 'all' to run 1..9"
        ), default=default_hint)

    try:
        vcrr_nums = parse_range_list(vcrr_input, min_val=1, max_val=9)
    except ValueError as e:
        print("Failed to parse VCRR selection:", e)
        sys.exit(2)
    if not vcrr_nums:
        print("No VCRR experiments selected. Exiting.")
        sys.exit(0)

    resolved = []
    for n in vcrr_nums:
        try:
            cfg_path = find_config_file(cfg_dir, n)
        except FileNotFoundError as e:
            print(f"WARNING: {e} -- this experiment will be skipped.")
            continue
        label = f"vcrr_exp{n}"
        resolved.append((n, label, cfg_path))
    if not resolved:
        print("No valid VCRR configs found for the requested experiments. Exiting.")
        sys.exit(0)

    print("\nSummary:")
    print(" Repo root:", repo_root)
    print(" Train script:", src_train)
    print(" Config directory:", cfg_dir)
    print(" Results directory:", results_dir)
    print(f" Number of seeds: {n_seeds} (seeds will be 0..{n_seeds-1})")
    print(" Experiments to run:")
    for n, label, cfg in resolved:
        print(f"  - {label} -> {cfg.name}")
    proceed = prompt_user("Proceed with these runs? (y/n)", default="y").strip().lower()
    if proceed not in ("y", "yes"):
        print("Aborted by user.")
        sys.exit(0)

    seeds = list(range(n_seeds))
    python_exe = args.python_exe
    method_arg = args.method_arg

    summary = []
    try:
        for (n, label, cfg_path) in resolved:
            for seed in seeds:
                outdir = results_dir / f"{label}_seed{seed}"
                if (outdir / "metrics_all.json").exists() or (outdir / "run_info.json").exists():
                    print(f"[SKIP] {outdir} already finished (metrics present).")
                    summary.append((label, seed, "skipped"))
                    continue
                print("\n=== RUN:", label, "seed=", seed, "config=", cfg_path.name, "===> out:", outdir)
                rc = run_job(python_exe=python_exe, train_py=src_train, cfg_path=cfg_path, method=method_arg, seed=seed, outdir=outdir)
                if rc == 0:
                    print(f"[DONE] {label} seed {seed}")
                    summary.append((label, seed, "done"))
                else:
                    print(f"[ERROR] {label} seed {seed} returned rc={rc} — check {outdir/'run.log'}")
                    summary.append((label, seed, f"error(rc={rc})"))
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting early.")
    finally:
        print("\nSummary (label, seed, status):")
        for s in summary:
            print(" -", s)
        print("\nLauncher finished. Inspect results/*_seed*/run.log, run_info.json and metrics_all.json for details.")

if __name__ == "__main__":
    main()