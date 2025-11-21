import os, re, json
from pathlib import Path
import pandas as pd
import numpy as np

RDIR = Path("Nebula-Labs-VCRR-Demo/results")
RDIR.mkdir(parents=True, exist_ok=True)

def load_json_safe(p):
    p = Path(p)
    if not p.exists():
        return None
    s = p.read_text(errors='ignore')
    try:
        return json.loads(s)
    except Exception:
        try:
            import yaml
            return yaml.safe_load(s)
        except Exception:
            return None

def parse_run_from_dir(d: Path):
    out = {"run": d.name}
    ri = load_json_safe(d / "run_info.json") or {}
    mm = load_json_safe(d / "metrics_all.json") or {}

    def getk(k, alt=()):
        for s in (mm, ri):
            if isinstance(s, dict) and k in s and s[k] is not None:
                return s[k]
        for a in alt:
            for s in (mm, ri):
                if isinstance(s, dict) and a in s and s[a] is not None:
                    return s[a]
        return None

    method = None
    if isinstance(ri, dict) and 'method' in ri and ri['method'] is not None:
        method = ri['method']
    if method is None:
        method = mm.get('method') if isinstance(mm, dict) else None
    out['method'] = str(method) if method is not None else None

    out['F'] = getk('F', alt=('forgetting','Forgetting'))
    out['avg_acc'] = getk('avg_acc', alt=('avg_accuracy','accuracy'))
    out['time_s'] = getk('time_s', alt=('total_time_s','train_time_s'))
    out['params'] = getk('params', alt=('param_count',))
    out['peak_mem_mb'] = getk('peak_mem_mb', alt=('peak_mem',))
    out['expected_steps'] = getk('expected_steps')
    out['initial_acc_task1'] = getk('initial_acc_task1', alt=('initial_acc_t1',))
    out['final_acc_task1'] = getk('final_acc_task1', alt=('final_acc_t1',))

    if out['avg_acc'] is None or out['F'] is None:
        for p in sorted(d.glob("metrics_task*.json")):
            jj = load_json_safe(p)
            if isinstance(jj, dict) and 'accs' in jj:
                try:
                    accs = jj['accs']
                    if out['initial_acc_task1'] is None and len(accs) > 0:
                        out['initial_acc_task1'] = float(accs[0])
                    if out['final_acc_task1'] is None and len(accs) > 0:
                        out['final_acc_task1'] = float(accs[-1])
                except Exception:
                    pass

    logp = d / "run.log"
    if logp.exists():
        try:
            txt = logp.read_text(errors='ignore')
            for token, k in [('F','F'), ('avg_acc','avg_acc'), ('time_s','time_s'),
                             ('params','params'), ('peak_mem_mb','peak_mem_mb')]:
                mm2 = re.search(rf"{re.escape(token)}\s*[=:\-]\s*([0-9.+-eE]+)", txt)
                if mm2 and out.get(k) is None:
                    try:
                        out[k] = float(mm2.group(1))
                    except Exception:
                        pass
            m7 = re.search(r"\bmethod[:=]\s*([A-Za-z0-9_+-]+)\b", txt)
            if m7 and not out.get('method'):
                out['method'] = m7.group(1)
        except Exception:
            pass

    for k in ['F','avg_acc','time_s','params','peak_mem_mb','initial_acc_task1','final_acc_task1','expected_steps']:
        v = out.get(k, None)
        try:
            out[k] = float(v) if v is not None else np.nan
        except Exception:
            out[k] = np.nan

    m = re.search(r"_seed(\d+)$", d.name)
    out['seed'] = int(m.group(1)) if m else np.nan
    out['run_prefix'] = str(d.name).split('_seed')[0]
    return out

candidate_dirs = sorted([p for p in RDIR.glob("*_seed*") if p.is_dir()])
rows = [parse_run_from_dir(d) for d in candidate_dirs]

master = pd.DataFrame(rows)
master_path = RDIR / "comparison_master_table.csv"
full_path = RDIR / "parity_full_table.csv"
master.to_csv(master_path, index=False)
master.to_csv(full_path, index=False)
print(f"Wrote {full_path} and {master_path}; rows: {len(master)}")
print("Columns:", master.columns.tolist())