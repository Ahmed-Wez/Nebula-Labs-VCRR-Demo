from pathlib import Path
import sys

PYFILE = Path.cwd() / "Nebula-Labs-VCRR-Demo" / "src" / "train.py"
if not PYFILE.exists():
    raise FileNotFoundError(f"{PYFILE} not found!")

s = PYFILE.read_text()

start_anchor = "opt.zero_grad()\n                out = model(x_all)\n\n"
end_anchor = "                # EWC penalty (if configured)"

if start_anchor not in s or end_anchor not in s:
    print("ERROR: expected anchors not found in train.py; aborting patch.")
    print("Counts: opt.zero_grad() occurrences =", s.count("opt.zero_grad()"))
    idx = s.find("opt.zero_grad()")
    excerpt = s[idx: idx+400] if idx != -1 else "(no opt.zero_grad() excerpt)"
    print("Excerpt around first 'opt.zero_grad()':\n", excerpt)
    raise SystemExit(1)

start_idx = s.find(start_anchor)
end_idx = s.find(end_anchor, start_idx)
if end_idx == -1:
    print("ERROR: could not locate end anchor after start; aborting.")
    raise SystemExit(1)

replacement_block = (
    start_anchor +
    "                # --- unified mixup/replay label handling to avoid UnboundLocalError ---\n"
    "                if mixup_alpha > 0.0:\n"
    "                    # If replay merging didn't produce y_all_a/y_all_b, fall back to y_a/y_b\n"
    "                    try:\n"
    "                        _ = y_all_a\n"
    "                    except NameError:\n"
    "                        y_all_a = y_a\n"
    "                    try:\n"
    "                        _ = y_all_b\n"
    "                    except NameError:\n"
    "                        y_all_b = y_b\n"
    "                    # Now compute loss using the precomputed 'out'\n"
    "                    if y_all_b is None:\n"
    "                        # y_all_a may be a tensor or None\n"
    "                        if y_all_a is None:\n"
    "                            # last-resort fallback to original batch labels\n"
    "                            loss = loss_fn(out, yb.to(out.device))\n"
    "                        else:\n"
    "                            loss = mixup_criterion(loss_fn, out, y_all_a.to(out.device), None, lam)\n"
    "                    else:\n"
    "                        loss = mixup_criterion(loss_fn, out, y_all_a.to(out.device), y_all_b.to(out.device), lam)\n"
    "                else:\n"
    "                    # Non-mixup path: ensure unified y_all exists\n"
    "                    try:\n"
    "                        _ = y_all\n"
    "                    except NameError:\n"
    "                        y_all = yb\n"
    "                    loss = loss_fn(out, y_all.to(out.device))\n\n"
)

s_new = s[:start_idx] + replacement_block + s[end_idx:]
PYFILE.write_text(s_new)
print("Patched train.py successfully (mixup/replay block replaced).")