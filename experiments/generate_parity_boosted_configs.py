from pathlib import Path
import textwrap

REPO = Path.cwd() / "Nebula-Labs-VCRR-Demo"
CFG_DIR = REPO / "configs"
CFG_DIR.mkdir(parents=True, exist_ok=True)

parity_boosted = textwrap.dedent("""\
dataset:
  name: cifar100
  num_tasks: 5
  classes_per_task: 20

training:
  model: resnet34
  epochs_per_task: 16
  batch_size: 128
  lr: 0.01
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5e-4
  scheduler:
    step_size: 4
    gamma: 0.5
  mixup_alpha: 0.1
  augment: True

hybrid:
  use_replay: true
  exemplar_per_class: 50
  replay_fraction: 0.2
  use_ewc_in_hybrid: false

vcrr:
  reconfig_k: 64
  soft_alpha: 0.0
  apply_to: all_linear
  randomized_svd: false

hope:
  eta: 0.01
  apply_to: fc_only
  use_with_optimizer: true
  normalize_inputs: true

ewc:
  lambda_ewc: 1000.0
""")

derived = {
    "parity_boosted.yaml": parity_boosted,
    "parity_boosted_gdumb.yaml": "include: parity_boosted.yaml\n\nhybrid:\n  use_replay: true\n  exemplar_per_class: 50\n  replay_fraction: 0.0\n\ngdumb:\n  enabled: true\n  train_only_at_end: true\n",
    "parity_boosted_icarl.yaml": "include: parity_boosted.yaml\n\nhybrid:\n  use_replay: true\n  exemplar_per_class: 50\n  replay_fraction: 0.0\n\nicarl:\n  enabled: true\n  nearest_exemplar: true\n  herding: true\n",
    "parity_boosted_mir.yaml": "include: parity_boosted.yaml\n\nhybrid:\n  use_replay: true\n  exemplar_per_class: 50\n  replay_fraction: 0.2\n\nmir:\n  enabled: true\n  lookahead_updates: 1\n  score_by_loss_increase: true\n",
    "parity_boosted_scr.yaml": "include: parity_boosted.yaml\n\nhybrid:\n  use_replay: true\n  exemplar_per_class: 50\n  replay_fraction: 0.2\n\nscr:\n  enabled: true\n  selection_mode: \"diversity\"\n",
}

written = []
for name, body in derived.items():
    p = CFG_DIR / name
    if not p.exists():
        p.write_text(textwrap.dedent(body).lstrip())
        written.append(name)

print("Config directory:", CFG_DIR)
if written:
    print("Wrote configs:", written)
else:
    print("All expected configs already present.")