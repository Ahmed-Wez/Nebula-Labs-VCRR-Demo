from pathlib import Path
import textwrap

REPO = Path.cwd() / "Nebula-Labs-VCRR-Demo"
CFG_DIR = REPO / "configs"
CFG_DIR.mkdir(parents=True, exist_ok=True)

configs = {
"parity_tuned_vcrr_exp1.yaml": """
dataset:
  name: cifar100
  num_tasks: 2
  classes_per_task: 10

training:
  model: resnet34
  epochs_per_task: 16
  batch_size: 128
  lr: 0.01
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5e-4
  augment: true
  mixup_alpha: 0.2
  label_smoothing: 0.1
  set_cudnn_benchmark: true
  scheduler:
    step_size: 3
    gamma: 0.5

hybrid:
  use_replay: true
  exemplar_per_class: 50
  replay_fraction: 0.1
  use_ewc_in_hybrid: false

vcrr:
  reconfig_k: 48
  soft_alpha: 0.1
  apply_to: all_linear
  randomized_svd: false

distill:
  use_lwf: true
  alpha: 0.5
""",
"parity_tuned_vcrr_exp2.yaml": """
dataset:
  name: cifar100
  num_tasks: 2
  classes_per_task: 10

training:
  model: resnet34
  epochs_per_task: 16
  batch_size: 128
  lr: 0.005
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5e-4
  augment: true
  mixup_alpha: 0.2
  label_smoothing: 0.1
  set_cudnn_benchmark: true
  scheduler:
    step_size: 3
    gamma: 0.5

hybrid:
  use_replay: true
  exemplar_per_class: 30
  replay_fraction: 0.1
  use_ewc_in_hybrid: false

vcrr:
  reconfig_k: 48
  soft_alpha: 0.1
  apply_to: all_linear
  randomized_svd: false

distill:
  use_lwf: true
  alpha: 0.5
""",
"parity_tuned_vcrr_exp3.yaml": """
dataset:
  name: cifar100
  num_tasks: 2
  classes_per_task: 10

training:
  model: resnet34
  epochs_per_task: 16
  batch_size: 128
  lr: 0.01
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5e-4
  augment: true
  mixup_alpha: 0.2
  label_smoothing: 0.05
  set_cudnn_benchmark: true
  scheduler:
    step_size: 3
    gamma: 0.5

hybrid:
  use_replay: false

vcrr:
  reconfig_k: 48
  soft_alpha: 0.2
  apply_to: all_linear

distill:
  use_lwf: true
  alpha: 0.7
""",
"parity_tuned_vcrr_exp4.yaml": """
dataset:
  name: cifar100
  num_tasks: 2
  classes_per_task: 10

training:
  model: resnet34
  epochs_per_task: 12
  batch_size: 128
  lr: 0.02
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5e-4
  augment: true
  mixup_alpha: 0.1
  label_smoothing: 0.05
  set_cudnn_benchmark: true

hybrid:
  use_replay: true
  exemplar_per_class: 20
  replay_fraction: 0.1

vcrr:
  reconfig_k: 16
  soft_alpha: 0.0
  apply_to: all_linear
""",
"parity_tuned_vcrr_exp5.yaml": """
dataset:
  name: cifar100
  num_tasks: 2
  classes_per_task: 10

training:
  model: resnet34
  epochs_per_task: 24
  batch_size: 128
  lr: 0.01
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5e-4
  augment: true
  mixup_alpha: 0.2
  label_smoothing: 0.05
  set_cudnn_benchmark: true

hybrid:
  use_replay: true
  exemplar_per_class: 20
  replay_fraction: 0.1

vcrr:
  reconfig_k: 8
  soft_alpha: 0.0
  apply_to: all_linear
""",
"parity_tuned_vcrr_exp6.yaml": """
dataset:
  name: cifar100
  num_tasks: 2
  classes_per_task: 10

training:
  model: resnet34
  epochs_per_task: 16
  batch_size: 128
  lr: 0.01
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5e-4
  augment: true
  mixup_alpha: 0.2
  label_smoothing: 0.1
  set_cudnn_benchmark: true

hybrid:
  use_replay: true
  exemplar_per_class: 30
  replay_fraction: 0.1
  use_ewc_in_hybrid: true

ewc:
  lambda_ewc: 500.0

vcrr:
  reconfig_k: 32
  soft_alpha: 0.05
  apply_to: all_linear
""",
"parity_tuned_vcrr_exp7.yaml": """
dataset:
  name: cifar100
  num_tasks: 2
  classes_per_task: 10

training:
  model: resnet50
  epochs_per_task: 16
  batch_size: 128
  lr: 0.01
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5e-4
  augment: true
  mixup_alpha: 0.2
  label_smoothing: 0.1
  set_cudnn_benchmark: true

hybrid:
  use_replay: true
  exemplar_per_class: 30
  replay_fraction: 0.1

vcrr:
  reconfig_k: 32
  soft_alpha: 0.1
  apply_to: all_linear
""",
"parity_tuned_vcrr_exp8.yaml": """
dataset:
  name: cifar100
  num_tasks: 2
  classes_per_task: 10

training:
  model: resnet34
  epochs_per_task: 16
  batch_size: 128
  lr: 0.01
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5e-4
  augment: true
  mixup_alpha: 0.4
  label_smoothing: 0.2
  set_cudnn_benchmark: true

hybrid:
  use_replay: true
  exemplar_per_class: 30
  replay_fraction: 0.1

vcrr:
  reconfig_k: 32
  soft_alpha: 0.05
  apply_to: all_linear
""",
"parity_tuned_vcrr_exp9.yaml": """
dataset:
  name: cifar100
  num_tasks: 2
  classes_per_task: 10

training:
  model: resnet18
  epochs_per_task: 16
  batch_size: 64
  lr: 0.01
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5e-4
  augment: true
  mixup_alpha: 0.2
  label_smoothing: 0.1
  set_cudnn_benchmark: true

hybrid:
  use_replay: true
  exemplar_per_class: 30
  replay_fraction: 0.1

vcrr:
  reconfig_k: 32
  soft_alpha: 0.05
  apply_to: all_linear
"""
}

created = []
for name, body in configs.items():
    p = CFG_DIR / name
    p.write_text(textwrap.dedent(body).lstrip())
    created.append(name)
print(f"Ensured configs in {CFG_DIR}. Created {len(created)} new files: {created}")
