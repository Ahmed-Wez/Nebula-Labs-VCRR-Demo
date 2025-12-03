from pathlib import Path
import textwrap

REPO = Path.cwd() / "Nebula-Labs-VCRR-Demo"
CFG_DIR = REPO / "configs"
CFG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Generating configs in: {CFG_DIR}\n")

# This will create any missing baseline configs
# Note: We already created these manually, so this is just a backup generator

DATASETS = {
    'cifar10': {
        'num_tasks': 5,
        'classes_per_task': 2,
        'model': 'smallcnn',
        'epochs': 10,
        'batch_size': 128,
        'buffer_size': 500,
        'augment': True
    },
    'cifar100': {
        'num_tasks': 5,
        'classes_per_task': 20,
        'model': 'resnet34',
        'epochs': 16,
        'batch_size': 128,
        'buffer_size': 2000,
        'augment': True
    },
    'permuted_mnist': {
        'num_tasks': 10,
        'classes_per_task': 10,
        'model': 'mlp',
        'epochs': 5,
        'batch_size': 128,
        'buffer_size': 1000,
        'augment': False
    },
    'tinyimagenet': {
        'num_tasks': 20,
        'classes_per_task': 10,
        'model': 'resnet18',
        'epochs': 10,
        'batch_size': 64,
        'buffer_size': 2000,
        'augment': True
    },
    'core50': {
        'num_tasks': 10,
        'classes_per_task': 5,
        'model': 'resnet18',
        'epochs': 10,
        'batch_size': 64,
        'buffer_size': 1000,
        'augment': True
    }
}

METHODS_TEMPLATES = {
    'baseline': {},
    'ewc': {'ewc': {'lambda_ewc': 5000.0}},
    'lwf': {'distill': {'use_lwf': True, 'alpha': 0.5, 'temperature': 2.0}},
    'er': {'er': {'buffer_size': '{buffer_size}'}},
    'agem': {'agem': {'buffer_size': '{buffer_size}', 'sample_size': 128}},
    'der': {'der': {'buffer_size': '{buffer_size}', 'alpha': 0.5, 'beta': 0.5}},
    'packnet': {'packnet': {'prune_percentage': 0.5, 'retrain_epochs': 5}},
    'prognn': {'prognn': {'hidden_sizes': [256, 256]}},
    'hope': {'hope': {'eta': 0.01, 'apply_to': 'fc_only', 'use_with_optimizer': True, 'normalize_inputs': True}},
    'icarl': {
        'hybrid': {'use_replay': True, 'exemplar_per_class': 20, 'replay_fraction': 0.0},
        'icarl': {'enabled': True, 'nearest_exemplar': True, 'herding': True}
    },
    'gdumb': {
        'hybrid': {'use_replay': True, 'exemplar_per_class': 50, 'replay_fraction': 0.0},
        'gdumb': {'enabled': True, 'train_only_at_end': True}
    },
    'mir': {
        'hybrid': {'use_replay': True, 'exemplar_per_class': 20, 'replay_fraction': 0.2},
        'mir': {'enabled': True, 'lookahead_updates': 1, 'score_by_loss_increase': True}
    },
    'scr': {
        'hybrid': {'use_replay': True, 'exemplar_per_class': 20, 'replay_fraction': 0.2},
        'scr': {'enabled': True, 'selection_mode': 'diversity'}
    },
    'vcrr': {'vcrr': {'reconfig_k': 32, 'soft_alpha': 0.0, 'apply_to': 'all_linear', 'randomized_svd': False, 'skip_output_linear': True}}
}

created_count = 0

for dataset, dset_cfg in DATASETS.items():
    for method, method_cfg_template in METHODS_TEMPLATES.items():
        cfg_name = f"{dataset}_{method}.yaml"
        cfg_path = CFG_DIR / cfg_name
        
        if cfg_path.exists():
            continue  # Don't overwrite existing configs
        
        # Build config
        config = {
            'dataset': {
                'name': dataset,
                'num_tasks': dset_cfg['num_tasks'],
                'classes_per_task': dset_cfg['classes_per_task']
            },
            'training': {
                'model': dset_cfg['model'],
                'epochs_per_task': dset_cfg['epochs'],
                'batch_size': dset_cfg['batch_size'],
                'lr': 0.01,
                'optimizer': 'sgd',
                'momentum': 0.9,
                'weight_decay': 5e-4 if dataset != 'permuted_mnist' else 0.0,
                'augment': dset_cfg['augment']
            },
            'method': method
        }
        
        # Add method-specific config
        for key, value in method_cfg_template.items():
            if isinstance(value, dict) and '{buffer_size}' in str(value):
                # Replace buffer_size placeholder
                value = eval(str(value).replace('{buffer_size}', str(dset_cfg['buffer_size'])))
            config[key] = value
        
        # Write YAML
        import yaml
        with open(cfg_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        created_count += 1
        print(f"Created: {cfg_name}")

print(f"\nTotal configs created: {created_count}")
print(f"Config directory: {CFG_DIR}")