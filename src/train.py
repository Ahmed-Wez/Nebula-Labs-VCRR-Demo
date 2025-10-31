#!/usr/bin/env python3
import os
import json
import argparse
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import SmallCNN
from utils import set_seed, get_split_cifar_loaders

def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

def fisher_information(model, loader, device):
    """
    Approximate Fisher diagonal for parameters using loader (one pass).
    Returns a dict mapping parameter name -> tensor (same shape as param).
    """
    model.eval()
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p.data)

    loss_fn = nn.CrossEntropyLoss()
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += (p.grad.data.clone() ** 2)
        count += 1

    if count > 0:
        for n in fisher:
            fisher[n] = fisher[n] / float(count)
    return fisher

def train_task_loop(model, train_loader, device, optimizer, epochs=3, ewc_fisher=None, ewc_params=None, lambda_ewc=1000.0, method="ewc"):
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            # EWC penalty if available
            if method == "ewc" and ewc_fisher is not None and ewc_params is not None:
                penalty = 0.0
                for n, p in model.named_parameters():
                    if n in ewc_fisher:
                        penalty = penalty + (ewc_fisher[n].to(p.device) * (p - ewc_params[n].to(p.device))**2).sum()
                loss = loss + 0.5 * lambda_ewc * penalty
            loss.backward()
            optimizer.step()

def cas_reconfigure(model, k=8):
    # Toy CAS: perform SVD on fc2 weight (if exists) and modify last layer by zeroing small singular values
    if hasattr(model, 'fc2'):
        W = model.fc2.weight.data.clone()  # [num_classes, feat_dim]
        # compute SVD on small matrix (cpu)
        try:
            U, S, Vt = torch.svd_lowrank(W, q=min(k, min(W.shape)))
        except Exception:
            try:
                U, S, Vt = torch.svd(W.cpu())
            except Exception:
                return  # no-op if SVD fails
        S_shrink = S.clone()
        if S_shrink.numel() > k:
            S_shrink[k:] = 0.0
        W_new = U @ torch.diag(S_shrink) @ Vt.t()
        model.fc2.weight.data.copy_(W_new.to(model.fc2.weight.data.device))

def run_experiment(config, method="ewc", seed=0, out_dir="results/run"):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = config.get('dataset', {}).get('num_tasks', 5)
    classes_per_task = config.get('dataset', {}).get('classes_per_task', 20)
    epochs = config.get('training', {}).get('epochs_per_task', 3)
    batch_size = config.get('training', {}).get('batch_size', 128)
    lr = config.get('training', {}).get('lr', 0.01)
    lambda_ewc = config.get('ewc', {}).get('lambda_ewc', 1000.0)
    cas_k = config.get('cas', {}).get('reconfig_k', 8)

    # load tasks -- each task will yield labels remapped to 0..classes_per_task-1
    tasks = get_split_cifar_loaders(num_tasks=num_tasks, batch_size=batch_size,
                                    classes_per_task=classes_per_task, seed=seed)

    # metrics accumulator: list of lists acc_matrix[t][i] = acc on task i after training on t
    acc_matrix = []
    prev_params = None
    fisher_prev = None

    # ensure out_dir exists (caller typically passes results/<method>_seedX)
    os.makedirs(out_dir, exist_ok=True)

    for t, (train_loader, test_loader, cls) in enumerate(tasks):
        # dynamic number of classes for this task
        num_classes = len(cls)
        print(f"[seed={seed}] Task {t+1}/{len(tasks)} — classes: {num_classes}")

        # build a fresh model for this task (simple approach for demo)
        model = SmallCNN(num_classes=num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        # Train on current task (with optional EWC penalty)
        train_task_loop(model, train_loader, device, optimizer, epochs=epochs,
                        ewc_fisher=fisher_prev, ewc_params=prev_params, lambda_ewc=lambda_ewc,
                        method=method)

        # After training: compute fisher (if using EWC)
        if method == "ewc":
            fisher_prev = fisher_information(model, train_loader, device)
            # store previous optimal params (on CPU for memory safety)
            prev_params = {n: p.data.clone().cpu() for n, p in model.named_parameters()}

        # If CAS method, perform toy reconfiguration
        if method == "cas":
            cas_reconfigure(model, k=cas_k)

        # Evaluate on all tasks seen so far
        accs = []
        for i, (_, test_loader_i, _) in enumerate(tasks[:t+1]):
            acc = compute_accuracy(model, test_loader_i, device)
            accs.append(acc)

        acc_matrix.append(accs)

        # Save interim metrics
        run_metrics = {
            'task': t,
            'accs': accs
        }
        with open(os.path.join(out_dir, f"metrics_task{t}.json"), 'w') as f:
            json.dump(run_metrics, f)

    # Save aggregate metrics file
    with open(os.path.join(out_dir, "metrics_all.json"), 'w') as f:
        json.dump({'acc_matrix': acc_matrix}, f)

    print(f"Run complete. Metrics written to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    parser.add_argument('--method', type=str, choices=['ewc', 'cas'], default='ewc')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='results/run')
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg, method=args.method, seed=args.seed, out_dir=args.out)