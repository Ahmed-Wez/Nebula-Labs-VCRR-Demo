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
    # approximate fisher diagonal for parameters using labels from loader
    model.eval()
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p.data)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += (p.grad.data.clone() ** 2)
    # average
    for n in fisher:
        fisher[n] = fisher[n] / len(loader)
    return fisher

def train_task(model, train_loader, optimizer, device, epochs=3):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

def apply_ewc_penalty(model, fisher, opt_params, lambda_ewc):
    loss = 0.0
    for n, p in model.named_parameters():
        if n in fisher and n in opt_params:
            loss = loss + (fisher[n].to(p.device) * (p - opt_params[n].to(p.device))**2).sum()
    return 0.5 * lambda_ewc * loss

def cas_reconfigure(model, k=8):
    # Toy CAS: perform SVD on fc2 weight (if exists) and modify last layer by zeroing small singular values
    if hasattr(model, 'fc2'):
        W = model.fc2.weight.data.clone()  # [num_classes, feat_dim]
        # compute SVD on small matrix (cpu)
        try:
            U, S, Vt = torch.svd_lowrank(W, q=k)  # approximate low-rank
        except Exception:
            try:
                U, S, Vt = torch.svd(W.cpu())
            except Exception:
                return  # no-op if SVD fails
        # reconstruct but shrink smaller singular values
        S_shrink = S.clone()
        # zero out small singular values beyond k (toy)
        if S_shrink.numel() > k:
            S_shrink[k:] = 0.0
        W_new = U @ torch.diag(S_shrink) @ Vt.t()
        model.fc2.weight.data.copy_(W_new.to(model.fc2.weight.data.device))

def run_experiment(config, method="ewc", seed=0, out_dir="results/run"):
    import torch
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = config.get('dataset', {}).get('num_tasks', 5)
    classes_per_task = config.get('dataset', {}).get('classes_per_task', 20)
    epochs = config.get('training', {}).get('epochs_per_task', 3)
    batch_size = config.get('training', {}).get('batch_size', 128)
    lr = config.get('training', {}).get('lr', 0.01)
    lambda_ewc = config.get('ewc', {}).get('lambda_ewc', 1000.0)
    cas_k = config.get('cas', {}).get('reconfig_k', 8)

    tasks = get_split_cifar_loaders(num_tasks=num_tasks, batch_size=batch_size,
                                    classes_per_task=classes_per_task, seed=seed)

    # For simplicity, we build a fresh model for each task head: small workaround to handle changing num_classes
    acc_matrix = []  # after training on task t, evaluate on tasks 0..t
    prev_params = None
    fisher_prev = None

    model = None
    for t, (train_loader, test_loader, cls) in enumerate(tasks):
        # initialize model for current task with output size = classes_per_task
        torch.cuda.empty_cache()
        model = SmallCNN(num_classes=classes_per_task).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        # If EWC: add penalty towards previous opt params using fisher_prev
        if method == "ewc" and fisher_prev is not None and prev_params is not None:
            # We'll train normally but include EWC penalty by manually adjusting grads
            # Simple approach: compute loss + penalty in train loop below
            pass

        # Train for epochs
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                # EWC penalty
                if method == "ewc" and fisher_prev is not None and prev_params is not None:
                    penalty = 0.0
                    for n, p in model.named_parameters():
                        if n in fisher_prev:
                            penalty = penalty + (fisher_prev[n].to(p.device) * (p - prev_params[n].to(p.device))**2).sum()
                    loss = loss + 0.5 * lambda_ewc * penalty
                loss.backward()
                optimizer.step()

        # After training on task t, compute and store Fisher (approx) for EWC
        # compute fisher using training loader (cheap approx)
        if method == "ewc":
            fisher_prev = {}
            for n, p in model.named_parameters():
                fisher_prev[n] = torch.zeros_like(p.data)
            model.eval()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                out = model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        fisher_prev[n] += (p.grad.data.clone() ** 2)
            # normalize
            for n in fisher_prev:
                fisher_prev[n] /= max(1.0, len(train_loader))

            # save previous optimal params
            prev_params = {n: p.data.clone().cpu() for n, p in model.named_parameters()}

        # If CAS method, perform toy reconfiguration after finishing task training
        if method == "cas":
            cas_reconfigure(model, k=cas_k)

        # Evaluate on all tasks seen so far
        accs = []
        for i, (_, test_loader_i, _) in enumerate(tasks[:t+1]):
            acc = compute_accuracy(model, test_loader_i, device)
            accs.append(acc)
        acc_matrix.append(accs)

        # Save interim checkpoint and metrics for this task
        os.makedirs(out_dir, exist_ok=True)
        ckpt = {
            'task': t,
            'accs': accs
        }
        with open(os.path.join(out_dir, f"metrics_task{t}.json"), 'w') as f:
            json.dump(ckpt, f)

    # Save aggregate metrics file
    with open(os.path.join(out_dir, "metrics_all.json"), 'w') as f:
        json.dump({'acc_matrix': acc_matrix}, f)

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