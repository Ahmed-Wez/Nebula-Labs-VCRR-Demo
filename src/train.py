import os, time, json, argparse, random, copy
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# ---------- utils ----------
def set_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def maybe_reset_cuda_peak(dev):
    if dev.type == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats(dev)
        except Exception:
            pass

def maybe_get_cuda_peak_mb(dev):
    if dev.type == 'cuda':
        try:
            peak = torch.cuda.max_memory_allocated(dev)
            return float(peak) / (1024.0**2)
        except Exception:
            return float('nan')
    return float('nan')

# ---------- data helpers ----------
def get_split_cifar_loaders(num_tasks=5, classes_per_task=20, batch_size=128, seed=0, num_workers=2, augment=True):
    # transforms (canonical CIFAR100 normalization)
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])
    root = "./data"
    trainset = datasets.CIFAR100(root=root, train=True, download=True)
    testset  = datasets.CIFAR100(root=root, train=False, download=True)
    rng = np.random.default_rng(seed)
    all_classes = list(range(100))
    perm = rng.permutation(all_classes).tolist()
    tasks = []
    for t in range(num_tasks):
        cls = perm[t*classes_per_task:(t+1)*classes_per_task]
        cls_set = set(cls)
        train_idx = [i for i,(img,lab) in enumerate(trainset) if int(lab) in cls_set]
        test_idx  = [i for i,(img,lab) in enumerate(testset) if int(lab) in cls_set]
        def build_subset(orig, indices, transform):
            imgs=[]; labs=[]
            for i in indices:
                img, lab = orig[i]
                if transform: img = transform(img)
                imgs.append(img)
                labs.append(int(lab))
            if len(imgs)==0:
                return torch.utils.data.TensorDataset(torch.zeros((0,3,32,32)), torch.zeros((0,), dtype=torch.long))
            X = torch.stack(imgs)
            Y = torch.tensor(labs, dtype=torch.long)
            return torch.utils.data.TensorDataset(X,Y)
        ds_train = build_subset(trainset, train_idx, transform_train)
        ds_test  = build_subset(testset, test_idx, transform_test)
        loader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        loader_test  = torch.utils.data.DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        tasks.append((loader_train, loader_test, cls))
    return tasks

# ---------- model ----------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,128,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(128*4*4,256)
        self.fc2 = nn.Linear(256,num_classes)
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        feat = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(feat))
        return self.fc2(x)

def get_model(backbone='smallcnn', total_classes=100):
    if backbone == 'resnet18':
        net = models.resnet18(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features, total_classes)
        return net
    elif backbone == 'resnet34':
        net = models.resnet34(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features, total_classes)
        return net
    elif backbone == 'resnet50':
        net = models.resnet50(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features, total_classes)
        return net
    else:
        return SmallCNN(num_classes=total_classes)

# ---------- Exemplar buffer ----------
class ExemplarBuffer:
    def __init__(self, per_class=20):
        self.per_class = int(per_class)
        self.store = {}
    def add_examples(self, xs_cpu, ys_cpu):
        for x,y in zip(xs_cpu, ys_cpu):
            g = int(y.item())
            if g not in self.store: self.store[g] = []
            lst = self.store[g]
            if len(lst) < self.per_class:
                lst.append(x.clone())
            else:
                # small random replacement
                if random.random() < 0.01:
                    idx = random.randrange(len(lst))
                    lst[idx] = x.clone()
    def sample(self, n_total):
        if len(self.store)==0: return None, None
        classes = list(self.store.keys())
        per_cls = max(1, n_total // len(classes))
        xs_list=[]; ys_list=[]
        for c in classes:
            lst = self.store[c]
            if len(lst)==0: continue
            k = min(per_cls, len(lst))
            idxs = np.random.choice(len(lst), size=k, replace=False)
            for i in idxs:
                xs_list.append(lst[i])
                ys_list.append(c)
            if len(xs_list) >= n_total: break
        if len(xs_list)==0: return None, None
        X = torch.stack(xs_list)[:n_total]
        Y = torch.tensor(ys_list[:n_total], dtype=torch.long)
        return X, Y
    def __len__(self):
        return sum(len(v) for v in self.store.values())

# ---------- VCRR ----------
def vcrr_reconfigure_all_linears(model, k=8, soft_alpha=0.0, randomized=False, skip_output_linear=True, num_classes=None):
    """Continual Adaptation through Selective reconfiguration via SVD rank reduction"""
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            try:
                if skip_output_linear and (num_classes is not None):
                    if getattr(m, 'out_features', None) == int(num_classes):
                        continue
            except Exception:
                pass
            W = m.weight.data.clone().cpu()
            try:
                U,S,Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception:
                try:
                    U,S,Vh = torch.svd(W)
                except Exception:
                    continue
            S_shrink = S.clone()
            if S_shrink.numel() > k:
                if soft_alpha <= 0.0:
                    S_shrink[k:] = 0.0
                else:
                    S_shrink[k:] = S_shrink[k:] * float(soft_alpha)
            W_new = (U * S_shrink.unsqueeze(0)) @ Vh
            W_new = W_new.to(m.weight.data.device)
            try:
                m.weight.data.copy_(W_new)
            except Exception:
                continue

# ---------- Fisher ----------
def fisher_information(model, loader, device, max_batches=100):
    model.eval()
    fisher = {n: torch.zeros_like(p.data).to('cpu') for n,p in model.named_parameters()}
    loss_fn = nn.CrossEntropyLoss()
    count = 0
    for x,y in loader:
        if count >= max_batches: break
        x,y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += (p.grad.data.clone().cpu() ** 2)
        count += 1
    if count > 0:
        for n in fisher: fisher[n] /= float(count)
    return fisher

# ---------- accuracy / mixup ----------
def compute_accuracy(model, loader, device):
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total>0 else 0.0

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0.0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    if batch_size <= 1:
        return x, y, None, 1.0
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    if y_b is None:
        return criterion(pred, y_a)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---------- HOPE helpers ----------
def hope_register_hooks(model, apply_to='fc_only'):
    forward_inputs = {}
    backward_grads = {}
    target_modules = []

    def make_fhook(m):
        def fhook(module, input, output):
            if input and isinstance(input[0], torch.Tensor):
                forward_inputs[id(module)] = input[0].detach().cpu()
        return fhook

    def make_bhook(m):
        def bhook(module, grad_input, grad_output):
            if grad_output and isinstance(grad_output[0], torch.Tensor):
                backward_grads[id(module)] = grad_output[0].detach().cpu()
        return bhook

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            try:
                is_fc2 = hasattr(model, 'fc2') and (m is getattr(model, 'fc2'))
            except Exception:
                is_fc2 = False
            if apply_to == 'all_linear' or (apply_to == 'fc_only' and is_fc2):
                target_modules.append(m)
                hooks.append(m.register_forward_hook(make_fhook(m)))
                try:
                    hooks.append(m.register_full_backward_hook(make_bhook(m)))
                except Exception:
                    hooks.append(m.register_backward_hook(make_bhook(m)))
    return target_modules, forward_inputs, backward_grads, hooks

def hope_apply_batch_update(target_modules, forward_inputs, backward_grads, eta=0.01, normalize_inputs=True):
    for m in target_modules:
        key = id(m)
        if key not in forward_inputs or key not in backward_grads:
            continue
        x = forward_inputs[key]      # cpu
        g_out = backward_grads[key]  # cpu
        if x.numel() == 0 or g_out.numel() == 0:
            continue
        dev = m.weight.data.device
        x = x.to(dev)
        g_out = g_out.to(dev)
        x_mean = x.mean(dim=0)
        g_mean = g_out.mean(dim=0)
        if normalize_inputs:
            norm = x_mean.norm(p=2)
            if norm > 0:
                x_mean = x_mean / (norm + 1e-12)
        in_dim = x_mean.shape[0]
        P = torch.eye(in_dim, device=dev) - torch.ger(x_mean, x_mean)
        with torch.no_grad():
            W = m.weight.data
            W_new = W @ P
            W_new = W_new - float(eta) * torch.ger(g_mean, x_mean)
            m.weight.data.copy_(W_new)

# ---------- main experiment runner ----------
def run_experiment(cfg, method="ewc", seed=0, out_dir="results/run", total_classes=100, deterministic=False):
    t0 = time.time()
    set_seed(seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read style safely
    dataset_cfg = cfg.get('dataset',{}) or {}
    training_cfg = cfg.get('training',{}) or {}
    vcrr_cfg = cfg.get('vcrr',{}) or {}
    hybrid_cfg = cfg.get('hybrid',{}) or {}
    ewc_cfg = cfg.get('ewc',{}) or {}
    hope_cfg = cfg.get('hope',{}) or {}
    distill_cfg = cfg.get('distill',{}) or {}

    num_tasks = int(dataset_cfg.get('num_tasks',5))
    classes_per_task = int(dataset_cfg.get('classes_per_task',20))
    batch_size = int(training_cfg.get('batch_size',128))
    epochs = int(training_cfg.get('epochs_per_task',1))
    lr = float(training_cfg.get('lr',0.01))
    backbone = training_cfg.get('model','smallcnn')
    vcrr_k = int(vcrr_cfg.get('reconfig_k',8))
    vcrr_soft_alpha = float(vcrr_cfg.get('soft_alpha',0.0))
    vcrr_apply = vcrr_cfg.get('apply_to','all_linear')
    randomized_svd = bool(vcrr_cfg.get('randomized_svd', False))
    vcrr_skip_output = bool(vcrr_cfg.get('skip_output_linear', True))

    use_replay = bool(hybrid_cfg.get('use_replay', False))
    exemplar_per_class = int(hybrid_cfg.get('exemplar_per_class',20))
    replay_fraction = float(hybrid_cfg.get('replay_fraction',0.1))
    use_ewc_in_hybrid = bool(hybrid_cfg.get('use_ewc_in_hybrid', False))
    lambda_ewc = float(ewc_cfg.get('lambda_ewc',1000.0))

    mixup_alpha = float(training_cfg.get('mixup_alpha', 0.0))
    label_smoothing = float(training_cfg.get('label_smoothing', 0.0))
    optimizer_name = training_cfg.get('optimizer','sgd')
    scheduler_cfg = training_cfg.get('scheduler', None)
    use_amp = bool(training_cfg.get('use_amp', False))
    set_cudnn_benchmark = bool(training_cfg.get('set_cudnn_benchmark', False))
    augment = bool(training_cfg.get('augment', True))

    # distillation (LwF style)
    use_lwf = bool(distill_cfg.get('use_lwf', False))
    distill_alpha = float(distill_cfg.get('alpha', 0.5))

    hope_eta = float(hope_cfg.get('eta', 0.01))
    hope_apply_to = hope_cfg.get('apply_to', 'fc_only')
    hope_use_with_opt = bool(hope_cfg.get('use_with_optimizer', True))
    hope_normalize = bool(hope_cfg.get('normalize_inputs', True))

    if set_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # loaders
    tasks = get_split_cifar_loaders(num_tasks=num_tasks, classes_per_task=classes_per_task, batch_size=batch_size, seed=seed, augment=augment)
    expected_steps = 0
    for loader,_,_ in tasks:
        try:
            expected_steps += len(loader) * epochs
        except Exception:
            pass

    model = get_model(backbone, total_classes).to(device)
    n_params = count_params(model)

    # optimizer
    weight_decay = float(training_cfg.get('weight_decay',5e-4))
    if optimizer_name.lower() == 'adamw':
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        momentum = float(training_cfg.get('momentum', 0.9))
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    scheduler = None
    if scheduler_cfg:
        step_size = int(scheduler_cfg.get('step_size', 4))
        gamma = float(scheduler_cfg.get('gamma', 0.5))
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    if label_smoothing > 0.0:
        try:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        except TypeError:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    replay_buf = ExemplarBuffer(per_class=exemplar_per_class) if use_replay else None

    # HOPE hooks registration
    hope_target_modules, hope_forward_inputs, hope_backward_grads, hope_hooks = hope_register_hooks(model, apply_to=hope_apply_to)

    fisher_sum = {}
    prev_params = {}
    tasks_seen = 0
    acc_matrix = []
    os.makedirs(out_dir, exist_ok=True)

    epoch_times = []
    maybe_reset_cuda_peak(device)

    # prev_model for LwF-style distillation
    prev_model = None

    for t, (train_loader, test_loader, cls_list) in enumerate(tasks):
        print(f"[seed={seed}] Training task {t+1}/{num_tasks} classes={len(cls_list)} method={method}")

        if use_lwf and prev_model is None and tasks_seen > 0:
            pass

        for epoch in range(epochs):
            e0 = time.time()
            model.train()
            for xb,yb in train_loader:
                xb,yb = xb.to(device), yb.to(device)

                # MixUp on the ORIGINAL minibatch
                if mixup_alpha > 0.0:
                    xb_mix, y_a, y_b, lam = mixup_data(xb, yb, mixup_alpha)
                    xb = xb_mix
                else:
                    y_a = None; y_b = None; lam = 1.0

                # initialize merged label vars
                y_all = None
                y_all_a = None
                y_all_b = None

                # replay merge
                if replay_buf is not None and len(replay_buf) > 0:
                    rep_n = max(1, int(replay_fraction * xb.size(0)))
                    xr, yr = replay_buf.sample(rep_n)
                    if xr is not None:
                        xr = xr.to(device); yr = yr.to(device)
                        x_all = torch.cat([xb, xr], dim=0)
                        # create merged labels depending on whether mixup active
                        if mixup_alpha > 0.0:
                            try:
                                y_all_a = torch.cat([y_a, yr], dim=0)
                            except Exception:
                                y_all_a = torch.cat([ (y_a if y_a is not None else yb), yr ], dim=0)
                            if y_b is not None:
                                try:
                                    y_all_b = torch.cat([y_b, yr], dim=0)
                                except Exception:
                                    y_all_b = None
                            else:
                                y_all_b = None
                        else:
                            y_all = torch.cat([yb, yr], dim=0)
                    else:
                        # replay sampling returned nothing
                        x_all = xb
                        if mixup_alpha > 0.0:
                            y_all_a = y_a
                            y_all_b = y_b
                        else:
                            y_all = yb
                else:
                    # no replay
                    x_all = xb
                    if mixup_alpha > 0.0:
                        y_all_a = y_a
                        y_all_b = y_b
                    else:
                        y_all = yb

                opt.zero_grad()
                out = model(x_all)

                # compute supervised loss
                if mixup_alpha > 0.0:
                    if y_all_a is None:
                        y_all_a = y_a
                    if y_all_b is None and y_b is None:
                        y_all_b = None
                    if y_all_b is None:
                        if y_all_a is None:
                            loss = loss_fn(out, yb.to(device))
                        else:
                            loss = mixup_criterion(loss_fn, out, y_all_a.to(device), y_all_b.to(device) if y_all_b is not None else None, lam)
                    else:
                        loss = mixup_criterion(loss_fn, out, y_all_a.to(device), y_all_b.to(device), lam)
                else:
                    loss = loss_fn(out, y_all.to(device))

                # EWC penalty
                if (method == 'ewc' and tasks_seen > 0) or (method == 'hybrid' and use_ewc_in_hybrid and tasks_seen > 0):
                    pen = 0.0
                    for n, p in model.named_parameters():
                        if n in fisher_sum and n in prev_params:
                            pen += (fisher_sum[n].to(p.device) * (p - prev_params[n].to(p.device))**2).sum()
                    loss = loss + 0.5 * lambda_ewc * pen

                # LwF-style distillation loss
                if use_lwf and (prev_model is not None):
                    try:
                        with torch.no_grad():
                            t_logits = prev_model(x_all.to(device))
                        kl = nn.KLDivLoss(reduction='batchmean')
                        loss = loss + float(distill_alpha) * kl(F.log_softmax(out, dim=1), F.softmax(t_logits, dim=1))
                    except Exception:
                        pass

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                # HOPE application logic
                if method == 'hope' or (method == 'hybrid' and cfg.get('hybrid',{}).get('use_hope_in_hybrid', False)):
                    hope_apply_batch_update(hope_target_modules, hope_forward_inputs, hope_backward_grads, eta=hope_eta, normalize_inputs=hope_normalize)
                    if not hope_use_with_opt:
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad = None

                # optimizer step
                if hope_use_with_opt or method != 'hope':
                    opt.step()

            e1 = time.time()
            epoch_times.append(e1 - e0)
            if scheduler is not None:
                scheduler.step()

        # update replay buffer
        if replay_buf is not None:
            for xb, yb in train_loader:
                replay_buf.add_examples(xb.cpu(), yb.cpu())

        # EWC fisher update
        if method == 'ewc' or (method == 'hybrid' and use_ewc_in_hybrid):
            fisher_t = fisher_information(model, train_loader, device, max_batches=200)
            if tasks_seen == 0:
                for n in fisher_t: fisher_sum[n] = fisher_t[n].cpu()
            else:
                for n in fisher_t:
                    fisher_sum[n] = (fisher_sum[n] * tasks_seen + fisher_t[n].cpu()) / float(tasks_seen + 1)
            prev_params = {n: p.data.clone().cpu() for n,p in model.named_parameters()}

        # VCRR reconfiguration
        if method in ('vcrr', 'hybrid'):
            vcrr_reconfigure_all_linears(model, k=vcrr_k, soft_alpha=vcrr_soft_alpha, randomized=randomized_svd, skip_output_linear=vcrr_skip_output, num_classes=total_classes)

        # set prev_model snapshot for LwF
        if use_lwf:
            try:
                prev_model = copy.deepcopy(model).to(device)
                prev_model.eval()
                for p in prev_model.parameters():
                    p.requires_grad = False
            except Exception:
                prev_model = None

        tasks_seen += 1
        accs = []
        for i, (_, test_loader_i, cls_i) in enumerate(tasks[:t+1]):
            acc = compute_accuracy(model, test_loader_i, device)
            accs.append(acc)
        acc_matrix.append(accs)
        with open(os.path.join(out_dir, f"metrics_task{t}.json"), 'w') as f:
            json.dump({'task': t, 'accs': accs}, f)

    # final artifacts & run_info
    metrics_all = {
        'acc_matrix': acc_matrix,
        'num_tasks': num_tasks,
        'classes_per_task': classes_per_task
    }
    with open(os.path.join(out_dir, "metrics_all.json"), 'w') as f:
        json.dump(metrics_all, f)

    # compute forgetting robustly
    runF = float('nan')
    initial_acc_t1 = float('nan'); final_acc_t1 = float('nan')
    if len(acc_matrix) >= 1:
        try:
            initial_acc_t1 = float(acc_matrix[0][0]) if len(acc_matrix[0])>0 else float('nan')
            final_acc_t1 = float(acc_matrix[-1][0]) if len(acc_matrix[-1])>0 else float('nan')
            if not (np.isnan(initial_acc_t1) or np.isnan(final_acc_t1)):
                runF = initial_acc_t1 - final_acc_t1
        except Exception:
            runF = float('nan')

    avg_acc = float(np.mean([row[-1] if len(row)>0 else np.nan for row in acc_matrix])) if len(acc_matrix)>0 else float('nan')
    total_time = time.time() - t0
    peak_mem_mb = maybe_get_cuda_peak_mb(device)

    run_info = {
        'F': runF,
        'initial_acc_task1': (initial_acc_t1 if not np.isnan(initial_acc_t1) else None),
        'final_acc_task1': (final_acc_t1 if not np.isnan(final_acc_t1) else None),
        'avg_acc': avg_acc,
        'time_s': total_time,
        'params': n_params,
        'epoch_times': epoch_times,
        'peak_mem_mb': peak_mem_mb,
        'expected_steps': expected_steps,
        'method': method,
        'cfg': cfg
    }
    with open(os.path.join(out_dir, "run_info.json"), 'w') as f:
        json.dump(run_info, f)

    os.makedirs('results', exist_ok=True)
    summary_csv = os.path.join('results','summary_all.csv')
    if not os.path.exists(summary_csv):
        with open(summary_csv,'w') as fh:
            fh.write("run,F,initial_acc_task1,final_acc_task1,avg_acc,time_s\n")
    with open(summary_csv,'a') as fh:
        fh.write(f"{os.path.basename(out_dir)},{str(runF)},{str(run_info.get('initial_acc_task1'))},{str(run_info.get('final_acc_task1'))},{avg_acc:.4f},{total_time:.2f}\n")

    print(f"Wrote {out_dir} F={runF} initial_acc_t1={run_info.get('initial_acc_task1')} final_acc_t1={run_info.get('final_acc_task1')} avg_acc={avg_acc:.4f} time_s={total_time:.2f} params={n_params} peak_mem_mb={peak_mem_mb:.1f}")

# CLI
if __name__ == "__main__":
    import yaml, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--method', default='vcrr')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='results/run')
    parser.add_argument('--total_classes', type=int, default=100)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config,'r'))
    run_experiment(cfg, method=args.method, seed=args.seed, out_dir=args.out, total_classes=args.total_classes, deterministic=args.deterministic)