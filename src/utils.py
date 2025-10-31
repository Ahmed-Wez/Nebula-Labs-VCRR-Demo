import random
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_split_cifar_loaders(num_tasks=5, batch_size=128, classes_per_task=20, seed=0):
    """
    Returns a list of (train_loader, test_loader, class_indices) for each task.
    We split CIFAR-100 into `num_tasks` tasks, each with `classes_per_task` classes.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    cifar_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # classes list is 0..99; shuffle deterministically by seed for reproducibility
    rng = np.random.RandomState(seed)
    classes = np.arange(100)
    rng.shuffle(classes)
    tasks = []
    for t in range(num_tasks):
        cls = classes[t*classes_per_task:(t+1)*classes_per_task]
        train_idx = [i for i, (_, y) in enumerate(cifar_train) if y in cls]
        test_idx = [i for i, (_, y) in enumerate(cifar_test) if y in cls]
        train_subset = Subset(cifar_train, train_idx)
        test_subset = Subset(cifar_test, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        tasks.append((train_loader, test_loader, cls.tolist()))
    return tasks