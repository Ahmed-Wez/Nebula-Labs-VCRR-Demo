import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random

# ------------------------
# Helper: fix random seeds
# ------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------
# Helper: subset wrapper
# ------------------------
class RemappedSubset(Dataset):
    """
    Wraps a dataset subset and remaps original CIFAR labels (0..99)
    to local contiguous range 0..(classes_per_task-1).
    """
    def __init__(self, base_dataset, indices, cls_map):
        self.dataset = base_dataset
        self.indices = indices
        self.label_to_local = {int(c): i for i, c in enumerate(cls_map)}
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, self.label_to_local[int(y)]

# ------------------------
# Tasked data loaders
# ------------------------
def get_split_cifar_loaders(num_tasks=5, batch_size=128, classes_per_task=20, seed=0):
    """
    Splits CIFAR100 into tasks, remaps labels locally for each task.
    Returns a list of (train_loader, test_loader, cls_indices).
    """
    set_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    cifar_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    classes = np.arange(100)
    np.random.shuffle(classes)
    tasks = []

    for t in range(num_tasks):
        cls = classes[t*classes_per_task:(t+1)*classes_per_task]
        train_idx = [i for i, (_, y) in enumerate(cifar_train) if y in cls]
        test_idx = [i for i, (_, y) in enumerate(cifar_test) if y in cls]

        train_subset = RemappedSubset(cifar_train, train_idx, cls.tolist())
        test_subset = RemappedSubset(cifar_test, test_idx, cls.tolist())

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        tasks.append((train_loader, test_loader, cls.tolist()))

    return tasks