"""CIFAR-10 yükleme, eğitim/doğrulama ayrımı ve veri artırma."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def image_transform(augment=False):
    steps = []
    if augment:
        steps += [transforms.RandomCrop(32, padding=4),
                  transforms.RandomHorizontalFlip()]
    return transforms.Compose(steps + [transforms.ToTensor(),
                                      transforms.Normalize(MEAN, STD)])


def get_training_loaders(augment=False, batch_size=128, root="data",
                         num_workers=2, val_fraction=0.1, seed=42):
    """Aynı seed ile sabit, örtüşmeyen eğitim/doğrulama alt kümeleri oluşturur."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction (0, 1) aralığında olmalı.")
    training = datasets.CIFAR10(root, train=True, download=True,
                                transform=image_transform(augment))
    validation = datasets.CIFAR10(root, train=True, download=True,
                                  transform=image_transform())
    val_size = int(len(training) * val_fraction)
    if not 0 < val_size < len(training):
        raise ValueError("Eğitim ve doğrulama kümeleri boş olmamalı.")
    indices = torch.randperm(len(training), generator=torch.Generator().manual_seed(seed)).tolist()
    train_set = Subset(training, indices[val_size:])
    val_set = Subset(validation, indices[:val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def get_test_loader(root="data", num_workers=2):
    """Test verisini eğitim verisini yüklemeden açar."""
    dataset = datasets.CIFAR10(root, train=False, download=True, transform=image_transform())
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)


def get_loaders(augment=False, batch_size=128, root="data", num_workers=2):
    """Eski çağrılar için tam eğitim ve test yükleyicilerini döndürür."""
    dataset = datasets.CIFAR10(root, train=True, download=True,
                               transform=image_transform(augment))
    return (DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            get_test_loader(root, num_workers))
