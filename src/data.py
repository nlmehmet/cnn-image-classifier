"""CIFAR-10 veri yükleme ve (isteğe bağlı) veri artırma."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def get_loaders(augment: bool = False, batch_size: int = 128, root: str = "data",
                num_workers: int = 2):
    norm = transforms.Normalize(MEAN, STD)
    train_tf = [transforms.ToTensor(), norm]
    if augment:
        train_tf = [transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip()] + train_tf
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    train_set = datasets.CIFAR10(root, train=True, download=True,
                                 transform=transforms.Compose(train_tf))
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
