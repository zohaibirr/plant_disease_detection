import os
from typing import Tuple, List

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(image_size: int = 128):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    return train_transform, eval_transform


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 0,   # 🔴 set to 0 to avoid multiprocessing issues in Colab
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_transform, eval_transform = get_transforms(image_size)

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    class_names = train_ds.classes
    return train_loader, val_loader, test_loader, class_names
