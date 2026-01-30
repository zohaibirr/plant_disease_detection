from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
    test_ds  = datasets.ImageFolder(f"{data_dir}/test", transform=transform)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        train_ds.classes
    )
