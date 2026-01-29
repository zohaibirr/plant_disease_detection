import os
import shutil
import random

SOURCE_DIR = "data/plantvillage/color"
TARGET_DIR = "data/plantvillage"
SPLIT = (0.7, 0.15, 0.15)  # train, val, test
SEED = 42

random.seed(SEED)

classes = os.listdir(SOURCE_DIR)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

for cls in classes:
    cls_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    n = len(images)
    train_end = int(SPLIT[0] * n)
    val_end = train_end + int(SPLIT[1] * n)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        split_cls_dir = os.path.join(TARGET_DIR, split, cls)
        os.makedirs(split_cls_dir, exist_ok=True)

        for f in files:
            shutil.copy(
                os.path.join(cls_path, f),
                os.path.join(split_cls_dir, f)
            )

print("✅ Dataset split completed successfully.")
