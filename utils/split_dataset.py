import os
import shutil
import random

SPLIT = (0.7, 0.15, 0.15)  # train, val, test
SEED = 42


def find_color_root(raw_root: str) -> str:
    """
    Find the directory under raw_root that contains a 'color' subfolder.
    """
    for entry in os.listdir(raw_root):
        path = os.path.join(raw_root, entry)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "color")):
            return os.path.join(path, "color")
    raise FileNotFoundError("Could not find a 'color' folder under " + raw_root)


def main():
    random.seed(SEED)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_root = os.path.join(base_dir, "data", "plantvillage_raw")
    source_dir = find_color_root(raw_root)
    target_dir = os.path.join(base_dir, "data", "plantvillage")

    print("Raw root:", raw_root)
    print("Source dir (color):", source_dir)
    print("Target dir:", target_dir)

    classes = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [
            f for f in os.listdir(cls_path)
            if os.path.isfile(os.path.join(cls_path, f))
        ]
        random.shuffle(images)

        n = len(images)
        train_end = int(SPLIT[0] * n)
        val_end = train_end + int(SPLIT[1] * n)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, files in splits.items():
            split_cls_dir = os.path.join(target_dir, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)

            for f in files:
                shutil.copy(
                    os.path.join(cls_path, f),
                    os.path.join(split_cls_dir, f),
                )

    print("✅ Dataset split completed successfully.")
    print(f"Data root: {target_dir}")


if __name__ == "__main__":
    main()
