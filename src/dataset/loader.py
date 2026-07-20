import sys
from pathlib import Path

from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_ROOT))

from preprocessing.transforms import (
    train_joint_transform, train_image_transform, train_mask_transform,
    val_joint_transform, val_image_transform, val_mask_transform,
    test_joint_transform, test_image_transform, test_mask_transform,
)
from dataset import LoveDADataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "splits"

BATCH_SIZE = 4
NUM_WORKERS = 0  # subprocesses for parallel loading

SPLITS = ("train", "test", "val")

# Each split has its own joint/image/mask transform trio:
# train uses augmentation, val/test use resize-only pipelines.
TRANSFORMS = {
    "train": {
        "joint_transform": train_joint_transform,
        "image_transform": train_image_transform,
        "mask_transform": train_mask_transform,
    },
    "val": {
        "joint_transform": val_joint_transform,
        "image_transform": val_image_transform,
        "mask_transform": val_mask_transform,
    },
    "test": {
        "joint_transform": test_joint_transform,
        "image_transform": test_image_transform,
        "mask_transform": test_mask_transform,
    },
}

# 1. Build one dataset per split
datasets = {
    split: LoveDADataset(
        images_dir=DATA_ROOT / split / "images",
        masks_dir=DATA_ROOT / split / "masks",
        **TRANSFORMS[split],
    )
    for split in SPLITS
}

# 2. Build the corresponding DataLoaders (only train is shuffled)
loaders = {
    split: DataLoader(
        dataset=datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=NUM_WORKERS,
    )
    for split in SPLITS
}

# 3. Sanity check: print stats for one batch per split
if __name__ == "__main__":
    import torch

    for split, loader in loaders.items():
        images, masks = next(iter(loader))

        print(split.upper())
        print("=" * 60)

        print("Images")
        print("Shape :", images.shape)
        print("dtype :", images.dtype)
        print("Min   :", images.min().item())
        print("Max   :", images.max().item())
        print()

        print("Masks")
        print("Shape :", masks.shape)
        print("dtype :", masks.dtype)
        print("Classes :", torch.unique(masks))
        print()