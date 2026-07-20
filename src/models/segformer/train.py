"""
Training script for SegFormer-B0 on the LoveDA semantic segmentation dataset.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# --- Make sibling packages importable (dataset, preprocessing, models, training) ---
# This file lives at src/models/segformer/train.py, so we need to go up 3 levels
# (segformer -> models -> src) to reach src/, the root that contains all sibling packages.
SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(SRC_ROOT))

from dataset.loader import loaders  # dict: {"train": DataLoader, "val": ..., "test": ...}
from models.segformer.segformer import SegFormer
from preprocessing.transforms import NUM_CLASSES
from training.metrics import (
    IGNORE_INDEX,
    intersection_and_union,
    iou_per_class,
    mean_iou,
    pixel_accuracy,
)

# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
LEARNING_RATE = 6e-5  # lower than U-Net: fine-tuning a pretrained transformer backbone
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0  # max gradient norm; None to disable
CHECKPOINT_DIR = SRC_ROOT.parent / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "segformer_best.pth"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "segformer_last.pth"


# ============================================================
# Train / validation loops
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total_union = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    progress_bar = tqdm(loader, desc="Train", leave=False)
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += pixel_accuracy(logits, masks) * images.size(0)

        batch_inter, batch_union = intersection_and_union(logits, masks)
        total_intersection += batch_inter
        total_union += batch_union

        progress_bar.set_postfix(loss=loss.item())

    n = len(loader.dataset)
    epoch_miou = mean_iou(iou_per_class(total_intersection, total_union))
    return running_loss / n, running_acc / n, epoch_miou


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total_union = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    progress_bar = tqdm(loader, desc="Eval", leave=False)
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        running_acc += pixel_accuracy(logits, masks) * images.size(0)

        batch_inter, batch_union = intersection_and_union(logits, masks)
        total_intersection += batch_inter
        total_union += batch_union

        progress_bar.set_postfix(loss=loss.item())

    n = len(loader.dataset)
    epoch_miou = mean_iou(iou_per_class(total_intersection, total_union))
    return running_loss / n, running_acc / n, epoch_miou


# ============================================================
# Main training entry point
# ============================================================
def main():
    print(f"Using device: {DEVICE}")

    model = SegFormer(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc, train_miou = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_miou = evaluate(model, val_loader, criterion, DEVICE)

        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_mIoU={train_miou:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_mIoU={val_miou:.4f} "
            f"({elapsed:.1f}s)"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_miou": val_miou,
            },
            LAST_CHECKPOINT_PATH,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_miou": val_miou,
                },
                BEST_CHECKPOINT_PATH,
            )
            print(f"  -> New best model saved (val_loss={val_loss:.4f}, val_mIoU={val_miou:.4f})")

    print("Training complete.")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"Last checkpoint: {LAST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()