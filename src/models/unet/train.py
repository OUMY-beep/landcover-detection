"""
Training script for U-Net on the LoveDA semantic segmentation dataset.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# --- Make sibling packages importable (dataset, preprocessing, models, training) ---
# This file lives at src/models/unet/train.py, so we need to go up 3 levels
# (unet -> models -> src) to reach src/, the root that contains all sibling packages.
SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(SRC_ROOT))

from dataset.loader import loaders  # dict: {"train": DataLoader, "val": ..., "test": ...}
from models.unet import UNet
from preprocessing.transforms import NUM_CLASSES
from training.metrics import (
    IGNORE_INDEX,
    intersection_and_union,
    iou_per_class,
    mean_iou,
    pixel_accuracy,
)
from training.losses import CEDiceLoss

# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 35
LEARNING_RATE = 1e-4
GRAD_CLIP_NORM = 1.0  # max gradient norm; None to disable
CHECKPOINT_DIR = SRC_ROOT.parent / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "unet_best.pth"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "unet_last.pth"


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


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total_union = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    with torch.inference_mode():
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

    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=True).to(DEVICE)
    criterion = CEDiceLoss(ignore_index=IGNORE_INDEX)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    start_epoch = 0
    best_val_miou = -1.0
    epochs_no_improve = 0
    patience = 10

    if BEST_CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint: {BEST_CHECKPOINT_PATH}")
        checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_miou = checkpoint.get("val_miou", -1.0)
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc, train_miou = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_miou = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()
        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"LR={optimizer.param_groups[0]['lr']:.2e} "
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

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            epochs_no_improve = 0
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
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                break

    print("Training complete.")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"Last checkpoint: {LAST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()