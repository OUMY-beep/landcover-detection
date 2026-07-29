"""
Training script for SegFormer-B0 on the LoveDA semantic segmentation dataset.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from training.losses import CEDiceLoss

# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 90
LEARNING_RATE = 6e-5  # lower than U-Net: fine-tuning a pretrained transformer backbone
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0  # max gradient norm; None to disable
CHECKPOINT_DIR = SRC_ROOT.parent / "outputs" / "models"
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

    model = SegFormer(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = CEDiceLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
        
        # Migrate older HF transformers keys (encoder) to newer keys (stages) if needed
        migrated_state_dict = {}
        for k, v in checkpoint["model_state_dict"].items():
            new_k = k
            if "encoder.patch_embeddings." in k:
                parts = k.split("encoder.patch_embeddings.")
                sub_parts = parts[1].split(".", 1)
                new_k = f"{parts[0]}stages.{sub_parts[0]}.patch_embeddings.{sub_parts[1]}"
            elif "encoder.block." in k:
                parts = k.split("encoder.block.")
                sub_parts = parts[1].split(".", 2)
                stage_idx, block_idx, rest = sub_parts[0], sub_parts[1], sub_parts[2]
                rest = rest.replace("layer_norm_1", "layernorm_before")
                rest = rest.replace("layer_norm_2", "layernorm_after")
                rest = rest.replace("attention.self.query", "attention.q_proj")
                rest = rest.replace("attention.self.key", "attention.k_proj")
                rest = rest.replace("attention.self.value", "attention.v_proj")
                rest = rest.replace("attention.self.sr", "attention.sequence_reduction.sequence_reduction")
                rest = rest.replace("attention.self.layer_norm", "attention.sequence_reduction.layer_norm")
                rest = rest.replace("attention.output.dense", "attention.o_proj")
                rest = rest.replace("mlp.dense1", "mlp.fc1")
                rest = rest.replace("mlp.dense2", "mlp.fc2")
                new_k = f"{parts[0]}stages.{stage_idx}.blocks.{block_idx}.{rest}"
            elif "encoder.layer_norm." in k:
                parts = k.split("encoder.layer_norm.")
                sub_parts = parts[1].split(".", 1)
                new_k = f"{parts[0]}stages.{sub_parts[0]}.layer_norm.{sub_parts[1]}"
            elif "decode_head.linear_c." in k:
                new_k = k.replace("decode_head.linear_c.", "decode_head.linear_projections.")
                
            # Skip ADE20K 150-class classifier
            if "decode_head.classifier" in new_k:
                continue

            migrated_state_dict[new_k] = v
            
        missing, unexpected = model.load_state_dict(
            migrated_state_dict,
            strict=False
        )
        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_miou = checkpoint.get("val_miou", -1.0)
        
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            scheduler.last_epoch = start_epoch - 1
            
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        
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
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_miou": val_miou,
                "epochs_no_improve": epochs_no_improve,
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
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_miou": val_miou,
                    "epochs_no_improve": epochs_no_improve,
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
    main()