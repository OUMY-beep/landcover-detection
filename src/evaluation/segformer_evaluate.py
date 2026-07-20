"""
Evaluate a trained SegFormer checkpoint on the LoveDA test set:
Pixel Accuracy, per-class IoU, Mean IoU, Mean Dice.
"""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_ROOT))

from dataset.loader import loaders
from evaluation.segformer_predict import DEFAULT_CHECKPOINT, DEVICE, load_model
from preprocessing.transforms import NUM_CLASSES
from training.metrics import (
    dice_per_class,
    intersection_and_union,
    iou_per_class,
    mean_dice,
    mean_iou,
    pixel_accuracy,
)

CLASS_NAMES = [
    "Background", "Building", "Road", "Water",
    "Barren", "Forest", "Agricultural", "Classe 7",
]


@torch.no_grad()
def evaluate_on_test(model, loader, device=DEVICE):
    """
    Run the model over the full loader and accumulate metrics.

    Returns:
        dict with pixel_accuracy, iou_per_class, mean_iou, dice_per_class, mean_dice
    """
    model.eval()

    running_acc = 0.0
    total_samples = 0
    total_intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total_union = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        running_acc += pixel_accuracy(logits, masks) * images.size(0)
        total_samples += images.size(0)

        batch_inter, batch_union = intersection_and_union(logits, masks)
        total_intersection += batch_inter
        total_union += batch_union

    iou = iou_per_class(total_intersection, total_union)
    dice = dice_per_class(total_intersection, total_union)

    return {
        "pixel_accuracy": running_acc / total_samples,
        "iou_per_class": iou,
        "mean_iou": mean_iou(iou),
        "dice_per_class": dice,
        "mean_dice": mean_dice(dice),
    }


def print_report(results: dict) -> None:
    print("=" * 60)
    print("TEST SET EVALUATION — SegFormer")
    print("=" * 60)
    print(f"Pixel Accuracy : {results['pixel_accuracy'] * 100:.1f}%")
    print(f"Mean IoU       : {results['mean_iou'] * 100:.1f}%")
    print(f"Mean Dice      : {results['mean_dice'] * 100:.1f}%")
    print()
    print("Per-class IoU / Dice")
    print("-" * 60)
    for i, name in enumerate(CLASS_NAMES):
        iou_value = results["iou_per_class"][i].item()
        dice_value = results["dice_per_class"][i].item()
        iou_str = f"{iou_value * 100:.1f}%" if iou_value == iou_value else "N/A"  # NaN check
        dice_str = f"{dice_value * 100:.1f}%" if dice_value == dice_value else "N/A"
        print(f"{name:15s} IoU: {iou_str:>8s}   Dice: {dice_str:>8s}")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    model = load_model(DEFAULT_CHECKPOINT, DEVICE)
    test_loader = loaders["test"]

    results = evaluate_on_test(model, test_loader, DEVICE)
    print_report(results)