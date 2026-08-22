"""
Segmentation metrics for LoveDA: pixel accuracy, IoU per class, mean IoU, Dice.

All functions accept:
    logits: (B, C, H, W) raw model output
    masks:  (B, H, W) ground-truth class indices

and ignore pixels equal to `ignore_index` when computing scores.
"""

import torch

NUM_CLASSES = 8
IGNORE_INDEX = 255


def pixel_accuracy(logits: torch.Tensor, masks: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> float:
    """Fraction of valid (non-ignored) pixels correctly classified in a batch."""
    preds = torch.argmax(logits, dim=1)
    valid = masks != ignore_index

    correct = (preds[valid] == masks[valid]).sum().item()
    total = valid.sum().item()

    return correct / total if total > 0 else 0.0


def intersection_and_union(
    logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-class intersection and union counts for one batch, ignoring
    `ignore_index` pixels. Returns two tensors of shape (num_classes,).
    """
    preds = torch.argmax(logits, dim=1)
    valid = masks != ignore_index

    preds = preds[valid]
    masks = masks[valid]

    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)

    for cls in range(num_classes):
        pred_cls = preds == cls
        mask_cls = masks == cls
        intersection[cls] = (pred_cls & mask_cls).sum()
        union[cls] = (pred_cls | mask_cls).sum()

    return intersection, union


def iou_per_class(intersection: torch.Tensor, union: torch.Tensor) -> torch.Tensor:
    """
    Per-class IoU from accumulated intersection/union counts.
    Classes with zero union (absent from both preds and ground truth) are
    returned as NaN so they can be excluded from the mean.
    """
    iou = intersection / union.clamp(min=1e-8)
    iou[union == 0] = float("nan")
    return iou


def mean_iou(iou: torch.Tensor) -> float:
    """Mean IoU across classes present in the accumulated data (ignores NaN)."""
    valid = ~torch.isnan(iou)
    return iou[valid].mean().item() if valid.any() else 0.0


def dice_per_class(intersection: torch.Tensor, union: torch.Tensor) -> torch.Tensor:
    """
    Per-class Dice score from accumulated intersection/union counts.
    Dice = 2*I / (I + U) since U = |A| + |B| - I, so |A|+|B| = U + I.
    """
    dice = (2 * intersection) / (union + intersection).clamp(min=1e-8)
    dice[union == 0] = float("nan")
    return dice


def mean_dice(dice: torch.Tensor) -> float:
    """Mean Dice across classes present in the accumulated data (ignores NaN)."""
    valid = ~torch.isnan(dice)
    return dice[valid].mean().item() if valid.any() else 0.0
