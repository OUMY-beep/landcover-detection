"""
inference.py
============
Model loading, inference, mask colorization and per-image metrics.

Wraps the existing project code in ``src/`` so the Flask backend can
call ``predict_image()`` and ``compute_per_image_metrics()`` without
duplicating any logic.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Project paths ─────────────────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parents[1]
SRC_ROOT     = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from evaluation.load_models import load_unet, load_segformer
from preprocessing.transforms import (
    MEAN, STD, NUM_CLASSES, IGNORE_INDEX,
    test_image_transform, test_joint_transform, test_mask_transform,
)
from postprocessing.morphology import apply_morphology

# ── Class info ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Ignore", "Background", "Building", "Road", "Water",
    "Barren", "Forest", "Agricultural",
]

# Tab10 palette (RGBA 0-255) — same as matplotlib scripts
_TAB10 = [
    (31,  119, 180, 180),  # 0 Ignore
    (255, 127, 14,  180),  # 1 Background
    (44,  160, 44,  180),  # 2 Building
    (214, 39,  40,  180),  # 3 Road
    (148, 103, 189, 180),  # 4 Water
    (140, 86,  75,  180),  # 5 Barren
    (227, 119, 194, 180),  # 6 Forest
    (127, 127, 127, 180),  # 7 Agricultural
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Singleton model cache ─────────────────────────────────────────────────────
_models: dict[str, torch.nn.Module] = {}


def get_model(name: str) -> torch.nn.Module:
    """Load a model by name ('unet' or 'segformer'), cached after first call."""
    if name not in _models:
        if name == "unet":
            _models[name] = load_unet(device=DEVICE)
        elif name == "segformer":
            _models[name] = load_segformer(device=DEVICE)
        else:
            raise ValueError(f"Unknown model: {name}")
    return _models[name]


# ── Image loading ─────────────────────────────────────────────────────────────

def prepare_image_tensor(img: Image.Image) -> torch.Tensor:
    """
    Apply test transforms (resize + normalize) to a PIL image.
    Returns a tensor of shape (1, C, H, W) ready for inference.
    """
    img = img.convert("RGB")
    dummy_mask = Image.new("L", img.size, 0)
    img, _ = test_joint_transform(img, dummy_mask)
    img = test_image_transform(img)
    return img.unsqueeze(0)


def load_and_transform_image(image_path: Path) -> torch.Tensor:
    """Load a satellite image from disk and prepare it for inference."""
    return prepare_image_tensor(Image.open(image_path))


def resize_image_for_display(img: Image.Image) -> Image.Image:
    """Resize an image to the model input size for consistent display."""
    img = img.convert("RGB")
    dummy_mask = Image.new("L", img.size, 0)
    img, _ = test_joint_transform(img, dummy_mask)
    return img


def load_mask(mask_path: Path) -> np.ndarray:
    """
    Load a ground-truth mask and apply the same resize as test images.
    Returns a 2D numpy array of class indices (H, W).
    """
    mask = Image.open(mask_path).convert("L")
    dummy_img = Image.new("RGB", mask.size, (0, 0, 0))
    _, mask = test_joint_transform(dummy_img, mask)
    mask = test_mask_transform(mask)
    return mask.numpy().astype(np.int64)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_from_image(
    model_name: str,
    img: Image.Image,
    postprocess: bool = False,
) -> np.ndarray:
    """
    Run inference on a PIL image.

    Returns:
        2D numpy array (H, W) of predicted class indices.
    """
    model = get_model(model_name)
    img_tensor = prepare_image_tensor(img).to(DEVICE)
    logits = model(img_tensor)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    if postprocess:
        pred = apply_morphology(pred, road_class=3, building_class=2)

    return pred


def predict_image(
    model_name: str,
    image_path: Path,
    postprocess: bool = False,
) -> np.ndarray:
    """Run inference on a single image file."""
    return predict_from_image(model_name, Image.open(image_path), postprocess=postprocess)


# ── Mask colorization ────────────────────────────────────────────────────────

def colorize_mask(mask: np.ndarray, alpha: int = 180) -> bytes:
    """
    Convert a 2D class-index array to a colored RGBA PNG (as bytes).
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for cls_idx in range(NUM_CLASSES):
        if cls_idx < len(_TAB10):
            r, g, b, a = _TAB10[cls_idx]
        else:
            r, g, b, a = (0, 0, 0, 0)
        where = mask == cls_idx
        rgba[where] = [r, g, b, alpha]

    # Ignore pixels (255) are fully transparent
    rgba[mask == IGNORE_INDEX] = [0, 0, 0, 0]

    img = Image.fromarray(rgba, "RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def serve_raw_image(image_path: Path) -> bytes:
    """Load an image and return it as PNG bytes."""
    return serve_pil_image(Image.open(image_path))


def serve_pil_image(img: Image.Image) -> bytes:
    """Return a resized RGB image as PNG bytes."""
    img = resize_image_for_display(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Per-image metrics ────────────────────────────────────────────────────────

def compute_per_image_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> dict:
    """
    Compute mIoU, pixel accuracy, and IoU per class for a single image.
    """
    valid = gt != IGNORE_INDEX
    pred_v = pred[valid]
    gt_v = gt[valid]

    total_pixels = valid.sum()
    correct = (pred_v == gt_v).sum()
    pixel_acc = float(correct / total_pixels) if total_pixels > 0 else 0.0

    ious = {}
    for cls in range(NUM_CLASSES):
        inter = ((pred_v == cls) & (gt_v == cls)).sum()
        union = ((pred_v == cls) | (gt_v == cls)).sum()
        if union > 0:
            ious[cls] = float(inter / union)

    mean_iou = sum(ious.values()) / len(ious) if ious else 0.0

    iou_per_class = []
    for cls in range(NUM_CLASSES):
        iou_per_class.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}",
            "iou": ious.get(cls),
        })

    return {
        "pixel_accuracy": round(pixel_acc, 4),
        "mean_iou": round(mean_iou, 4),
        "iou_per_class": iou_per_class,
    }
