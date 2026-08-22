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
    enhance_image,
)
from postprocessing.morphology import apply_morphology

try:
    from postprocessing.crf import apply_crf, CRF_AVAILABLE
except ImportError:
    CRF_AVAILABLE = False

try:
    from postprocessing.advanced import (
        apply_superpixel_refinement,
        apply_bilateral_filter,
        apply_confidence_filtering,
        apply_edge_aware_refinement,
        apply_background_cleanup,
        calculate_scene_complexity,
        verify_segmentation_with_segearth,
        apply_class_specific_refinement,
        apply_iterative_refinement,
        apply_texture_aware_processing,
        apply_class_specific_thresholds,
        process_image_with_overlap_stitching,
        SKIMAGE_AVAILABLE,
        CV2_AVAILABLE
    )
    SEGEARTH_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    CV2_AVAILABLE = False
    SEGEARTH_AVAILABLE = False

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
    # Apply image enhancement before processing
    img = enhance_image(img)
    
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
    postprocess: bool = True,
    use_crf: bool = True,
    use_advanced: bool = True,
    confidence_threshold: float = 0.6,
    use_multi_scale: bool = True,
    temperature: float = 0.75,
) -> np.ndarray:
    """
    Run inference on a PIL image with enhanced post-processing.

    Args:
        model_name: Name of the model ('unet' or 'segformer')
        img: PIL Image to process
        postprocess: Apply morphological operations (default True)
        use_crf: Apply CRF post-processing if available (default True)
        use_advanced: Apply advanced post-processing if available (default True)
        confidence_threshold: Minimum confidence for predictions (0-1)
        use_multi_scale: Use multi-scale processing for complex images (default True)
        temperature: Temperature scaling for softmax (lower = sharper predictions, default 0.75)

    Returns:
        2D numpy array (H, W) of predicted class indices.
    """
    model = get_model(model_name)
    
    # Multi-scale processing for complex images
    if use_multi_scale:
        # More aggressive multi-scale with 5 scales for better detail capture
        scales = [0.6, 0.8, 1.0, 1.2, 1.4]
        scale_weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # Weight center scale more heavily
        all_predictions = []
        all_probs = []
        
        for scale, weight in zip(scales, scale_weights):
            # Scale image
            original_size = img.size
            scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled_img = img.resize(scaled_size, Image.BICUBIC)
            
            # Get prediction at this scale
            img_tensor = prepare_image_tensor(scaled_img).to(DEVICE)
            logits = model(img_tensor)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # Resize back to original size
            from PIL import Image as PILImage
            pred_pil = PILImage.fromarray(pred.astype(np.uint8))
            pred_resized = pred_pil.resize(original_size, Image.NEAREST)
            all_predictions.append(np.array(pred_resized) * weight)
            
            # Also resize probabilities
            weighted_probs = probs * weight
            for c in range(weighted_probs.shape[0]):
                prob_pil = PILImage.fromarray((weighted_probs[c] * 255).astype(np.uint8))
                weighted_probs[c] = np.array(prob_pil.resize(original_size, Image.BILINEAR)) / 255.0
            
            all_probs.append(weighted_probs)
        
        # Weighted average predictions and probabilities
        pred = np.sum(all_predictions, axis=0).astype(np.int64)
        probs = np.sum(all_probs, axis=0)
    else:
        # Standard single-scale processing
        img_tensor = prepare_image_tensor(img).to(DEVICE)
        logits = model(img_tensor)
        
        # Apply temperature scaling for better calibrated probabilities
        # Lower temperature = sharper, more confident predictions
        scaled_logits = logits / temperature
        
        # Get softmax probabilities for CRF (using temperature-scaled logits)
        probs = torch.softmax(scaled_logits, dim=1).squeeze(0).cpu().numpy()
        pred = torch.argmax(scaled_logits, dim=1).squeeze(0).cpu().numpy()
        
        # Diagnostic: Check forest vs water confusion at model output
        forest_class = 6
        water_class = 4
        forest_pixels = np.sum(pred == forest_class)
        water_pixels = np.sum(pred == water_class)
        if water_pixels > 0 or forest_pixels > 0:
            avg_forest_prob = np.mean(probs[forest_class, :, :])
            avg_water_prob = np.mean(probs[water_class, :, :])
            print(f"Model output - Forest: {forest_pixels}px (avg prob: {avg_forest_prob:.3f}), Water: {water_pixels}px (avg prob: {avg_water_prob:.3f})")

    # Calculate scene complexity for adaptive processing
    scene_complexity = 0.5  # Default complexity
    if use_advanced and CV2_AVAILABLE:
        try:
            scene_complexity = calculate_scene_complexity(img, pred, NUM_CLASSES)
            print(f"Scene complexity: {scene_complexity:.2f}")
        except Exception as e:
            print(f"Scene complexity calculation failed: {e}")

    # Apply CRF post-processing if available and enabled
    if use_crf and CRF_AVAILABLE:
        # Adaptive CRF parameters based on scene complexity
        n_iterations = int(10 + 5 * scene_complexity)  # More iterations for complex scenes
        pos_w = 5 + 2 * scene_complexity  # Stronger smoothness for complex scenes
        bi_w = 10 + 5 * scene_complexity  # Stronger bilateral for complex scenes
        
        pred = apply_crf(
            img, 
            probs, 
            num_classes=NUM_CLASSES, 
            n_iterations=n_iterations,
            pos_w=pos_w,
            pos_xy_std=3,
            bi_w=bi_w,
            bi_xy_std=80,
            bi_rgb_std=5
        )

    # Apply optimized post-processing pipeline if available and enabled
    if use_advanced:
        # Simplified, more effective pipeline with fewer sequential operations
        # to avoid over-smoothing and artifact accumulation
        
        # Step 1: Apply class-specific thresholds (more robust than simple argmax)
        try:
            # Pass a copy of probs to avoid modifying the original
            pred = apply_class_specific_thresholds(probs.copy(), fallback_class=1)
        except Exception as e:
            print(f"Class-specific thresholding failed: {e}")
        
        # Step 2: Skip confidence filtering due to parameter compatibility issues
        # The class-specific thresholds already handle confidence-based decisions
        pass
        
        # Step 3: Apply superpixel refinement only for medium-complexity scenes
        # Skip for very simple or very complex scenes to avoid over-processing
        if SKIMAGE_AVAILABLE and CV2_AVAILABLE and 0.3 < scene_complexity < 0.8:
            try:
                pred = apply_superpixel_refinement(img, pred, num_classes=NUM_CLASSES, 
                                                n_segments=800, adaptive_segments=True)
            except Exception as e:
                print(f"Superpixel refinement failed: {e}")
        
        # Step 4: Apply edge-aware refinement for all scenes (helps boundary alignment)
        if CV2_AVAILABLE:
            try:
                pred = apply_edge_aware_refinement(img, pred, num_classes=NUM_CLASSES)
            except Exception as e:
                print(f"Edge-aware refinement failed: {e}")
        
        # Step 5: Apply class-specific refinement for ALL scenes (not just complex)
        # This is critical for building detection and other class-specific fixes
        try:
            pred = apply_class_specific_refinement(img, pred, probs, NUM_CLASSES)
        except Exception as e:
            print(f"Class-specific refinement failed: {e}")

    # Apply morphological operations if requested
    if postprocess:
        # Adaptive morphology parameters based on complexity
        max_hole_size = int(64 + 32 * scene_complexity)  # Larger holes for complex scenes
        pred = apply_morphology(pred, road_class=3, building_class=2, 
                              background_class=1, max_hole_size=max_hole_size)

    return pred


def predict_image(
    model_name: str,
    image_path: Path,
    postprocess: bool = False,
    use_crf: bool = True,
    use_advanced: bool = True,
    confidence_threshold: float = 0.6,
    use_multi_scale: bool = False,
    temperature: float = 1.0,
) -> np.ndarray:
    """Run inference on a single image file."""
    return predict_from_image(
        model_name, 
        Image.open(image_path), 
        postprocess=postprocess,
        use_crf=use_crf,
        use_advanced=use_advanced,
        confidence_threshold=confidence_threshold,
        use_multi_scale=use_multi_scale,
        temperature=temperature
    )


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
