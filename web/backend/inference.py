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

try:
    import cv2
except ImportError:
    cv2 = None

# ── Project paths ─────────────────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parents[1]
SRC_ROOT     = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from evaluation.load_models import load_segformer
from preprocessing.transforms import (
    IMAGE_SIZE, MEAN, STD, NUM_CLASSES, IGNORE_INDEX,
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

WATER_CLASS = 4
FOREST_CLASS = 6
AGRICULTURAL_CLASS = 7
BARREN_CLASS = 5

def calibrate_landcover_probabilities(probs: np.ndarray) -> np.ndarray:
    """Apply conservative class calibration before choosing a label.

    The saved model is biased toward Water on the user-supplied agricultural
    imagery.  This is probability calibration, not a class remap: all eight
    shifted IDs and the trained weights remain exactly the same.  Strong water
    predictions remain water after renormalisation; only close decisions can
    move to the land class the model also supports.
    """
    if probs.ndim != 3 or probs.shape[0] <= AGRICULTURAL_CLASS:
        raise ValueError("probs must have shape (at least 8 classes, height, width)")

    class_weights = np.ones(probs.shape[0], dtype=probs.dtype)
    class_weights[WATER_CLASS] = 0.70
    class_weights[BARREN_CLASS] = 1.10
    class_weights[FOREST_CLASS] = 1.10
    class_weights[AGRICULTURAL_CLASS] = 1.15
    calibrated = probs * class_weights[:, np.newaxis, np.newaxis]
    return calibrated / np.maximum(calibrated.sum(axis=0, keepdims=True), 1e-8)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Singleton model cache ─────────────────────────────────────────────────────
_models: dict[str, torch.nn.Module] = {}


def get_model(name: str) -> torch.nn.Module:
    """Load the SegFormer model, cached after first call."""
    if name not in _models:
        if name == "segformer":
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


def resolve_forest_water_ambiguity(
    pred: np.ndarray,
    probs: np.ndarray,
    image: Image.Image | None = None,
    max_margin: float = 0.10,
    max_open_land_margin: float = 0.12,
    min_forest_probability: float = 0.10,
    texture_threshold: float = 160.0,
) -> np.ndarray:
    """Correct water predictions that have model or texture evidence of forest.

    This keeps the project's shifted class IDs (water=4, forest=6) and leaves
    smooth water unchanged.  The texture check targets the common SegFormer
    failure mode in large satellite scenes: green, high-detail tree canopy
    predicted as water even when forest is only its best alternative class.
    """
    if probs.ndim != 3 or probs.shape[0] <= AGRICULTURAL_CLASS:
        raise ValueError("probs must have shape (at least 8 classes, height, width)")
    if pred.shape != probs.shape[1:]:
        raise ValueError("pred shape must match the probability-map dimensions")

    water_probs = probs[WATER_CLASS]
    forest_probs = probs[FOREST_CLASS]
    agricultural_probs = probs[AGRICULTURAL_CLASS]
    barren_probs = probs[BARREN_CLASS]
    open_land_probs = np.maximum(agricultural_probs, barren_probs)
    forest_like_water = (
        (pred == WATER_CLASS)
        & (water_probs > min_forest_probability)
        & (forest_probs >= water_probs - max_margin)
    )
    # Agricultural and Barren use the same evidence-based rule as Forest.
    # It resolves a close Water-vs-land decision without replacing a strongly
    # supported river or lake with land.
    open_land_like_water = (
        (pred == WATER_CLASS)
        & (water_probs <= 0.95)
        & (open_land_probs >= water_probs - max_open_land_margin)
    )

    if image is not None and cv2 is not None:
        height, width = pred.shape
        image_array = np.array(
            image.convert("RGB").resize((width, height), Image.BILINEAR)
        )
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
        local_mean = cv2.boxFilter(gray, -1, (7, 7))
        local_variance = cv2.boxFilter(gray * gray, -1, (7, 7)) - local_mean * local_mean

        # A single sharp water-edge pixel is not enough evidence.  Forest
        # texture must also be present throughout its local neighbourhood.
        textured = (local_variance >= texture_threshold).astype(np.float32)
        texture_support = cv2.boxFilter(textured, -1, (5, 5)) >= 0.70
        non_water_probs = probs.copy()
        non_water_probs[WATER_CLASS] = -np.inf
        forest_is_best_alternative = forest_probs >= np.max(non_water_probs, axis=0)
        vegetation_probs = np.maximum(forest_probs, agricultural_probs)
        vegetation_is_best_alternative = vegetation_probs >= np.max(non_water_probs, axis=0)
        textured_forest = (
            (pred == WATER_CLASS)
            & texture_support
            & (
                (forest_probs >= min_forest_probability / 2)
                | forest_is_best_alternative
            )
        )
        forest_like_water |= textured_forest

        # SegFormer can favour Water very strongly over a plantation or crop
        # field. If its best non-water explanation is still vegetation and
        # the image has sustained canopy texture, use that vegetation class.
        textured_vegetation = (
            (pred == WATER_CLASS)
            & texture_support
            & vegetation_is_best_alternative
        )

        # Some SegFormer false positives are so confident that neither forest
        # nor agricultural is its second-highest class.  A dark green canopy
        # still has a recognisable signature: sustained fine texture, moderate
        # green/yellow hue, and lower brightness than open fields or smooth
        # blue-green water.  This deliberately requires *all* three signals;
        # it is not a blanket "green means forest" rule.
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        canopy_coloured = (
            (hsv[:, :, 0] >= 25)
            & (hsv[:, :, 0] <= 65)
            & (hsv[:, :, 1] >= 25)
            & (hsv[:, :, 2] <= 125)
        )
        textured_canopy = (pred == WATER_CLASS) & texture_support & canopy_coloured

        # Fields and exposed soil can also be confused with Water, especially
        # when a large image is reduced before inference.  Change them only
        # when an existing land class has meaningful model support and the
        # local scene contains repeated land texture.  This deliberately does
        # not turn low-texture or high-confidence water into land.
        open_land_is_best_alternative = open_land_probs >= np.max(non_water_probs, axis=0)
        land_textured = (local_variance >= 45.0).astype(np.float32)
        land_texture_support = cv2.boxFilter(land_textured, -1, (5, 5)) >= 0.70
        textured_open_land = (
            (pred == WATER_CLASS)
            & (water_probs <= 0.92)
            & land_texture_support
            & (
                (open_land_probs >= 0.06)
                | open_land_is_best_alternative
            )
        )
        open_land_like_water |= textured_open_land

    corrected = pred.copy()
    corrected[forest_like_water] = FOREST_CLASS
    if image is not None and cv2 is not None:
        corrected[textured_vegetation] = np.where(
            forest_probs[textured_vegetation] >= agricultural_probs[textured_vegetation],
            FOREST_CLASS,
            AGRICULTURAL_CLASS,
        )
        corrected[textured_canopy] = FOREST_CLASS
        corrected[textured_open_land] = np.where(
            agricultural_probs[textured_open_land] >= barren_probs[textured_open_land],
            AGRICULTURAL_CLASS,
            BARREN_CLASS,
        )
    corrected[open_land_like_water] = np.where(
        agricultural_probs[open_land_like_water] >= barren_probs[open_land_like_water],
        AGRICULTURAL_CLASS,
        BARREN_CLASS,
    )
    return corrected


def resolve_field_like_water_components(
    pred: np.ndarray,
    probs: np.ndarray,
    image: Image.Image | None,
    min_area: int = 96,
) -> np.ndarray:
    """Replace broad field-like Water components with the supported land class.

    A pixel-only test misses smooth agricultural parcels.  This component
    check protects true water by keeping blue-green components (lakes) and
    elongated components (rivers/canals), then corrects only large compact
    components whose imagery looks like green crop land or exposed soil.
    """
    if image is None or cv2 is None:
        return pred
    if probs.ndim != 3 or probs.shape[0] <= AGRICULTURAL_CLASS:
        raise ValueError("probs must have shape (at least 8 classes, height, width)")
    if pred.shape != probs.shape[1:]:
        raise ValueError("pred shape must match the probability-map dimensions")

    height, width = pred.shape
    image_array = np.array(image.convert("RGB").resize((width, height), Image.BILINEAR))
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (7, 7))
    variance = cv2.boxFilter(gray * gray, -1, (7, 7)) - mean * mean

    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Cultivated and barren surfaces in the supplied scenes are green/brown,
    # moderately saturated, and not the high-frequency canopy handled by the
    # forest safeguard.
    field_like = (
        (hue >= 15)
        & (hue <= 45)
        & (saturation >= 35)
        & (value >= 115)
        & (variance <= 200.0)
    )
    # The quarry lake in the examples is green-blue and brighter than fields.
    # This is deliberately only positive water evidence, not a universal
    # definition of water; elongated rivers are protected separately below.
    blue_green_water = (
        (image_array[:, :, 2] >= image_array[:, :, 0] * 0.80)
        & (image_array[:, :, 1] >= image_array[:, :, 0] + 8)
        & (value >= 130)
    )

    water = (pred == WATER_CLASS).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=8)
    corrected = pred.copy()
    agricultural_probs = probs[AGRICULTURAL_CLASS]
    barren_probs = probs[BARREN_CLASS]
    model_land_choice = np.where(
        agricultural_probs >= barren_probs, AGRICULTURAL_CLASS, BARREN_CLASS
    )
    visual_land_choice = np.where(hue >= 28, AGRICULTURAL_CLASS, BARREN_CLASS)
    has_land_support = np.maximum(agricultural_probs, barren_probs) >= 0.04

    for component in range(1, component_count):
        area = stats[component, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        component_mask = labels == component
        component_width = stats[component, cv2.CC_STAT_WIDTH]
        component_height = stats[component, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(component_width, component_height) / max(1, min(component_width, component_height))
        if aspect_ratio >= 3.5:
            continue
        if np.mean(blue_green_water[component_mask]) >= 0.20:
            continue
        if np.mean(field_like[component_mask]) < 0.55:
            continue

        corrected[component_mask] = np.where(
            has_land_support[component_mask],
            model_land_choice[component_mask],
            visual_land_choice[component_mask],
        )

    return corrected


@torch.no_grad()
def predict_large_image_with_tiles(
    model: torch.nn.Module,
    img: Image.Image,
    grid_size: int = 2,
    global_weight: float = 0.25,
    overlap_fraction: float = 0.15,
) -> np.ndarray:
    """Blend global and tiled probabilities onto the normal mask size.

    Very large source scenes lose important canopy and shoreline detail when
    they are resized as a single image, while independent tiles lose the
    wider context needed to distinguish fields, barren ground, and large
    water bodies. This retains both views without changing the model.
    """
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")
    if not 0.0 <= global_weight < 1.0:
        raise ValueError("global_weight must be in the range [0.0, 1.0)")
    if not 0.0 <= overlap_fraction < 0.5:
        raise ValueError("overlap_fraction must be in the range [0.0, 0.5)")

    output_height, output_width = IMAGE_SIZE
    probabilities = np.zeros((NUM_CLASSES, output_height, output_width), dtype=np.float32)
    probability_weights = np.full(
        (output_height, output_width), global_weight, dtype=np.float32
    )
    if global_weight:
        global_tensor = prepare_image_tensor(img).to(DEVICE)
        global_probabilities = torch.softmax(model(global_tensor), dim=1)
        global_probabilities = torch.nn.functional.interpolate(
            global_probabilities,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).cpu().numpy()
        probabilities = global_probabilities * global_weight
    tile_weight = 1.0 - global_weight
    source_width, source_height = img.size

    for row in range(grid_size):
        output_top = output_height * row // grid_size
        output_bottom = output_height * (row + 1) // grid_size
        vertical_overlap = int(round((output_bottom - output_top) * overlap_fraction))
        tile_output_top = max(0, output_top - vertical_overlap)
        tile_output_bottom = min(output_height, output_bottom + vertical_overlap)

        for column in range(grid_size):
            output_left = output_width * column // grid_size
            output_right = output_width * (column + 1) // grid_size
            horizontal_overlap = int(round((output_right - output_left) * overlap_fraction))
            tile_output_left = max(0, output_left - horizontal_overlap)
            tile_output_right = min(output_width, output_right + horizontal_overlap)

            source_top = source_height * tile_output_top // output_height
            source_bottom = source_height * tile_output_bottom // output_height
            source_left = source_width * tile_output_left // output_width
            source_right = source_width * tile_output_right // output_width

            tile = img.crop((source_left, source_top, source_right, source_bottom))
            tile_tensor = prepare_image_tensor(tile).to(DEVICE)
            tile_probs = torch.softmax(model(tile_tensor), dim=1)
            tile_probs = torch.nn.functional.interpolate(
                tile_probs,
                size=(tile_output_bottom - tile_output_top, tile_output_right - tile_output_left),
                mode="bilinear",
                align_corners=False,
            )
            vertical_weights = np.ones(tile_output_bottom - tile_output_top, dtype=np.float32)
            if tile_output_top < output_top:
                ramp = output_top - tile_output_top
                vertical_weights[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False)
            if tile_output_bottom > output_bottom:
                ramp = tile_output_bottom - output_bottom
                vertical_weights[-ramp:] = np.linspace(1.0, 0.0, ramp, endpoint=False)
            horizontal_weights = np.ones(tile_output_right - tile_output_left, dtype=np.float32)
            if tile_output_left < output_left:
                ramp = output_left - tile_output_left
                horizontal_weights[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False)
            if tile_output_right > output_right:
                ramp = tile_output_right - output_right
                horizontal_weights[-ramp:] = np.linspace(1.0, 0.0, ramp, endpoint=False)
            tile_blend = np.outer(vertical_weights, horizontal_weights) * tile_weight

            probabilities[:, tile_output_top:tile_output_bottom, tile_output_left:tile_output_right] += (
                tile_probs.squeeze(0).cpu().numpy() * tile_blend[np.newaxis, :, :]
            )
            probability_weights[
                tile_output_top:tile_output_bottom, tile_output_left:tile_output_right
            ] += tile_blend

    return probabilities / np.maximum(probability_weights[np.newaxis, :, :], 1e-8)


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
    return_probabilities: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a PIL image with enhanced post-processing.

    Args:
        model_name: SegFormer model identifier ('segformer')
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
        if max(img.size) > 2 * max(IMAGE_SIZE):
            image_extent = max(img.size)
            if model_name == "segformer" and image_extent > 8 * max(IMAGE_SIZE):
                grid_size = 4
            elif model_name == "segformer" and image_extent > 4 * max(IMAGE_SIZE):
                grid_size = 3
            else:
                grid_size = 2
            # At 4x4, the local tiles already carry strong scene context.
            # A lighter global contribution avoids blurring narrow field
            # boundaries, canals, roads, and small water bodies.
            global_weight = 0.15 if grid_size >= 4 else 0.25
            probs = predict_large_image_with_tiles(
                model, img, grid_size=grid_size, global_weight=global_weight
            )
        else:
            scales = [0.6, 0.8, 1.0, 1.2, 1.4]
            scale_weights = [0.1, 0.2, 0.4, 0.2, 0.1]
            probabilities = []

            for scale, weight in zip(scales, scale_weights):
                scaled_size = (
                    int(img.size[0] * scale),
                    int(img.size[1] * scale),
                )
                scaled_img = img.resize(scaled_size, Image.BICUBIC)
                img_tensor = prepare_image_tensor(scaled_img).to(DEVICE)
                logits = model(img_tensor)
                probabilities.append(torch.softmax(logits, dim=1).squeeze(0).cpu().numpy() * weight)

            probs = np.sum(probabilities, axis=0)

    else:
        # Standard single-scale processing
        img_tensor = prepare_image_tensor(img).to(DEVICE)
        logits = model(img_tensor)
        
        # Apply temperature scaling for better calibrated probabilities
        # Lower temperature = sharper, more confident predictions
        scaled_logits = logits / temperature
        
        # Get softmax probabilities for CRF (using temperature-scaled logits)
        probs = torch.softmax(scaled_logits, dim=1).squeeze(0).cpu().numpy()
    probs = calibrate_landcover_probabilities(probs)
    # Class IDs are categorical: select the class with the strongest calibrated
    # probability instead of averaging numeric class IDs.
    pred = np.argmax(probs, axis=0).astype(np.int64)

    # Diagnostic: Check forest vs water confusion at model output
    forest_pixels = np.sum(pred == FOREST_CLASS)
    water_pixels = np.sum(pred == WATER_CLASS)
    if water_pixels > 0 or forest_pixels > 0:
        avg_forest_prob = np.mean(probs[FOREST_CLASS, :, :])
        avg_water_prob = np.mean(probs[WATER_CLASS, :, :])
        print(f"Model output - Forest: {forest_pixels}px (avg prob: {avg_forest_prob:.3f}), Water: {water_pixels}px (avg prob: {avg_water_prob:.3f})")

    # Tile inference returns the project's normal mask resolution, whereas a
    # GeoTIFF may be much larger.  Every image-guided post-processing stage
    # must use that same resolution as the probability map.
    probability_height, probability_width = probs.shape[1:]
    processing_image = img
    if img.size != (probability_width, probability_height):
        processing_image = img.convert("RGB").resize(
            (probability_width, probability_height), Image.BILINEAR
        )

    # Calculate scene complexity for adaptive processing
    scene_complexity = 0.5  # Default complexity
    if use_advanced and CV2_AVAILABLE:
        try:
            scene_complexity = calculate_scene_complexity(processing_image, pred, NUM_CLASSES)
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
            processing_image,
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
                pred = apply_superpixel_refinement(processing_image, pred, num_classes=NUM_CLASSES,
                                                n_segments=800, adaptive_segments=True)
            except Exception as e:
                print(f"Superpixel refinement failed: {e}")
        
        # Step 4: Apply edge-aware refinement for all scenes (helps boundary alignment)
        if CV2_AVAILABLE:
            try:
                pred = apply_edge_aware_refinement(processing_image, pred, num_classes=NUM_CLASSES)
            except Exception as e:
                print(f"Edge-aware refinement failed: {e}")
        
        # Step 5: Apply class-specific refinement for ALL scenes (not just complex)
        # This is critical for building detection and other class-specific fixes
        try:
            pred = apply_class_specific_refinement(processing_image, pred, probs, NUM_CLASSES)
        except Exception as e:
            print(f"Class-specific refinement failed: {e}")

    # Apply morphological operations if requested
    if postprocess:
        # Adaptive morphology parameters based on complexity
        max_hole_size = int(64 + 32 * scene_complexity)  # Larger holes for complex scenes
        pred = apply_morphology(pred, road_class=3, building_class=2, 
                              background_class=1, max_hole_size=max_hole_size)

    pred = resolve_forest_water_ambiguity(pred, probs, image=img)
    pred = resolve_field_like_water_components(pred, probs, image=img)
    return (pred, probs) if return_probabilities else pred


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
