"""Conservative, class-safe cleanup for segmentation masks."""

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi


def apply_morphology(
    pred_mask: np.ndarray,
    road_class: int = 3,
    building_class: int = 2,
    background_class: int = 1,
    max_hole_size: int = 64,
) -> np.ndarray:
    """Remove only clear one-pixel speckle and tiny enclosed building holes.

    This deliberately never uses image inpainting or a large closing kernel.
    Those operations invent class IDs and can spread roads/buildings across
    unrelated land cover. Class 0 (``Ignore``) is always preserved.
    """
    if pred_mask.ndim != 2:
        raise ValueError("pred_mask must be a 2D class-index array")

    processed = pred_mask.astype(np.int64, copy=True)
    h, w = processed.shape

    # A pixel is a speckle only when all eight surrounding pixels agree on a
    # different valid class. This cannot erase thin roads or small buildings.
    if h >= 3 and w >= 3:
        centre = processed[1:-1, 1:-1]
        neighbours = np.stack([
            processed[:-2, :-2], processed[:-2, 1:-1], processed[:-2, 2:],
            processed[1:-1, :-2],                       processed[1:-1, 2:],
            processed[2:, :-2],  processed[2:, 1:-1],  processed[2:, 2:],
        ])
        unanimous = np.all(neighbours == neighbours[0], axis=0)
        replacement = neighbours[0]
        change = (
            unanimous
            & (centre != replacement)
            & (centre != 0)
            & (replacement != 0)
        )
        centre[change] = replacement[change]

    # Fill only small, fully enclosed holes in predicted building regions.
    if building_class is not None:
        building = processed == building_class
        holes = ndi.binary_fill_holes(building) & ~building
        count, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] <= max_hole_size:
                fill = (labels == label) & (processed != 0)
                processed[fill] = building_class

    # Additional: Remove very small isolated regions (speckles) for all classes except ignore
    # Using a reasonable default for LoveDA (8 classes)
    num_classes = 8
    # Apply class-specific morphological operations for boundary cleanup
    class_kernel_sizes = {
        0: 3,  # Ignore
        1: 5,  # Background - larger kernel for smoother regions
        2: 3,  # Building - smaller kernel to preserve details
        3: 5,  # Road - larger kernel for connectivity
        4: 7,  # Water - larger kernel for smooth boundaries
        5: 3,  # Barren - smaller kernel
        6: 5,  # Forest - medium kernel for canopy regions
        7: 5,  # Agricultural - medium kernel for field uniformity
    }
    
    refined = processed.copy()
    for c in range(num_classes):
        class_mask = (refined == c).astype(np.uint8)
        
        # Use class-specific kernel size
        kernel_size = class_kernel_sizes.get(c, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Closing to fill small holes (more iterations for larger kernels)
        iterations = 2 if kernel_size >= 5 else 1
        closed = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # Opening to remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        # Update refined prediction - preserve original class where it was
        refined[class_mask == 1] = c
        refined[opened == 1] = c
    
    # Background-specific cleanup
    if background_class is not None:
        # Remove small non-background regions that are likely noise
        background_mask = refined == background_class
        inverted_mask = ~background_mask & (refined != 0)  # Non-background, non-ignore regions
        
        # Find small isolated non-background regions and convert to background
        count, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_mask.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] <= max_hole_size:  # Small isolated regions
                # Check if region is surrounded by background
                region_mask = labels == label
                # Simple check: if region is small and isolated, make it background
                refined[region_mask] = background_class
    
    processed = refined

    # Retained for the public API. Broad road closing is unsafe without model
    # probabilities to justify every new road pixel.
    del road_class
    return processed


def apply_morphology_batch(preds: torch.Tensor, road_class: int = 3, building_class: int = 2, background_class: int = 1) -> torch.Tensor:
    device = preds.device
    output = np.stack([
        apply_morphology(mask, road_class=road_class, building_class=building_class, background_class=background_class)
        for mask in preds.detach().cpu().numpy()
    ])
    return torch.from_numpy(output).to(device)
