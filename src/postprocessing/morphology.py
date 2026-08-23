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

    # Do not run sequential opening/closing for every class. Those operations
    # let later classes overwrite earlier ones and erase small true buildings,
    # roads, water channels, and field boundaries. The conservative edits
    # above are the only changes justified without model probabilities.
    # Retained for the public API; broad road closing is unsafe here.
    del road_class
    return processed


def apply_morphology_batch(preds: torch.Tensor, road_class: int = 3, building_class: int = 2, background_class: int = 1) -> torch.Tensor:
    device = preds.device
    output = np.stack([
        apply_morphology(mask, road_class=road_class, building_class=building_class, background_class=background_class)
        for mask in preds.detach().cpu().numpy()
    ])
    return torch.from_numpy(output).to(device)
