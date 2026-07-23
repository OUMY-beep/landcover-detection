import cv2
import numpy as np
import torch

def apply_morphology(pred_mask: np.ndarray, road_class: int = 2, building_class: int = 1) -> np.ndarray:
    """
    Applies morphological closing to specific classes in the prediction mask
    to fix broken roads and small holes in buildings.
    
    Args:
        pred_mask: 2D numpy array of shape (H, W) containing class indices.
        road_class: index of the road class.
        building_class: index of the building class.
        
    Returns:
        Post-processed 2D numpy array of shape (H, W).
    """
    # Create a copy so we don't modify the original in-place
    processed = pred_mask.copy()
    
    # 1. Morphological closing for roads (connect broken segments)
    if road_class is not None:
        road_mask = (processed == road_class).astype(np.uint8)
        # Using an elliptical kernel which is often good for roads
        kernel_road = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        road_closed = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_road)
        
        # Apply the closed mask back to the prediction
        # Only overwrite if the original wasn't already a building (to avoid overwriting valid buildings)
        # Or just overwrite unconditionally - let's overwrite for simplicity
        processed[road_closed == 1] = road_class

    # 2. Morphological closing for buildings (fill holes)
    if building_class is not None:
        building_mask = (processed == building_class).astype(np.uint8)
        # Using a rectangular kernel which is good for buildings
        kernel_building = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        building_closed = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel_building)
        
        processed[building_closed == 1] = building_class
        
    return processed

def apply_morphology_batch(preds: torch.Tensor, road_class: int = 2, building_class: int = 1) -> torch.Tensor:
    """
    Applies morphological post-processing to a batched tensor of predictions.
    
    Args:
        preds: Tensor of shape (B, H, W) containing class indices.
        
    Returns:
        Tensor of shape (B, H, W) post-processed.
    """
    device = preds.device
    preds_np = preds.cpu().numpy()
    
    out = np.zeros_like(preds_np)
    for b in range(preds_np.shape[0]):
        out[b] = apply_morphology(preds_np[b], road_class, building_class)
        
    return torch.from_numpy(out).to(device)
