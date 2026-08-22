"""
crf.py
======
Conditional Random Field post-processing for semantic segmentation.

Uses dense CRF to refine segmentation boundaries and improve spatial consistency.
This is particularly effective for improving edge quality and reducing noise.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch
import cv2

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("Warning: pydensecrf not installed. Using fallback CRF implementation.")
    print("Install with: conda install -c conda-forge pydensecrf")


def apply_crf_fallback(
    image: Image.Image,
    softmax_probs: np.ndarray,
    num_classes: int = 8,
) -> np.ndarray:
    """
    Fallback CRF implementation using OpenCV when pydensecrf is not available.
    
    This implements a simplified version of CRF using bilateral filtering
    and morphological operations to achieve similar boundary refinement.
    
    Args:
        image: Original PIL Image (used for color features)
        softmax_probs: Softmax probabilities from model, shape (C, H, W)
        num_classes: Number of segmentation classes
        
    Returns:
        Refined segmentation mask as 2D numpy array (H, W)
    """
    # Convert image to numpy array
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]
    
    # Ensure softmax_probs has correct shape
    if len(softmax_probs.shape) == 3:
        # (C, H, W) -> (H, W, C)
        softmax_probs = softmax_probs.transpose(1, 2, 0)
    
    # Initial prediction
    pred = np.argmax(softmax_probs, axis=2).astype(np.int64)
    
    # Apply bilateral filter to each probability map for edge-preserving smoothing
    # Optimized parameters for satellite imagery: larger spatial sigma for better context
    filtered_probs = np.zeros_like(softmax_probs)
    for c in range(num_classes):
        prob_map = softmax_probs[:, :, c].astype(np.float32)
        # Apply bilateral filter with optimized parameters for satellite imagery
        # Larger d and sigmaSpace for better spatial consistency
        # Lower sigmaColor to preserve fine details
        filtered = cv2.bilateralFilter(prob_map, d=15, sigmaColor=50, sigmaSpace=100)
        filtered_probs[:, :, c] = filtered
    
    # Renormalize probabilities
    filtered_probs = filtered_probs / (np.sum(filtered_probs, axis=2, keepdims=True) + 1e-8)
    
    # Get refined prediction
    refined = np.argmax(filtered_probs, axis=2)
    
    # Apply morphological operations for boundary cleanup
    for c in range(num_classes):
        class_mask = (refined == c).astype(np.uint8)
        
        # Closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
        
        # Opening to remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # Update refined prediction
        refined[class_mask == 1] = c
        refined[opened == 1] = c
    
    return refined.astype(np.int64)


def apply_crf(
    image: Image.Image,
    softmax_probs: np.ndarray,
    num_classes: int = 8,
    n_iterations: int = 5,
    pos_w: float = 3,
    pos_xy_std: float = 3,
    bi_w: float = 5,
    bi_xy_std: float = 50,
    bi_rgb_std: float = 5,
) -> np.ndarray:
    """
    Apply Dense CRF to refine segmentation predictions.
    
    Args:
        image: Original PIL Image (used for color features)
        softmax_probs: Softmax probabilities from model, shape (C, H, W)
        num_classes: Number of segmentation classes
        n_iterations: Number of CRF iterations
        pos_w: Weight of positional Gaussian
        pos_xy_std: Standard deviation of positional Gaussian
        bi_w: Weight of bilateral Gaussian
        bi_xy_std: Standard deviation of bilateral spatial Gaussian
        bi_rgb_std: Standard deviation of bilateral color Gaussian
        
    Returns:
        Refined segmentation mask as 2D numpy array (H, W)
    """
    if not CRF_AVAILABLE:
        # Use fallback implementation
        return apply_crf_fallback(image, softmax_probs, num_classes)
    
    # Convert image to numpy array
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]
    
    # Ensure softmax_probs has correct shape
    if len(softmax_probs.shape) == 3:
        # (C, H, W) -> (H, W, C)
        softmax_probs = softmax_probs.transpose(1, 2, 0)
    
    # Create CRF model
    d = dcrf.DenseCRF2D(w, h, num_classes)
    
    # Set unary potentials from softmax
    unary = unary_from_softmax(softmax_probs)
    d.setUnaryEnergy(unary)
    
    # Add pairwise potentials
    # 1. Positional Gaussian (encourages nearby pixels to have same label)
    d.addPairwiseGaussian(sxy=(pos_xy_std, pos_xy_std), compat=pos_w, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # 2. Bilateral Gaussian (encourages nearby pixels with similar colors to have same label)
    d.addPairwiseBilateral(sxy=(bi_xy_std, bi_xy_std), srgb=(bi_rgb_std, bi_rgb_std, bi_rgb_std), 
                           rgbim=img_array, compat=bi_w, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # Run inference
    Q = d.inference(n_iterations)
    
    # Get most likely label for each pixel
    refined = np.argmax(Q, axis=0).reshape((h, w))
    
    return refined.astype(np.int64)


def apply_crf_batch(
    images: list[Image.Image],
    softmax_probs: list[np.ndarray],
    num_classes: int = 8,
    n_iterations: int = 5,
) -> list[np.ndarray]:
    """
    Apply CRF to a batch of images and predictions.
    
    Args:
        images: List of PIL Images
        softmax_probs: List of softmax probability arrays, each shape (C, H, W)
        num_classes: Number of segmentation classes
        n_iterations: Number of CRF iterations
        
    Returns:
        List of refined segmentation masks
    """
    refined_masks = []
    for img, probs in zip(images, softmax_probs):
        refined = apply_crf(img, probs, num_classes, n_iterations)
        refined_masks.append(refined)
    return refined_masks
