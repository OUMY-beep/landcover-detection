"""
advanced.py
==========
Advanced post-processing techniques for semantic segmentation without CRF.

Implements multiple refinement methods using only standard libraries:
- Superpixel-based refinement using SLIC
- Bilateral filtering for edge preservation
- Guided filtering using original image as guide
- Majority voting within superpixels
- API-based verification using SegEarth-OV model
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import sys
import os
import requests
import io
import torch.nn.functional as F

try:
    from skimage.segmentation import slic
    from skimage.restoration import denoise_bilateral
    from skimage.util import img_as_float
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not installed. Advanced post-processing will be limited.")
    print("Install with: pip install scikit-image")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Some advanced post-processing will be limited.")
    print("Install with: pip install opencv-python")

# SegEarth-OV integration
SEGEARTH_AVAILABLE = False
segearth_model = None

try:
    # Add SegEarth-OV to path - try multiple possible locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'SegEarth-OV'),  # Relative to project root
        os.path.join(os.getcwd(), 'SegEarth-OV'),  # Current working directory
        '/mnt/c/Users/oumai/OneDrive/Bureau/satellite-landcover-ai/SegEarth-OV',  # WSL path
        'SegEarth-OV',  # Direct path
    ]
    
    for segearth_path in possible_paths:
        if os.path.exists(segearth_path):
            sys.path.insert(0, segearth_path)
            try:
                from segearth_segmentor import SegEarthSegmentation
                SEGEARTH_AVAILABLE = True
                print(f"SegEarth-OV loaded successfully from: {segearth_path}")
                break
            except ImportError:
                continue
    
    if not SEGEARTH_AVAILABLE:
        print("Warning: SegEarth-OV not available in any expected location")
        print("SegEarth-OV verification will be disabled")
        
except ImportError as e:
    print(f"Warning: SegEarth-OV not available: {e}")
    print("SegEarth-OV verification will be disabled")
except Exception as e:
    print(f"Warning: SegEarth-OV initialization failed: {e}")
    print("SegEarth-OV verification will be disabled")


def apply_superpixel_refinement(
    image: Image.Image,
    pred: np.ndarray,
    num_classes: int = 8,
    n_segments: int = 1000,
    compactness: float = 10.0,
    adaptive_segments: bool = True,
) -> np.ndarray:
    """
    Refine segmentation using superpixel-based majority voting.
    
    Groups pixels into superpixels and assigns the most common class
    within each superpixel, reducing noise and improving spatial consistency.
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        num_classes: Number of segmentation classes
        n_segments: Approximate number of superpixels
        compactness: Balance between color proximity and space proximity
        adaptive_segments: Adapt number of segments based on image complexity
        
    Returns:
        Refined segmentation mask
    """
    if not SKIMAGE_AVAILABLE:
        return pred
    
    # The model prediction is produced at the model input size, while an
    # uploaded image can retain its original resolution.  Generate
    # superpixels at the prediction resolution so the boolean masks index the
    # same (H, W) grid as ``pred``.
    h, w = pred.shape
    image_for_pred = image.convert("RGB").resize((w, h), Image.BILINEAR)
    img_float = img_as_float(np.array(image_for_pred))
    
    # Adapt number of segments based on image size and complexity
    if adaptive_segments:
        # More segments for larger images to maintain detail
        base_segments = int(n_segments * (h * w) / (512 * 512))
        n_segments = max(500, min(2000, base_segments))
    
    # Generate superpixels
    segments = slic(img_float, n_segments=n_segments, compactness=compactness, 
                    start_label=0, enforce_connectivity=True)
    
    # Apply majority voting within each superpixel
    refined = pred.copy()
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        if np.sum(mask) > 0:
            # Get most common class in this superpixel
            classes_in_segment = pred[mask]
            if len(classes_in_segment) > 0:
                unique_classes, counts = np.unique(classes_in_segment, return_counts=True)
                majority_class = unique_classes[np.argmax(counts)]
                
                # Only apply majority voting if it's significantly dominant
                # This preserves small but important regions
                if counts[np.argmax(counts)] / np.sum(counts) > 0.6:
                    refined[mask] = majority_class
    
    return refined


def apply_bilateral_filter_refinement(
    image: Image.Image,
    pred: np.ndarray,
    num_classes: int = 8,
    sigma_spatial: float = 2.0,
    sigma_color: float = 0.1,
) -> np.ndarray:
    """
    Refine segmentation using bilateral filtering on probability maps.
    
    Applies bilateral filtering to each class probability map to smooth
    while preserving edges.
    
    Args:
        image: Original PIL Image (for reference)
        pred: Prediction mask, shape (H, W)
        num_classes: Number of segmentation classes
        sigma_spatial: Spatial standard deviation
        sigma_color: Color standard deviation
        
    Returns:
        Refined segmentation mask
    """
    if not SKIMAGE_AVAILABLE:
        return pred
    
    # Convert prediction to one-hot probability maps
    h, w = pred.shape
    prob_maps = np.zeros((num_classes, h, w), dtype=np.float32)
    for c in range(num_classes):
        prob_maps[c] = (pred == c).astype(np.float32)
    
    # Apply bilateral filter to each probability map
    filtered_probs = np.zeros_like(prob_maps)
    for c in range(num_classes):
        # Bilateral filter on each channel
        filtered_probs[c] = denoise_bilateral(
            prob_maps[c], 
            sigma_color=sigma_color, 
            sigma_spatial=sigma_spatial,
            mode='constant',
            channel_axis=None
        )
    
    # Convert back to class labels
    refined = np.argmax(filtered_probs, axis=0)
    return refined.astype(np.int64)


# Alias for compatibility with inference.py
apply_bilateral_filter = apply_bilateral_filter_refinement


def apply_confidence_filtering(
    probs: np.ndarray,
    confidence_threshold: float = 0.7,
    fallback_class: int = 1,  # Default to background
    scene_complexity: float = 0.5,
) -> np.ndarray:
    """
    Filter predictions based on model confidence with adaptive thresholding.
    
    Pixels with low confidence (max probability below threshold) are
    assigned to a fallback class (typically background).
    
    Args:
        probs: Softmax probabilities, shape (C, H, W)
        confidence_threshold: Minimum confidence to keep prediction
        fallback_class: Class to assign to low-confidence pixels
        scene_complexity: Scene complexity score (0-1) for adaptive thresholding
        
    Returns:
        Refined segmentation mask
    """
    if len(probs.shape) == 3:
        # (C, H, W) -> (H, W, C)
        probs = probs.transpose(1, 2, 0)
    
    h, w, num_classes = probs.shape
    
    # Adaptive threshold: more permissive for complex scenes
    adaptive_threshold = confidence_threshold * (1.0 - 0.3 * scene_complexity)
    
    # Get maximum probability and corresponding class for each pixel
    max_probs = np.max(probs, axis=2)
    pred_classes = np.argmax(probs, axis=2)
    
    # Apply confidence threshold
    low_confidence = max_probs < adaptive_threshold
    pred_classes[low_confidence] = fallback_class
    
    return pred_classes.astype(np.int64)


def apply_edge_aware_refinement(
    image: Image.Image,
    pred: np.ndarray,
    num_classes: int = 8,
    edge_threshold: float = 0.1,
) -> np.ndarray:
    """
    Refine segmentation using edge information from the original image.
    
    Detects edges in the original image and ensures segmentation boundaries
    align with detected edges.
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        num_classes: Number of segmentation classes
        edge_threshold: Threshold for edge detection
        
    Returns:
        Refined segmentation mask
    """
    if not CV2_AVAILABLE:
        return pred
    
    h, w = pred.shape
    image_resized = image.convert("RGB").resize((w, h), Image.BILINEAR)
    img_gray = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2GRAY)
    
    # Detect edges using Canny
    edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)
    
    # Dilate edges to create a boundary region
    kernel = np.ones((3, 3), np.uint8)
    edge_region = cv2.dilate(edges, kernel, iterations=1)
    
    # For pixels near edges, apply additional smoothing
    refined = pred.copy()
    
    # Simple edge-aware smoothing: if a pixel is near an edge, 
    # make it match the majority of its non-edge neighbors
    edge_pixels = edge_region > 0
    
    if np.any(edge_pixels):
        # Get coordinates of edge pixels
        edge_coords = np.argwhere(edge_pixels)
        
        for y, x in edge_coords:
            # Get neighborhood
            y_min, y_max = max(0, y-2), min(h, y+3)
            x_min, x_max = max(0, x-2), min(w, x+3)
            
            neighborhood = pred[y_min:y_max, x_min:x_max]
            neighborhood_edges = edge_region[y_min:y_max, x_min:x_max]
            
            # Consider only non-edge neighbors
            non_edge_neighbors = neighborhood[neighborhood_edges == 0]
            
            if len(non_edge_neighbors) > 0:
                # Use majority class from non-edge neighbors
                unique_classes, counts = np.unique(non_edge_neighbors, return_counts=True)
                if len(unique_classes) > 0:
                    majority_class = unique_classes[np.argmax(counts)]
                    refined[y, x] = majority_class
    
    return refined


def apply_iterative_refinement(
    image: Image.Image,
    pred: np.ndarray,
    probs: np.ndarray,
    num_classes: int = 8,
    max_iterations: int = 3,
) -> np.ndarray:
    """
    Apply iterative refinement to gradually improve segmentation.
    
    Each iteration applies different refinement techniques and uses the
    result as input for the next iteration, converging to a better solution.
    
    Args:
        image: Original PIL Image
        pred: Initial prediction mask, shape (H, W)
        probs: Softmax probabilities, shape (C, H, W)
        num_classes: Number of segmentation classes
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        Refined segmentation mask
    """
    refined = pred.copy()
    
    for iteration in range(max_iterations):
        # Apply different refinement in each iteration
        if iteration == 0:
            # First iteration: class-specific refinement
            refined = apply_class_specific_refinement(image, refined, probs, num_classes)
        elif iteration == 1:
            # Second iteration: superpixel refinement
            if SKIMAGE_AVAILABLE:
                refined = apply_superpixel_refinement(image, refined, num_classes=num_classes, 
                                                    n_segments=1500, adaptive_segments=True)
        else:
            # Later iterations: edge-aware and bilateral filtering
            if CV2_AVAILABLE:
                refined = apply_edge_aware_refinement(image, refined, num_classes=num_classes)
                refined = apply_bilateral_filter(image, refined, num_classes=num_classes)
    
    return refined


def apply_class_specific_thresholds(
    probs: np.ndarray,
    class_thresholds: dict[int, float] = None,
    fallback_class: int = 1,
) -> np.ndarray:
    """
    Apply class-specific decision thresholds instead of global argmax.
    
    Instead of using argmax for all pixels, each class has its own
    confidence threshold. This is particularly useful when certain classes
    are systematically under-detected or confused with others.
    
    Args:
        probs: Softmax probabilities, shape (C, H, W)
        class_thresholds: Dictionary mapping class_id to threshold (0-1)
                      If None, uses default calibrated thresholds
        fallback_class: Class to assign when no class meets its threshold
        
    Returns:
        Refined segmentation mask
    """
    if len(probs.shape) == 3:
        # (C, H, W) -> (H, W, C)
        probs = probs.transpose(1, 2, 0)
    
    h, w, num_classes = probs.shape
    total_pixels = h * w
    
    # Default calibrated thresholds based on common confusions
    # Building (2) vs Background (1): Building needs VERY low threshold for dense areas
    # Road (3) vs Background (1): Road needs lower threshold
    # Water (4) vs Forest (6): Water needs EXTREMELY high threshold to avoid confusion
    # Forest (6) vs Water (4): Forest needs EXTREMELY low threshold to avoid being classified as water
    # Class imbalance handling: adjust thresholds based on class frequency in prediction
    if class_thresholds is None:
        class_thresholds = {
            0: 0.9,   # Ignore - very high threshold
            1: 0.55,  # Background - higher threshold (often dominant class)
            2: 0.12,  # Building - VERY low threshold (critical for dense building areas)
            3: 0.30,  # Road - very low threshold (connectivity important, often fragmented)
            4: 0.85,  # Water - EXTREMELY high threshold (to avoid confusion with forest)
            5: 0.45,  # Barren - medium threshold
            6: 0.15,  # Forest - EXTREMELY low threshold (to avoid being confused with water)
            7: 0.40,  # Agricultural - medium-low threshold
        }
    
    # Adaptive threshold adjustment based on class frequency
    # If a class is very rare in the prediction, lower its threshold to help detection
    # Calculate class frequencies
    class_frequencies = {}
    for c in range(num_classes):
        # probs is (C, H, W), so extract class channel correctly
        class_pixels = np.sum(probs[c, :, :] > 0.1)  # Count pixels with >10% confidence
        class_frequencies[c] = class_pixels / total_pixels
    
    # Adjust thresholds for rare classes
    adjusted_thresholds = class_thresholds.copy()
    for c in range(num_classes):
        freq = class_frequencies.get(c, 0)
        if freq < 0.01:  # Very rare class (<1% of pixels)
            # Lower threshold by 20% for rare classes
            adjusted_thresholds[c] = max(0.1, class_thresholds[c] * 0.8)
        elif freq < 0.05:  # Rare class (<5% of pixels)
            # Lower threshold by 10% for rare classes
            adjusted_thresholds[c] = max(0.15, class_thresholds[c] * 0.9)
    
    class_thresholds = adjusted_thresholds
    
    refined = np.zeros((h, w), dtype=np.int64)
    
    # For each pixel, find the class that meets its threshold with highest confidence
    for c in range(num_classes):
        threshold = class_thresholds.get(c, 0.5)
        class_mask = probs[:, :, c] >= threshold
        
        # For pixels where this class meets threshold, assign if it's the best so far
        class_probs = probs[:, :, c]
        
        # Initialize with -infinity for comparison
        if c == 0:
            best_probs = np.full((h, w), -np.inf)
        else:
            best_probs = refined_probs if 'refined_probs' in locals() else np.full((h, w), -np.inf)
        
        # Update where this class is better
        better_mask = (class_probs > best_probs) & class_mask
        refined[better_mask] = c
        
        if 'refined_probs' not in locals():
            refined_probs = class_probs.copy()
        else:
            refined_probs[better_mask] = class_probs[better_mask]
    
    # Assign fallback class to pixels that didn't meet any threshold
    no_class = (refined == 0)
    refined[no_class] = fallback_class
    
    return refined


def process_image_with_overlap_stitching(
    image: Image.Image,
    model_predict_fn,
    patch_size: int = 512,
    overlap_ratio: float = 0.25,
    num_classes: int = 8,
) -> np.ndarray:
    """
    Process large images using overlapping patches to avoid edge artifacts.
    
    Instead of brutal stitching, this method:
    1. Divides image into overlapping patches
    2. Processes each patch independently
    3. Blends overlapping regions using weighted averaging
    4. Eliminates visible tile boundaries
    
    Args:
        image: PIL Image to process
        model_predict_fn: Function that takes PIL Image and returns (H, W) prediction
        patch_size: Size of each patch (default 512)
        overlap_ratio: Overlap between patches (0.25 = 25%)
        num_classes: Number of segmentation classes
        
    Returns:
        Full-resolution segmentation mask
    """
    w, h = image.size
    
    # Calculate patch parameters
    overlap = int(patch_size * overlap_ratio)
    stride = patch_size - overlap
    
    # Initialize output and weight arrays
    full_prediction = np.zeros((h, w), dtype=np.float64)
    full_weights = np.zeros((h, w), dtype=np.float64)
    
    # Calculate number of patches
    num_patches_x = (w - overlap) // stride + 1
    num_patches_y = (h - overlap) // stride + 1
    
    # Process each patch
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            # Calculate patch boundaries
            y_start = y * stride
            y_end = min(y_start + patch_size, h)
            x_start = x * stride
            x_end = min(x_start + patch_size, w)
            
            # Extract patch
            patch = image.crop((x_start, y_start, x_end, y_end))
            
            # Get prediction for this patch
            try:
                patch_pred = model_predict_fn(patch)
                
                # Resize prediction to patch size if needed
                if patch_pred.shape != (y_end - y_start, x_end - x_start):
                    from PIL import Image as PILImage
                    pred_pil = PILImage.fromarray(patch_pred.astype(np.uint8))
                    patch_pred = np.array(pred_pil.resize((x_end - x_start, y_end - y_start), Image.NEAREST))
                
                # Create weight mask for blending (linear feathering)
                weight_mask = np.ones((y_end - y_start, x_end - x_start), dtype=np.float64)
                
                # Apply feathering at edges
                feather_size = overlap // 2
                if feather_size > 0:
                    # Top edge feathering
                    if y_start > 0:
                        for i in range(feather_size):
                            weight = i / feather_size
                            weight_mask[i, :] *= weight
                    
                    # Bottom edge feathering
                    if y_end < h:
                        for i in range(feather_size):
                            weight = (feather_size - i) / feather_size
                            weight_mask[-(i+1), :] *= weight
                    
                    # Left edge feathering
                    if x_start > 0:
                        for i in range(feather_size):
                            weight = i / feather_size
                            weight_mask[:, i] *= weight
                    
                    # Right edge feathering
                    if x_end < w:
                        for i in range(feather_size):
                            weight = (feather_size - i) / feather_size
                            weight_mask[:, -(i+1)] *= weight
                
                # Add to full prediction with weights
                full_prediction[y_start:y_end, x_start:x_end] += patch_pred * weight_mask
                full_weights[y_start:y_end, x_start:x_end] += weight_mask
                
            except Exception as e:
                print(f"Error processing patch ({x}, {y}): {e}")
                continue
    
    # Normalize by weights
    full_weights[full_weights == 0] = 1  # Avoid division by zero
    full_prediction = full_prediction / full_weights
    
    return full_prediction.astype(np.int64)


def apply_texture_aware_processing(
    image: Image.Image,
    pred: np.ndarray,
    num_classes: int = 8,
) -> np.ndarray:
    """
    Apply texture-aware processing to improve segmentation in textured regions.
    
    Uses texture analysis to guide refinement, particularly useful for:
    - Forest regions (varying canopy texture)
    - Agricultural fields (crop patterns)
    - Urban areas (building textures)
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        num_classes: Number of segmentation classes
        
    Returns:
        Refined segmentation mask
    """
    if not CV2_AVAILABLE:
        return pred
    
    h, w = pred.shape
    refined = pred.copy()
    
    # Convert image to grayscale for texture analysis
    image_resized = image.convert("RGB").resize((w, h), Image.BILINEAR)
    img_gray = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2GRAY)
    
    # Calculate Local Binary Pattern for texture analysis
    from skimage.feature import local_binary_pattern
    if SKIMAGE_AVAILABLE:
        # LBP parameters
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        
        # Calculate texture variance for each region
        for c in range(1, num_classes):  # Skip ignore class
            class_mask = (pred == c)
            if np.sum(class_mask) > 0:
                # Get LBP values for this class
                class_lbp = lbp[class_mask]
                
                # Calculate texture variance
                texture_variance = np.std(class_lbp)
                
                # If texture variance is high, apply additional smoothing
                if texture_variance > 5:  # High texture variation
                    # Apply stronger smoothing for textured regions
                    kernel_size = 5
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    class_mask_uint8 = class_mask.astype(np.uint8)
                    smoothed = cv2.morphologyEx(class_mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
                    refined[smoothed == 1] = c
    
    return refined


def calculate_scene_complexity(
    image: Image.Image,
    pred: np.ndarray,
    num_classes: int = 8,
) -> float:
    """
    Calculate scene complexity score based on class distribution and edges.
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        num_classes: Number of segmentation classes
        
    Returns:
        Complexity score (0-1, higher = more complex)
    """
    h, w = pred.shape
    
    # Count number of unique classes present
    unique_classes = len(np.unique(pred[pred != 0]))  # Exclude ignore class
    class_diversity = unique_classes / num_classes
    
    # Calculate class entropy (distribution uniformity)
    class_counts = []
    for c in range(1, num_classes):  # Skip ignore class
        count = np.sum(pred == c)
        if count > 0:
            class_counts.append(count)
    
    if len(class_counts) > 0:
        total_pixels = sum(class_counts)
        proportions = [c / total_pixels for c in class_counts]
        entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
        max_entropy = np.log2(len(class_counts))
        distribution_uniformity = entropy / max_entropy if max_entropy > 0 else 0
    else:
        distribution_uniformity = 0
    
    # Calculate edge density in the image
    if CV2_AVAILABLE:
        image_resized = image.convert("RGB").resize((w, h), Image.BILINEAR)
        img_gray = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)
        edge_density = np.sum(edges > 0) / (h * w)
    else:
        edge_density = 0.5  # Default value if OpenCV not available
    
    # Calculate boundary density in prediction
    boundaries = np.zeros_like(pred, dtype=np.uint8)
    for c in range(1, num_classes):
        class_mask = (pred == c).astype(np.uint8)
        class_boundaries = cv2.Canny(class_mask, 1, 1) if CV2_AVAILABLE else np.zeros_like(class_mask)
        boundaries = np.maximum(boundaries, class_boundaries)
    
    boundary_density = np.sum(boundaries > 0) / (h * w) if CV2_AVAILABLE else 0.5
    
    # Combined complexity score
    complexity = (
        0.3 * class_diversity +
        0.3 * distribution_uniformity +
        0.2 * edge_density +
        0.2 * boundary_density
    )
    
    return min(1.0, max(0.0, complexity))


def apply_class_specific_refinement(
    image: Image.Image,
    pred: np.ndarray,
    probs: np.ndarray,
    num_classes: int = 8,
) -> np.ndarray:
    """
    Apply class-specific refinement for complex scenes.
    
    Different classes require different processing strategies:
    - Buildings: Need sharp boundaries, preserve small structures
    - Roads: Need connectivity, remove gaps
    - Water: Need smooth boundaries, handle reflections
    - Forest: Need texture consistency, handle canopy variations
    - Agricultural: Need field uniformity, handle crop patterns
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        probs: Softmax probabilities, shape (C, H, W)
        num_classes: Number of segmentation classes
        
    Returns:
        Refined segmentation mask
    """
    if not CV2_AVAILABLE:
        return pred
    
    # Small object detection and recovery
    # This helps recover small buildings, roads, and other important structures
    h, w = pred.shape
    refined = pred.copy()
    
    # Keep probs in (C, H, W) format throughout
    # No transposing needed - we'll handle indexing correctly
    
    # Define minimum sizes for each class (in pixels)
    class_min_sizes = {
        0: 10,    # Ignore
        1: 100,   # Background
        2: 25,    # Building - small buildings are important
        3: 50,    # Road - small road segments should be preserved
        4: 200,   # Water - small water regions likely noise
        5: 50,    # Barren
        6: 100,   # Forest
        7: 75,    # Agricultural
    }
    
    # For each class, recover small objects that have high confidence
    for class_id in range(num_classes):
        if class_id == 0:  # Skip ignore class
            continue
            
        min_size = class_min_sizes.get(class_id, 50)
        
        # Find pixels predicted as this class
        class_mask = (pred == class_id).astype(np.uint8)
        
        if np.sum(class_mask) == 0:
            continue
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            
            # If region is too small, check if it has high confidence
            if area < min_size:
                region_mask = (labels == label)
                
                # Get average confidence for this region
                # probs is (C, H, W), extract class channel then apply mask
                class_probs = probs[class_id, :, :]
                region_probs = class_probs[region_mask]
                avg_confidence = np.mean(region_probs) if len(region_probs) > 0 else 0
                
                # If confidence is high enough, keep it
                # Lower threshold for important classes (buildings, roads)
                confidence_threshold = 0.4 if class_id in [2, 3] else 0.5
                
                if avg_confidence < confidence_threshold:
                    # Remove low-confidence small regions
                    refined[region_mask] = 1  # Convert to background
    
    # Building-specific refinement (class 2)
    building_class = 2
    background_class = 1
    building_mask = (pred == building_class).astype(np.uint8)
    if np.sum(building_mask) > 0:
        # Morphological closing to fill small gaps in buildings
        kernel_building = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        building_closed = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel_building, iterations=2)
        # Opening to remove small noise
        building_opened = cv2.morphologyEx(building_closed, cv2.MORPH_OPEN, kernel_building, iterations=1)
        refined[building_opened == 1] = building_class
    
    # CRITICAL: Fix dense building areas misclassified as background
    # Use color and edge information to recover buildings
    if CV2_AVAILABLE:
        background_mask = (pred == background_class).astype(np.uint8)
        if np.sum(background_mask) > 0:
            # Get image resized to prediction size
            h, w = pred.shape
            image_resized = image.convert("RGB").resize((w, h), Image.BILINEAR)
            img_array = np.array(image_resized)
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect edges using Canny - buildings have strong edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to capture building regions
            kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges_dilated = cv2.dilate(edges, kernel_edge, iterations=2)
            
            # Calculate edge density in background regions
            background_edges = edges_dilated & background_mask
            
            # Find background regions with high edge density (likely buildings)
            # Use sliding window to detect dense building areas
            window_size = 32
            edge_density_map = np.zeros((h, w), dtype=np.float32)
            
            for i in range(0, h - window_size, window_size // 2):
                for j in range(0, w - window_size, window_size // 2):
                    window = background_edges[i:i+window_size, j:j+window_size]
                    if np.sum(window) > 0:
                        density = np.sum(window) / (window_size * window_size)
                        edge_density_map[i:i+window_size, j:j+window_size] = density
            
            # Threshold edge density to find building candidates
            building_candidates = (edge_density_map > 0.15) & background_mask
            
            # Also use color information - buildings are typically gray/white
            # Calculate saturation - buildings have low saturation
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1].astype(np.float32) / 255.0
            low_saturation = (saturation < 0.3) & background_mask
            
            # Combine edge density and low saturation for building detection
            likely_buildings = building_candidates & low_saturation
            
            # Convert likely buildings to building class
            refined[likely_buildings] = building_class
    
    # Road-specific refinement (class 3)
    road_class = 3
    road_mask = (pred == road_class).astype(np.uint8)
    if np.sum(road_mask) > 0:
        # First, fill small gaps in roads using morphological closing
        kernel_road_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        road_closed = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_road_close, iterations=2)
        
        # Then use skeletonization to ensure connectivity
        try:
            road_skeleton = cv2.ximgproc.thinning(road_closed)
        except:
            # Fallback if ximgproc not available
            road_skeleton = road_closed
        
        # Dilate skeleton back to road width with larger kernel
        kernel_road = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        road_dilated = cv2.dilate(road_skeleton, kernel_road, iterations=2)
        
        # Apply additional morphological operations to smooth road boundaries
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        road_smoothed = cv2.morphologyEx(road_dilated, cv2.MORPH_OPEN, kernel_smooth, iterations=1)
        
        refined[road_smoothed == 1] = road_class
    
    # Water-specific refinement (class 4)
    water_class = 4
    water_mask = (pred == water_class).astype(np.uint8)
    if np.sum(water_mask) > 0:
        # Smooth water boundaries
        kernel_water = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        water_smoothed = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel_water, iterations=1)
        # Remove small water islands (likely forest misclassified as water)
        # EXTREMELY aggressive threshold - remove almost all small water regions
        num_labels, labels, stats = cv2.connectedComponentsWithStats(water_smoothed, connectivity=8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < 2000:  # VERY high threshold to remove small regions
                water_smoothed[labels == label] = 0
        refined[water_smoothed == 1] = water_class
    
    # Forest-specific refinement (class 6)
    forest_class = 6
    forest_mask = (pred == forest_class).astype(np.uint8)
    if np.sum(forest_mask) > 0:
        # Ensure forest regions are reasonably sized
        kernel_forest = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        forest_processed = cv2.morphologyEx(forest_mask, cv2.MORPH_CLOSE, kernel_forest, iterations=1)
        refined[forest_processed == 1] = forest_class
    
    # Additional: Fix forest vs water confusion using advanced color and texture analysis
    # Forests are typically green (higher G channel), water is blue (higher B channel)
    # Color gradients can cause confusion - need more sophisticated analysis
    if CV2_AVAILABLE:
        water_class = 4
        forest_class = 6
        
        # Get image resized to prediction size
        h, w = pred.shape
        image_resized = image.convert("RGB").resize((w, h), Image.BILINEAR)
        img_array = np.array(image_resized)
        
        # CRITICAL: Direct probability comparison - favor forest over water
        # If forest and water probabilities are close, choose forest
        if len(probs.shape) == 3:
            # probs is (C, H, W), need to extract forest and water channels
            forest_probs = probs[forest_class, :, :]
            water_probs = probs[water_class, :, :]
        else:
            # probs is (H, W, C)
            forest_probs = probs[:, :, forest_class]
            water_probs = probs[:, :, water_class]
        
        # If forest probability is within 0.1 of water probability, prefer forest
        # This handles cases where the model is uncertain between forest and water
        close_probs = (water_probs > 0.1) & (forest_probs >= water_probs - 0.1)
        refined[close_probs] = forest_class
        
        # Find pixels classified as water
        water_pixels = (pred == water_class)
        if np.sum(water_pixels) > 0:
            # Convert to multiple color spaces for robust analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # 1. RGB Green-Blue ratio analysis
            # Calculate G/B ratio for all pixels
            gb_ratio = img_array[:, :, 1].astype(np.float32) / (img_array[:, :, 2].astype(np.float32) + 1e-6)
            # Water pixels with high G/B ratio are likely forest
            forest_misclassified_rgb = water_pixels & (gb_ratio > 0.85)
            refined[forest_misclassified_rgb] = forest_class
            
            # 2. HSV Hue analysis (more robust to color gradients)
            hue = hsv[:, :, 0].astype(np.float32)
            # Forest: green range (25-95), Water: blue range (95-145)
            # Extended ranges to handle color gradients
            forest_hue_mask = (hue >= 25) & (hue <= 95)
            water_hue_mask = (hue >= 95) & (hue <= 145)
            
            # Water pixels with forest-like hue are likely misclassified
            water_hue_check = water_pixels & forest_hue_mask
            refined[water_hue_check] = forest_class
            
            # 3. HSV Saturation analysis
            # Water typically has higher saturation, forest has lower saturation
            saturation = hsv[:, :, 1].astype(np.float32) / 255.0
            # Lower threshold to catch more forest-like pixels
            low_saturation = saturation < 0.30
            water_sat_check = water_pixels & low_saturation
            refined[water_sat_check] = forest_class
            
            # 4. LAB Color Space analysis (better for perceptual color differences)
            # L: lightness, A: green-red, B: blue-yellow
            # Forest: higher A (green), Water: lower A (blue)
            a_channel = lab[:, :, 1].astype(np.float32)
            
            # Forest has positive A (green), water has negative A (blue-ish)
            forest_lab_mask = a_channel > 5
            water_lab_check = water_pixels & forest_lab_mask
            refined[water_lab_check] = forest_class
            
            # 5. Texture analysis using local variance
            # Forest has more texture (higher local variance) than water (smoother)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate local variance using a small kernel
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
            variance = sqr_mean - mean**2
            
            # Forest has higher variance, water has lower variance
            # Use adaptive threshold based on local context
            high_variance = variance > 15  # Pixels with high texture
            water_texture_check = water_pixels & high_variance
            refined[water_texture_check] = forest_class
            
            # 6. Gradient analysis for smooth regions (water is typically smoother)
            # Calculate gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Water has lower gradient magnitude (smoother)
            # Forest has higher gradient magnitude (more edges from trees)
            high_gradient = gradient_magnitude > 12
            water_gradient_check = water_pixels & high_gradient
            refined[water_gradient_check] = forest_class
    
    # Agricultural-specific refinement (class 7)
    ag_class = 7
    ag_mask = (pred == ag_class).astype(np.uint8)
    if np.sum(ag_mask) > 0:
        # Agricultural fields should be relatively uniform
        kernel_ag = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        ag_processed = cv2.morphologyEx(ag_mask, cv2.MORPH_CLOSE, kernel_ag, iterations=2)
        refined[ag_processed == 1] = ag_class
    
    # Boundary refinement using edge information
    # This helps align segmentation boundaries with actual image edges
    if CV2_AVAILABLE:
        # Get image resized to prediction size
        h, w = pred.shape
        image_resized = image.convert("RGB").resize((w, h), Image.BILINEAR)
        img_array = np.array(image_resized)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create boundary regions
        kernel_boundary = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_boundary, iterations=1)
        
        # For each class, refine boundaries using edge information
        for class_id in range(1, num_classes):  # Skip ignore class
            class_mask = (refined == class_id).astype(np.uint8)
            
            if np.sum(class_mask) == 0:
                continue
            
            # Find class boundaries
            class_dilated = cv2.dilate(class_mask, kernel_boundary, iterations=1)
            boundaries = class_dilated & ~class_mask
            
            # Check if boundaries align with image edges
            boundary_edges = boundaries & edges_dilated
            
            # If boundaries don't align with edges, smooth them
            if np.sum(boundary_edges) < np.sum(boundaries) * 0.5:
                # Apply morphological smoothing to boundaries
                kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                class_smoothed = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
                refined[class_smoothed == 1] = class_id
    
    return refined


def apply_background_cleanup(
    image: Image.Image,
    pred: np.ndarray,
    background_class: int = 1,
    min_region_size: int = 100,
) -> np.ndarray:
    """
    Specialized cleanup for background class to handle segmentation issues.
    
    This function specifically targets background-related problems:
    1. Removes small isolated non-background regions
    2. Expands background regions in homogeneous areas
    3. Uses color information to guide background assignment
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        background_class: Class ID for background
        min_region_size: Minimum size for non-background regions
        
    Returns:
        Refined segmentation mask with improved background handling
    """
    if not CV2_AVAILABLE:
        return pred
    
    h, w = pred.shape
    refined = pred.copy()
    
    # Get image for color analysis
    image_resized = image.convert("RGB").resize((w, h), Image.BILINEAR)
    img_array = np.array(image_resized)
    
    # Find all non-background regions
    non_background_mask = (pred != background_class) & (pred != 0)  # Exclude ignore class
    
    # Find connected components of non-background regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        non_background_mask.astype(np.uint8), connectivity=8
    )
    
    # Process each connected component
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        
        # Remove very small non-background regions
        if area < min_region_size:
            # Get the region's color characteristics
            region_mask = labels == label
            region_pixels = img_array[region_mask]
            
            if len(region_pixels) > 0:
                # Calculate color variance
                color_std = np.std(region_pixels, axis=0)
                is_homogeneous = np.mean(color_std) < 15  # Low variance threshold
                
                # If region is small and homogeneous, it's likely noise
                if is_homogeneous:
                    refined[region_mask] = background_class
    
    # Expand background in homogeneous areas using morphological dilation
    background_mask = refined == background_class
    
    # Create a larger kernel for background expansion
    kernel_large = np.ones((5, 5), np.uint8)
    dilated_background = cv2.dilate(background_mask.astype(np.uint8), kernel_large, iterations=1)
    
    # Find regions where background expanded but doesn't conflict with strong edges
    expansion_mask = dilated_background & ~background_mask
    
    if np.any(expansion_mask):
        # Check color homogeneity in expansion regions
        expansion_coords = np.argwhere(expansion_mask)
        
        for y, x in expansion_coords:
            # Get neighborhood
            y_min, y_max = max(0, y-3), min(h, y+4)
            x_min, x_max = max(0, x-3), min(w, x+4)
            
            neighborhood = img_array[y_min:y_max, x_min:x_max]
            neighborhood_pred = refined[y_min:y_max, x_min:x_max]
            
            # If neighborhood is mostly background and homogeneous
            background_neighbors = neighborhood_pred == background_class
            if np.mean(background_neighbors) > 0.7:
                color_std = np.std(neighborhood, axis=(0, 1))
                if np.mean(color_std) < 20:  # Homogeneous region
                    refined[y, x] = background_class
    
    return refined


def apply_guided_filter_refinement(
    image: Image.Image,
    pred: np.ndarray,
    num_classes: int = 8,
    radius: int = 8,
    eps: float = 1e-2,
) -> np.ndarray:
    """
    Refine segmentation using guided filtering with original image as guide.
    
    Uses the original image as a guide to filter the prediction,
    preserving edges that align with image boundaries.
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        num_classes: Number of segmentation classes
        radius: Radius of the guided filter
        eps: Regularization parameter
        
    Returns:
        Refined segmentation mask
    """
    if not CV2_AVAILABLE:
        return pred
    
    # Keep the guide aligned with the prediction probability maps.
    h, w = pred.shape
    image_for_pred = image.convert("RGB").resize((w, h), Image.BILINEAR)
    guide = np.array(image_for_pred)
    guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Convert prediction to one-hot probability maps
    prob_maps = np.zeros((num_classes, h, w), dtype=np.float32)
    for c in range(num_classes):
        prob_maps[c] = (pred == c).astype(np.float32)
    
    # Apply guided filter to each probability map
    filtered_probs = np.zeros_like(prob_maps)
    for c in range(num_classes):
        filtered_probs[c] = cv2.ximgproc.guidedFilter(
            guide=guide,
            src=prob_maps[c],
            radius=radius,
            eps=eps
        )
    
    # Normalize and take argmax
    filtered_probs = filtered_probs / (np.sum(filtered_probs, axis=0, keepdims=True) + 1e-8)
    refined = np.argmax(filtered_probs, axis=0)
    
    return refined.astype(np.int64)


def apply_combined_refinement(
    image: Image.Image,
    pred: np.ndarray,
    num_classes: int = 8,
    use_superpixel: bool = True,
    use_bilateral: bool = True,
    use_guided: bool = False,
    use_confidence: bool = True,
    use_spatial: bool = True,
    use_api_verification: bool = False,
    use_background_verification: bool = False,
) -> np.ndarray:
    """
    Apply conservative cleanup and optional SegEarth-OV verification.
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        num_classes: Number of segmentation classes
        use_superpixel: Whether to apply superpixel refinement
        use_bilateral: Whether to apply bilateral filtering
        use_guided: Whether to apply guided filtering
        use_confidence: Whether to apply confidence-based filtering
        use_spatial: Whether to apply spatial consistency refinement
        use_api_verification: Whether to use Hugging Face API for verification
        use_background_verification: Whether to verify background specifically
        
    Returns:
        Refined segmentation mask
    """
    del num_classes, use_superpixel, use_bilateral, use_guided, use_confidence, use_spatial

    refined = pred.copy()
    if use_api_verification or use_background_verification:
        if use_api_verification:
            refined = verify_segmentation_with_segearth(image, refined)
        if use_background_verification:
            refined = verify_background_with_segearth(image, refined)

    from postprocessing.morphology import apply_morphology
    return apply_morphology(refined, road_class=3, building_class=2)


def apply_edge_preserving_smooth(
    image: Image.Image,
    pred: np.ndarray,
    num_classes: int = 8,
    iterations: int = 3,
) -> np.ndarray:
    """
    Apply edge-preserving smoothing using simple spatial consistency.
    
    For each pixel, consider its neighbors and if most neighbors agree
    on a class, change the pixel to that class (unless it's on a strong edge).
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        num_classes: Number of segmentation classes
        iterations: Number of smoothing iterations
        
    Returns:
        Refined segmentation mask
    """
    refined = pred.copy()
    h, w = pred.shape
    
    # Simple edge detection from image
    img_gray = np.array(image.convert("L"))
    # Sobel edge detection
    sobel_x = np.gradient(img_gray.astype(np.float32), axis=1)
    sobel_y = np.gradient(img_gray.astype(np.float32), axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_threshold = np.percentile(edges, 80)  # Top 20% as edges
    
    for _ in range(iterations):
        new_refined = refined.copy()
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Skip if on strong edge
                if edges[i, j] > edge_threshold:
                    continue
                
                # Get neighborhood
                neighborhood = refined[i-1:i+2, j-1:j+2].flatten()
                
                # Majority vote
                unique_classes, counts = np.unique(neighborhood, return_counts=True)
                majority_class = unique_classes[np.argmax(counts)]
                new_refined[i, j] = majority_class
        
        refined = new_refined
    
    return refined


def apply_confidence_filtering(
    image: Image.Image,
    pred: np.ndarray,
    num_classes: int = 8,
    confidence_threshold: float = 0.7,
) -> np.ndarray:
    """
    Apply confidence-based filtering to uncertain predictions.
    
    Uses the original image to identify uncertain regions and applies
    spatial smoothing to those regions only.
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        num_classes: Number of segmentation classes
        confidence_threshold: Threshold for confidence filtering
        
    Returns:
        Refined segmentation mask
    """
    refined = pred.copy()
    h, w = pred.shape
    
    # Calculate local entropy as a measure of uncertainty
    from scipy import ndimage as ndi
    
    # Create a confidence map based on local class consistency
    confidence = np.zeros((h, w), dtype=np.float32)
    for i in range(1, h-1):
        for j in range(1, w-1):
            neighborhood = pred[i-1:i+2, j-1:j+2].flatten()
            unique_classes, counts = np.unique(neighborhood, return_counts=True)
            # Confidence = proportion of most common class
            confidence[i, j] = np.max(counts) / len(neighborhood)
    
    # Smooth the confidence map
    confidence = ndi.gaussian_filter(confidence, sigma=1)
    
    # Apply smoothing to low-confidence regions
    for i in range(1, h-1):
        for j in range(1, w-1):
            if confidence[i, j] < confidence_threshold:
                # Apply majority voting in this region
                neighborhood = pred[i-1:i+2, j-1:j+2].flatten()
                unique_classes, counts = np.unique(neighborhood, return_counts=True)
                majority_class = unique_classes[np.argmax(counts)]
                refined[i, j] = majority_class
    
    return refined


def apply_spatial_consistency(
    image: Image.Image,
    pred: np.ndarray,
    num_classes: int = 8,
    window_size: int = 5,
) -> np.ndarray:
    """
    Apply spatial consistency refinement using larger neighborhoods.
    
    Uses a larger sliding window to ensure spatial consistency across
    larger regions, particularly useful for homogeneous areas like roads.
    
    Args:
        image: Original PIL Image
        pred: Prediction mask, shape (H, W)
        num_classes: Number of segmentation classes
        window_size: Size of the sliding window (odd number)
        
    Returns:
        Refined segmentation mask
    """
    refined = pred.copy()
    h, w = pred.shape
    half_window = window_size // 2
    
    # Edge detection to avoid smoothing across boundaries
    img_gray = np.array(image.convert("L"))
    sobel_x = np.gradient(img_gray.astype(np.float32), axis=1)
    sobel_y = np.gradient(img_gray.astype(np.float32), axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_threshold = np.percentile(edges, 70)
    
    for i in range(half_window, h - half_window):
        for j in range(half_window, w - half_window):
            # Skip if on strong edge
            if edges[i, j] > edge_threshold:
                continue
            
            # Get larger neighborhood
            neighborhood = pred[i-half_window:i+half_window+1, 
                            j-half_window:j+half_window+1].flatten()
            
            # Majority vote
            unique_classes, counts = np.unique(neighborhood, return_counts=True)
            majority_class = unique_classes[np.argmax(counts)]
            
            # Only change if majority is strong (> 60%)
            if np.max(counts) / len(neighborhood) > 0.6:
                refined[i, j] = majority_class
    
    return refined


# Hugging Face API-based verification (fallback if SegEarth-OV not available)
ADE20K_TO_LANDCOVER = {
    # Background / Misc
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
    # Buildings
    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2,
    # Vegetation
    21: 5, 22: 5, 23: 5, 24: 5, 25: 5, 26: 5, 27: 5, 28: 5, 29: 5, 30: 5,
    # Water
    31: 4, 32: 4, 33: 4, 34: 4, 35: 4, 36: 4, 37: 4, 38: 4, 39: 4, 40: 4,
    # Roads
    41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 3, 47: 3, 48: 3, 49: 3, 50: 3,
    # Agricultural
    51: 6, 52: 6, 53: 6, 54: 6, 55: 6, 56: 6, 57: 6, 58: 6, 59: 6, 60: 6,
}


def predict_with_huggingface_api(
    image: Image.Image,
    model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512",
) -> np.ndarray:
    """
    Get prediction from Hugging Face Inference API.
    
    Args:
        image: PIL Image to segment
        model_id: Hugging Face model ID
        
    Returns:
        Prediction mask with class indices
    """
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    # Call API
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Content-Type": "image/jpeg"}
    
    try:
        response = requests.post(api_url, headers=headers, data=img_bytes, timeout=30)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
        
        # Parse response
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            # Some models return a list of masks
            mask = np.array(result[0])
        elif isinstance(result, dict) and 'mask' in result:
            mask = np.array(result['mask'])
        elif isinstance(result, dict) and 'label' in result:
            # Some models return label maps
            mask = np.array(result['label'])
        else:
            # Try to convert directly
            mask = np.array(result)
        
        return mask
        
    except Exception as e:
        print(f"API request failed: {e}")
        return None


def map_ade20k_to_landcover(ade20k_mask: np.ndarray) -> np.ndarray:
    """
    Map ADE20K class indices to land cover class indices.
    
    Args:
        ade20k_mask: Mask with ADE20K class indices
        
    Returns:
        Mask with land cover class indices (0-7)
    """
    # Create output mask
    landcover_mask = np.zeros_like(ade20k_mask)
    
    # Map each ADE20K class to land cover class
    for ade20k_class in np.unique(ade20k_mask):
        # Use mapping dictionary, default to background (0)
        landcover_class = ADE20K_TO_LANDCOVER.get(int(ade20k_class), 0)
        landcover_mask[ade20k_mask == ade20k_class] = landcover_class
    
    return landcover_mask


def verify_segmentation_with_huggingface_api(
    image: Image.Image,
    segformer_pred: np.ndarray,
    model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512",
) -> np.ndarray:
    """
    Verify SegFormer prediction using Hugging Face API (fallback).
    
    Pipeline:
    1. SegFormer makes initial prediction
    2. API verifies the entire segmentation
    3. Keep correct classifications
    4. Replace wrong classifications with API prediction
    
    Args:
        image: Original PIL Image
        segformer_pred: SegFormer prediction mask (8 classes)
        model_id: Hugging Face model ID for verification
        
    Returns:
        Verified and corrected prediction mask
    """
    # Get API prediction
    api_pred_ade20k = predict_with_huggingface_api(image, model_id)
    
    if api_pred_ade20k is None:
        # API failed, return original prediction
        print("Hugging Face API verification failed, using original prediction")
        return segformer_pred
    
    # Resize API prediction to match SegFormer prediction size
    api_pred_ade20k_resized = Image.fromarray(api_pred_ade20k.astype(np.uint8))
    api_pred_ade20k_resized = api_pred_ade20k_resized.resize(
        segformer_pred.shape[::-1], 
        Image.NEAREST
    )
    api_pred_ade20k = np.array(api_pred_ade20k_resized)
    
    # Map ADE20K to land cover classes
    api_pred_landcover = map_ade20k_to_landcover(api_pred_ade20k)
    
    # Compare predictions
    # If they agree, keep SegFormer prediction
    # If they disagree, use API prediction
    
    # Calculate agreement
    agreement = (segformer_pred == api_pred_landcover)
    
    # Create final mask
    final_pred = segformer_pred.copy()
    
    # Correct disagreements with API prediction
    disagreements = ~agreement
    final_pred[disagreements] = api_pred_landcover[disagreements]
    
    # Count corrections
    num_corrections = np.sum(disagreements)
    total_pixels = segformer_pred.size
    correction_rate = num_corrections / total_pixels
    
    print(f"Hugging Face API Verification: {num_corrections}/{total_pixels} pixels corrected ({correction_rate:.2%})")
    
    return final_pred


def verify_background_with_huggingface_api(
    image: Image.Image,
    segformer_pred: np.ndarray,
    background_class: int = 0,
    model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512",
) -> np.ndarray:
    """
    Verify background-specific predictions using Hugging Face API (fallback).
    
    Pipeline:
    1. SegFormer predicts background
    2. API checks if it's really background
    3. If not background, determine the correct class
    
    Args:
        image: Original PIL Image
        segformer_pred: SegFormer prediction mask (8 classes)
        background_class: Index of background class
        model_id: Hugging Face model ID for verification
        
    Returns:
        Verified and corrected prediction mask
    """
    # Get API prediction
    api_pred_ade20k = predict_with_huggingface_api(image, model_id)
    
    if api_pred_ade20k is None:
        return segformer_pred
    
    # Resize API prediction
    api_pred_ade20k_resized = Image.fromarray(api_pred_ade20k.astype(np.uint8))
    api_pred_ade20k_resized = api_pred_ade20k_resized.resize(
        segformer_pred.shape[::-1], 
        Image.NEAREST
    )
    api_pred_ade20k = np.array(api_pred_ade20k_resized)
    
    # Map to land cover classes
    api_pred_landcover = map_ade20k_to_landcover(api_pred_ade20k)
    
    # Find background pixels in SegFormer prediction
    background_mask = (segformer_pred == background_class)
    
    # Check if API agrees on background
    api_agrees_on_background = (api_pred_landcover == background_class)
    
    # Create final mask
    final_pred = segformer_pred.copy()
    
    # Correct background pixels where API disagrees
    background_disagreements = background_mask & ~api_agrees_on_background
    final_pred[background_disagreements] = api_pred_landcover[background_disagreements]
    
    # Count corrections
    num_corrections = np.sum(background_disagreements)
    total_background = np.sum(background_mask)
    
    if total_background > 0:
        correction_rate = num_corrections / total_background
        print(f"Background Verification (Hugging Face API): {num_corrections}/{total_background} background pixels corrected ({correction_rate:.2%})")
    
    return final_pred


# SegEarth-OV-based verification
# SegEarth-OV class names:
# ['background', 'bareland,barren', 'grass', 'pavement', 'road',
#  'tree,forest', 'water,river', 'cropland', 'building,roof,house']
#
# Our app classes are:
# 0 Ignore, 1 Background, 2 Building, 3 Road, 4 Water, 5 Barren, 6 Forest, 7 Agricultural
#
# The old mapping targeted a different class order and could overwrite large
# regions with the wrong label. This mapping keeps only the classes that have a
# reasonable land-cover counterpart.
SEGEARTH_TO_LANDCOVER = {
    0: 1,  # background -> Background
    1: 5,  # bareland,barren -> Barren
    2: 7,  # grass -> Agricultural
    3: 3,  # pavement -> Road
    4: 3,  # road -> Road
    5: 6,  # tree,forest -> Forest
    6: 4,  # water,river -> Water
    7: 7,  # cropland -> Agricultural
    8: 2,  # building,roof,house -> Building
}

SEGEARTH_QUERY_NAMES = [
    "background",
    "bareland,barren",
    "grass",
    "pavement",
    "road",
    "tree,forest",
    "water,river",
    "cropland",
    "building,roof,house",
]

SEGEARTH_BACKGROUND_CLASS = 1
SEGEARTH_CONFIDENCE_THRESHOLD = 0.70
SEGEARTH_BACKGROUND_THRESHOLD = 0.80
SEGEARTH_MIN_LOCAL_SUPPORT = 0.55


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.uint8)).resize(size, Image.NEAREST)
    return np.array(resized)


def _map_segearth_to_landcover(segearth_mask: np.ndarray) -> np.ndarray:
    landcover_mask = np.zeros_like(segearth_mask, dtype=np.int64)
    for segearth_class in np.unique(segearth_mask):
        landcover_mask[segearth_mask == segearth_class] = SEGEARTH_TO_LANDCOVER.get(int(segearth_class), 0)
    return landcover_mask


def _local_support_mask(labels: np.ndarray, min_fraction: float = SEGEARTH_MIN_LOCAL_SUPPORT) -> np.ndarray:
    if not CV2_AVAILABLE:
        return np.ones_like(labels, dtype=bool)

    window = 3
    kernel = np.ones((window, window), dtype=np.float32)
    support = np.zeros_like(labels, dtype=bool)
    for cls in np.unique(labels):
        cls_mask = (labels == cls).astype(np.float32)
        cls_support = cv2.filter2D(cls_mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
        support |= (labels == cls) & (cls_support >= (window * window * min_fraction))
    return support


def _predict_with_segearth_logits(image: Image.Image):
    global segearth_model

    if not SEGEARTH_AVAILABLE:
        print("SegEarth-OV not available")
        return None, None

    try:
        import torch
        from torchvision import transforms

        if segearth_model is None:
            segearth_model = initialize_segearth_model()
            if segearth_model is None:
                return None, None

        img_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            transforms.Resize((768, 768)),
        ])(image).unsqueeze(0)

        device = next(segearth_model.parameters()).device
        img_tensor = img_tensor.to(device)
        batch_img_metas = [
            dict(
                ori_shape=img_tensor.shape[2:],
                img_shape=img_tensor.shape[2:],
                pad_shape=img_tensor.shape[2:],
                padding_size=[0, 0, 0, 0],
            )
        ]

        with torch.no_grad():
            if segearth_model.slide_crop > 0:
                seg_logits = segearth_model.forward_slide(
                    img_tensor,
                    batch_img_metas,
                    segearth_model.slide_stride,
                    segearth_model.slide_crop,
                )
            else:
                seg_logits = segearth_model.forward_feature(img_tensor, batch_img_metas[0]["ori_shape"])

            seg_logits = seg_logits[0] * segearth_model.logit_scale
            seg_probs = seg_logits.softmax(0)

            num_cls = max(int(segearth_model.query_idx.max().item()) + 1, 1)
            num_queries = len(segearth_model.query_idx)
            if num_cls != num_queries:
                cls_index = F.one_hot(segearth_model.query_idx, num_classes=num_cls).T.contiguous()
                cls_index = cls_index.view(num_cls, num_queries, 1, 1).to(seg_probs.device, dtype=seg_probs.dtype)
                seg_probs = (seg_probs.unsqueeze(0) * cls_index).max(1)[0]

            confidence, seg_pred = seg_probs.max(0)
            seg_pred = seg_pred.cpu().numpy().astype(np.int64)
            confidence = confidence.cpu().numpy().astype(np.float32)
            seg_pred[confidence < segearth_model.prob_thd] = int(segearth_model.bg_idx)

        return seg_pred, confidence

    except Exception as e:
        print(f"SegEarth-OV prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def initialize_segearth_model():
    """
    Initialize SegEarth-OV model for verification.
    
    Returns:
        SegEarth-OV model instance or None if initialization fails
    """
    global segearth_model
    
    if not SEGEARTH_AVAILABLE:
        print("SegEarth-OV not available, cannot initialize model")
        return None
    
    if segearth_model is not None:
        return segearth_model
    
    try:
        import torch
        from torchvision import transforms
        
        # Create temporary name file
        name_list = SEGEARTH_QUERY_NAMES
        
        # Use the same path detection logic as the import
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'SegEarth-OV'),  # Relative to project root
            os.path.join(os.getcwd(), 'SegEarth-OV'),  # Current working directory
            '/mnt/c/Users/oumai/OneDrive/Bureau/satellite-landcover-ai/SegEarth-OV',  # WSL path
            'SegEarth-OV',  # Direct path
        ]
        
        segearth_path = None
        for path in possible_paths:
            if os.path.exists(path):
                segearth_path = path
                break
        
        if segearth_path is None:
            print("SegEarth-OV path not found")
            return None
        
        # Create configs directory if it doesn't exist
        configs_dir = os.path.join(segearth_path, 'configs')
        if not os.path.exists(configs_dir):
            os.makedirs(configs_dir, exist_ok=True)
        
        name_path = os.path.join(configs_dir, 'my_name.txt')
        
        with open(name_path, 'w') as f:
            for i, name in enumerate(name_list):
                if i == len(name_list) - 1:
                    f.write(name)
                else:
                    f.write(name + '\n')
        
        # Initialize model - force CPU only to avoid CUDA issues
        device = torch.device('cpu')
        
        # Set environment variable to force CPU.  ``os`` is imported at module
        # scope; importing it here would make it a local name for the whole
        # function and break the path checks above.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        segearth_model = SegEarthSegmentation(
            clip_type='CLIP',
            vit_type='ViT-B/16',
            model_type='SegEarth',
            ignore_residual=True,
            feature_up=False,  # Disable feature up to avoid missing weights
            cls_token_lambda=-0.3,
            name_path=name_path,
            prob_thd=0.1,
            device=device,
        )
        
        # Force model to CPU
        segearth_model = segearth_model.cpu()
        segearth_model.eval()
        
        print(f"SegEarth-OV model initialized on {device}")
        return segearth_model
        
    except Exception as e:
        print(f"Failed to initialize SegEarth-OV model: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_with_segearth(
    image: Image.Image,
) -> np.ndarray:
    """
    Get prediction from SegEarth-OV model.
    
    Args:
        image: PIL Image to segment
        
    Returns:
        Prediction mask with class indices
    """
    seg_pred, _ = _predict_with_segearth_logits(image)
    return seg_pred


def verify_segmentation_with_segearth(
    image: Image.Image,
    segformer_pred: np.ndarray,
) -> np.ndarray:
    """
    Verify SegFormer prediction using SegEarth-OV model.
    
    Pipeline:
    1. SegFormer makes initial prediction
    2. SegEarth-OV verifies the entire segmentation
    3. Keep correct classifications
    4. Replace wrong classifications with SegEarth-OV prediction
    
    Args:
        image: Original PIL Image
        segformer_pred: SegFormer prediction mask (8 classes)
        
    Returns:
        Verified and corrected prediction mask
    """
    # Get SegEarth-OV prediction and confidence.
    segearth_pred, segearth_conf = _predict_with_segearth_logits(image)

    if segearth_pred is None or segearth_conf is None:
        # SegEarth-OV failed, return original prediction
        print("SegEarth-OV verification failed, using original prediction")
        return segformer_pred

    segearth_pred = _resize_mask(segearth_pred, segformer_pred.shape[::-1])
    segearth_conf = _resize_mask((segearth_conf * 255).astype(np.uint8), segformer_pred.shape[::-1]).astype(np.float32) / 255.0
    segearth_pred_landcover = _map_segearth_to_landcover(segearth_pred)
    local_support = _local_support_mask(segearth_pred_landcover)

    final_pred = segformer_pred.copy()
    changed = np.zeros_like(segformer_pred, dtype=bool)

    # Safest changes: background/ignore pixels that SegEarth predicts with
    # good confidence and local agreement.
    backgroundish = (segformer_pred == 0) | (segformer_pred == 1)
    confident = segearth_conf >= SEGEARTH_CONFIDENCE_THRESHOLD
    change_mask = backgroundish & confident & local_support & (segearth_pred_landcover != 0)
    final_pred[change_mask] = segearth_pred_landcover[change_mask]
    changed |= change_mask

    # Only allow non-background corrections when SegEarth is very confident.
    strong_foreground = (~backgroundish) & confident & local_support & (segearth_pred_landcover != segformer_pred)
    strong_foreground &= segearth_conf >= 0.93
    final_pred[strong_foreground] = segearth_pred_landcover[strong_foreground]
    changed |= strong_foreground

    num_corrections = int(changed.sum())
    total_pixels = int(segformer_pred.size)
    correction_rate = num_corrections / total_pixels if total_pixels else 0.0

    print(f"SegEarth-OV Verification: {num_corrections}/{total_pixels} pixels corrected ({correction_rate:.2%})")

    return final_pred


def verify_background_with_segearth(
    image: Image.Image,
    segformer_pred: np.ndarray,
    background_class: int = 0,
) -> np.ndarray:
    """
    Verify background-specific predictions using SegEarth-OV model.
    
    Pipeline:
    1. SegFormer predicts background
    2. SegEarth-OV checks if it's really background
    3. If not background, determine the correct class
    
    Args:
        image: Original PIL Image
        segformer_pred: SegFormer prediction mask (8 classes)
        background_class: Index of background class
        
    Returns:
        Verified and corrected prediction mask
    """
    # Get SegEarth-OV prediction and confidence.
    segearth_pred, segearth_conf = _predict_with_segearth_logits(image)

    if segearth_pred is None or segearth_conf is None:
        return segformer_pred

    segearth_pred = _resize_mask(segearth_pred, segformer_pred.shape[::-1])
    segearth_conf = _resize_mask((segearth_conf * 255).astype(np.uint8), segformer_pred.shape[::-1]).astype(np.float32) / 255.0
    segearth_pred_landcover = _map_segearth_to_landcover(segearth_pred)
    local_support = _local_support_mask(segearth_pred_landcover)

    background_mask = (segformer_pred == background_class)

    final_pred = segformer_pred.copy()
    background_disagreements = (
        background_mask
        & (segearth_pred_landcover != background_class)
        & (segearth_pred_landcover != 0)
        & (segearth_conf >= SEGEARTH_BACKGROUND_THRESHOLD)
        & local_support
    )
    final_pred[background_disagreements] = segearth_pred_landcover[background_disagreements]

    num_corrections = int(np.sum(background_disagreements))
    total_background = int(np.sum(background_mask))

    if total_background > 0:
        correction_rate = num_corrections / total_background
        print(f"Background Verification (SegEarth-OV): {num_corrections}/{total_background} background pixels corrected ({correction_rate:.2%})")

    return final_pred
