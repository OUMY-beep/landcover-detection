"""
Test script to verify segmentation improvements without retraining models.
Tests the key fixes: disabled image enhancement, optimized post-processing, 
improved upsampling, and temperature scaling.
"""

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# Add src to path
SRC_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from web.backend.inference import predict_from_image, get_model

def test_improvements():
    """Test the improved segmentation pipeline."""
    print("=" * 60)
    print("Testing Segmentation Improvements")
    print("=" * 60)
    
    # Check if test image exists
    test_image_path = Path(__file__).resolve().parent / "tmp_test.tif"
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        print("Please provide a test image to verify improvements.")
        return
    
    print(f"\nLoading test image: {test_image_path}")
    img = Image.open(test_image_path)
    print(f"Image size: {img.size}, mode: {img.mode}")
    
    # Test both models with different configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    for model_name in ["unet", "segformer"]:
        print(f"\n{'=' * 60}")
        print(f"Testing {model_name.upper()} model")
        print(f"{'=' * 60}")
        
        try:
            # Test with optimized settings
            print("\n1. Testing with optimized settings (temperature=0.8, advanced=True)")
            pred_optimized = predict_from_image(
                model_name,
                img,
                postprocess=True,
                use_crf=True,
                use_advanced=True,
                confidence_threshold=0.6,
                use_multi_scale=False,
                temperature=0.8  # Sharper predictions
            )
            print(f"   Prediction shape: {pred_optimized.shape}")
            print(f"   Unique classes: {np.unique(pred_optimized)}")
            print(f"   Class distribution:")
            for cls in np.unique(pred_optimized):
                count = np.sum(pred_optimized == cls)
                percentage = (count / pred_optimized.size) * 100
                print(f"     Class {cls}: {count} pixels ({percentage:.2f}%)")
            
            # Test with conservative settings (baseline comparison)
            print("\n2. Testing with conservative settings (temperature=1.0, advanced=False)")
            pred_conservative = predict_from_image(
                model_name,
                img,
                postprocess=False,
                use_crf=False,
                use_advanced=False,
                confidence_threshold=0.6,
                use_multi_scale=False,
                temperature=1.0  # No temperature scaling
            )
            print(f"   Prediction shape: {pred_conservative.shape}")
            print(f"   Unique classes: {np.unique(pred_conservative)}")
            
            # Compare predictions
            print("\n3. Comparison:")
            difference = np.sum(pred_optimized != pred_conservative)
            similarity = 1 - (difference / pred_optimized.size)
            print(f"   Pixel difference: {difference} ({(1-similarity)*100:.2f}%)")
            print(f"   Similarity: {similarity*100:.2f}%")
            
            # Save results for visual inspection
            output_dir = Path(__file__).resolve().parent / "outputs" / "test_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from web.backend.inference import colorize_mask
            optimized_colored = colorize_mask(pred_optimized)
            conservative_colored = colorize_mask(pred_conservative)
            
            optimized_path = output_dir / f"{model_name}_optimized.png"
            conservative_path = output_dir / f"{model_name}_conservative.png"
            
            with open(optimized_path, "wb") as f:
                f.write(optimized_colored)
            with open(conservative_path, "wb") as f:
                f.write(conservative_colored)
            
            print(f"\n4. Saved results:")
            print(f"   Optimized: {optimized_path}")
            print(f"   Conservative: {conservative_path}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_improvements()
