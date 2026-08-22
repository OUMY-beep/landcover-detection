"""Test-time augmentation that averages *probabilities*, not class labels."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class TestTimeAugmentation:
    """Geometric TTA for square segmentation inputs.

    Each transformed output is restored to the original orientation before
    averaging its softmax probabilities. Averaging argmax masks loses the
    model's confidence and was a source of unstable class changes.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def predict_proba_with_tta(
        self,
        img: Image.Image,
        use_flip: bool = True,
        use_rotate: bool = True,
    ) -> np.ndarray:
        from inference import prepare_image_tensor

        image_tensor = prepare_image_tensor(img).to(self.device)
        variants = [(image_tensor, lambda x: x)]

        if use_flip:
            variants.extend([
                (torch.flip(image_tensor, dims=(3,)), lambda x: torch.flip(x, dims=(2,))),
                (torch.flip(image_tensor, dims=(2,)), lambda x: torch.flip(x, dims=(1,))),
            ])
        if use_rotate:
            for turns in (1, 2, 3):
                variants.append((
                    torch.rot90(image_tensor, turns, dims=(2, 3)),
                    lambda x, k=turns: torch.rot90(x, -k, dims=(1, 2)),
                ))

        probabilities = []
        for tensor, undo in variants:
            probs = torch.softmax(self.model(tensor), dim=1).squeeze(0)
            probabilities.append(undo(probs))

        return torch.stack(probabilities).mean(dim=0).cpu().numpy()
    
    @torch.no_grad()
    def predict_proba_with_tta_advanced(
        self,
        img: Image.Image,
        use_flip: bool = True,
        use_rotate: bool = True,
        use_scale: bool = True,
        use_color_jitter: bool = True,
    ) -> np.ndarray:
        """Advanced TTA with additional augmentations including scaling and color."""
        from inference import prepare_image_tensor
        from torchvision.transforms import functional as TF
        
        image_tensor = prepare_image_tensor(img).to(self.device)
        variants = [(image_tensor, lambda x: x)]

        if use_flip:
            variants.extend([
                (torch.flip(image_tensor, dims=(3,)), lambda x: torch.flip(x, dims=(2,))),
                (torch.flip(image_tensor, dims=(2,)), lambda x: torch.flip(x, dims=(1,))),
                (torch.flip(torch.flip(image_tensor, dims=(2,)), dims=(3,)), lambda x: torch.flip(torch.flip(x, dims=(1,)), dims=(2,))),
            ])
        if use_rotate:
            for turns in (1, 2, 3):
                variants.append((
                    torch.rot90(image_tensor, turns, dims=(2, 3)),
                    lambda x, k=turns: torch.rot90(x, -k, dims=(1, 2)),
                ))
        
        # Add scaled variants if enabled (for multi-scale inference)
        if use_scale:
            for scale in [0.85, 1.15]:
                scaled_tensor = torch.nn.functional.interpolate(
                    image_tensor, 
                    scale_factor=scale, 
                    mode='bilinear', 
                    align_corners=False,
                    recompute_scale_factor=True
                )
                # Create undo function that scales back
                def undo_scale(x, original_size=image_tensor.shape[-2:], scale_factor=scale):
                    return torch.nn.functional.interpolate(
                        x, 
                        size=original_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                variants.append((scaled_tensor, undo_scale))
        
        # Add color jitter variants if enabled
        if use_color_jitter:
            # Apply brightness and contrast adjustments
            for brightness_factor in [0.8, 1.2]:
                for contrast_factor in [0.8, 1.2]:
                    jittered = TF.adjust_brightness(image_tensor, brightness_factor)
                    jittered = TF.adjust_contrast(jittered, contrast_factor)
                    # No undo needed for color jitter as it doesn't change spatial dimensions
                    variants.append((jittered, lambda x: x))

        probabilities = []
        for tensor, undo in variants:
            probs = torch.softmax(self.model(tensor), dim=1).squeeze(0)
            probabilities.append(undo(probs))

        return torch.stack(probabilities).mean(dim=0).cpu().numpy()

    def predict_with_tta(self, img: Image.Image, num_classes: int = 8, **_: object) -> np.ndarray:
        """Compatibility wrapper returning the most likely class per pixel."""
        probabilities = self.predict_proba_with_tta(img)
        if probabilities.shape[0] != num_classes:
            raise ValueError(f"Expected {num_classes} classes, got {probabilities.shape[0]}")
        return np.argmax(probabilities, axis=0).astype(np.int64)


def ensemble_models(
    models: list[torch.nn.Module],
    img: Image.Image,
    device: torch.device,
    weights: list[float] | None = None,
    use_tta: bool = False,
    use_advanced_tta: bool = False,
    voting_method: str = "soft",  # "soft", "hard", or "weighted"
) -> np.ndarray:
    """Advanced ensemble with multiple voting methods."""
    if not models:
        raise ValueError("At least one model is required")
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    if len(weights) != len(models) or not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must match models and sum to 1")

    if use_advanced_tta:
        model_probs = [TestTimeAugmentation(model, device).predict_proba_with_tta_advanced(img) for model in models]
    elif use_tta:
        model_probs = [TestTimeAugmentation(model, device).predict_proba_with_tta(img) for model in models]
    else:
        from inference import prepare_image_tensor
        image_tensor = prepare_image_tensor(img).to(device)
        model_probs = []
        with torch.no_grad():
            for model in models:
                model.eval()
                model_probs.append(torch.softmax(model(image_tensor), dim=1).squeeze(0).cpu().numpy())

    if voting_method == "soft":
        # Soft voting - weighted average of probabilities
        probabilities = sum(weight * probs for weight, probs in zip(weights, model_probs))
        return np.argmax(probabilities, axis=0).astype(np.int64)
    
    elif voting_method == "hard":
        # Hard voting - majority vote from predictions
        predictions = [np.argmax(probs, axis=0) for probs in model_probs]
        
        # Weighted voting
        h, w = predictions[0].shape
        weighted_votes = np.zeros((h, w, 8), dtype=np.float32)  # 8 classes
        
        for pred, weight in zip(predictions, weights):
            for c in range(8):
                weighted_votes[:, :, c] += (pred == c).astype(np.float32) * weight
        
        return np.argmax(weighted_votes, axis=2).astype(np.int64)
    
    elif voting_method == "weighted":
        # Hybrid approach - use probabilities but with class-specific weighting
        # Give more weight to classes that are typically hard to classify
        class_weights = np.array([1.0, 1.2, 1.5, 1.3, 1.4, 1.1, 1.2, 1.1])  # Higher for building, road, water
        
        weighted_probs = []
        for probs, model_weight in zip(model_probs, weights):
            # Apply class-specific weights
            weighted = probs * class_weights[np.newaxis, :, :]
            weighted = weighted / weighted.sum(axis=0, keepdims=True)  # Renormalize
            weighted_probs.append(weighted * model_weight)
        
        final_probs = sum(weighted_probs)
        return np.argmax(final_probs, axis=0).astype(np.int64)
    
    else:
        # Default to soft voting
        probabilities = sum(weight * probs for weight, probs in zip(weights, model_probs))
        return np.argmax(probabilities, axis=0).astype(np.int64)
