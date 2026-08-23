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
        from inference import IMAGE_SIZE, predict_large_image_with_tiles, prepare_image_tensor

        # TTA otherwise first reduces a whole scene to one model input.  That
        # loses the canopy detail needed to distinguish forest from water in
        # the large GeoTIFFs used by this project.  Tiled probabilities retain
        # that detail; geometric TTA is intentionally skipped in this case.
        if max(img.size) > 2 * max(IMAGE_SIZE):
            image_extent = max(img.size)
            grid_size = 4 if image_extent > 8 * max(IMAGE_SIZE) else 3 if image_extent > 4 * max(IMAGE_SIZE) else 2
            global_weight = 0.15 if grid_size >= 4 else 0.25
            return predict_large_image_with_tiles(
                self.model, img, grid_size=grid_size, global_weight=global_weight
            )

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
        from inference import IMAGE_SIZE, predict_large_image_with_tiles, prepare_image_tensor
        from torchvision.transforms import functional as TF

        # Keep the same large-scene behavior as normal TTA.  Applying colour
        # and geometric variants to an already downsampled 9k GeoTIFF cannot
        # recover the canopy texture that tiling preserves.
        if max(img.size) > 2 * max(IMAGE_SIZE):
            image_extent = max(img.size)
            grid_size = 4 if image_extent > 8 * max(IMAGE_SIZE) else 3 if image_extent > 4 * max(IMAGE_SIZE) else 2
            global_weight = 0.15 if grid_size >= 4 else 0.25
            return predict_large_image_with_tiles(
                self.model, img, grid_size=grid_size, global_weight=global_weight
            )
        
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
