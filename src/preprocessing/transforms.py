"""
Transforms used for LoveDA semantic segmentation.

Images:
    RGB, normalized (ImageNet stats), augmented (train only)

Masks:
    class indices (0-7), never normalized, nearest-neighbor interpolation only
"""

import random

import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2 import InterpolationMode

# --- Constants ---
IMAGE_SIZE = (512, 512)
ROTATION = 30
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
ZOOM_SCALE = (0.8, 1.2)
COLOR_JITTER = 0.2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 8
IGNORE_INDEX = 255


class ResizeImageAndMask:
    """
    Resize an (image, mask) pair, guaranteeing bilinear interpolation for the
    image and nearest-neighbor interpolation for the mask.
    """

    def __init__(self, size: tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        return image, mask


class RandomRotationImageAndMask:
    """
    Rotate an (image, mask) pair by the same randomly-drawn angle,
    guaranteeing bilinear interpolation for the image and nearest-neighbor
    interpolation for the mask.
    """

    def __init__(self, degrees: float):
        self.degrees = degrees

    def __call__(self, image: Image.Image, mask: Image.Image):
        angle = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        return image, mask


class RandomZoomImageAndMask:
    """
    Zoom in/out on an (image, mask) pair by the same randomly-drawn scale
    factor, guaranteeing bilinear interpolation for the image and
    nearest-neighbor interpolation for the mask.
    """

    def __init__(self, scale: tuple[float, float]):
        self.scale = scale

    def __call__(self, image: Image.Image, mask: Image.Image):
        factor = random.uniform(*self.scale)
        image = F.affine(
            image, angle=0.0, translate=[0, 0], scale=factor, shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = F.affine(
            mask, angle=0.0, translate=[0, 0], scale=factor, shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
        )
        return image, mask


class SqueezeMask:
    """
    Remove the channel dimension added by v2.ToImage() on a single-channel
    mask, so masks come out as (H, W) per-sample / (B, H, W) per-batch,
    matching what nn.CrossEntropyLoss expects as the target.
    """

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        return mask.squeeze(0)


# ============================================================
# 1. Geometric transforms — applied to BOTH image and mask.
#    Each custom class guarantees bilinear interpolation for the image
#    and nearest-neighbor interpolation for the mask.
# ============================================================
train_joint_transform = v2.Compose([
    ResizeImageAndMask(IMAGE_SIZE),
    RandomRotationImageAndMask(ROTATION),
    v2.RandomHorizontalFlip(HORIZONTAL_FLIP_PROB),
    v2.RandomVerticalFlip(VERTICAL_FLIP_PROB),
    RandomZoomImageAndMask(ZOOM_SCALE),
])

val_joint_transform = v2.Compose([
    ResizeImageAndMask(IMAGE_SIZE),
])

test_joint_transform = v2.Compose([
    ResizeImageAndMask(IMAGE_SIZE),
])

# ============================================================
# 2. Image-only transforms — color jitter, tensor conversion, normalization.
#    Never applied to the mask.
# ============================================================
train_image_transform = v2.Compose([
    v2.ColorJitter(brightness=COLOR_JITTER, contrast=COLOR_JITTER, saturation=COLOR_JITTER),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

val_image_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

test_image_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

# ============================================================
# 3. Mask-only transforms — dtype conversion, no normalization,
#    no color transforms, channel dimension removed. Identical
#    across splits, named separately for readability when wiring
#    into LoveDADataset.
# ============================================================
train_mask_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.int64, scale=False),
    SqueezeMask(),
])

val_mask_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.int64, scale=False),
    SqueezeMask(),
])

test_mask_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.int64, scale=False),
    SqueezeMask(),
])

__all__ = [
    "train_joint_transform",
    "train_image_transform",
    "train_mask_transform",
    "val_joint_transform",
    "val_image_transform",
    "val_mask_transform",
    "test_joint_transform",
    "test_image_transform",
    "test_mask_transform",
]