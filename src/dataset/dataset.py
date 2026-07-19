from pathlib import Path
from typing import Callable, Optional, Tuple
from torchvision import tv_tensors
from PIL import Image
from torch.utils.data.dataset import Dataset


class LoveDADataset(Dataset):
    """
    PyTorch Dataset for LoveDA semantic segmentation.

    Expects the following folder layout:
        images_dir/xxx.png
        masks_dir/xxx.png   (same filename, single-channel label mask, classes 0-7)

    Args:
        images_dir: path to the folder containing the RGB images.
        masks_dir: path to the folder containing the label masks.
        image_transform: transform applied only to the image (e.g. normalization,
            color jitter). Should NOT be applied to the mask.
        mask_transform: transform applied only to the mask (e.g. resize with
            nearest-neighbor interpolation, conversion to a long tensor).
        joint_transform: transform applied identically to both image and mask
            (e.g. random flip, random rotation), taking and returning a
            (image, mask) pair. Applied before image_transform/mask_transform.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform
        self.image_paths = self._list_images()

    def _list_images(self) -> list:
        """List all .png images found in images_dir, sorted for determinism."""
        image_paths = sorted(self.images_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No .png images found in {self.images_dir}")
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Image.Image]:
        if not 0 <= index < len(self.image_paths):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self.image_paths)}"
            )

        # 1. get the image path
        image_path = self.image_paths[index]

        # 2. get the mask path (same name, in the masks folder, no suffix)
        mask_path = self.masks_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image {image_path}: expected {mask_path}")

        # 3. load the image and convert it to RGB
        image = Image.open(image_path).convert("RGB")

        # 4. load the mask, keep single-channel label values (no RGB conversion)
        mask = Image.open(mask_path).convert("L")

        # 5. apply joint (geometric) transforms to image and mask together
        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        # 6. apply transforms specific to image / mask
        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        if self.joint_transform is not None:
            image = tv_tensors.Image(image)
            mask = tv_tensors.Mask(mask)
            image, mask = self.joint_transform(image, mask)

        return image, mask