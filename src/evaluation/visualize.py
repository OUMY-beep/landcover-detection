"""
Visualize original image, ground-truth mask, and predicted mask side by side.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_ROOT))

from preprocessing.transforms import MEAN, NUM_CLASSES, STD

CLASS_NAMES = [
    "Ignore", "Background", "Building", "Road", "Water",
    "Barren", "Forest", "Agricultural",
]
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
MASK_CMAP = ListedColormap(CLASS_COLORS)


def denormalize(image: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization on a (C, H, W) tensor for display,
    returning a (H, W, C) array with values clamped to [0, 1].
    """
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def show_prediction(
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    title: str = "",
) -> None:
    """
    Display Original / Ground Truth / Prediction side by side for one sample.

    Args:
        image: (C, H, W) normalized image tensor.
        ground_truth: (H, W) class index tensor.
        prediction: (H, W) class index tensor.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(denormalize(image))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(ground_truth.cpu().numpy(), cmap=MASK_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(prediction.cpu().numpy(), cmap=MASK_CMAP, vmin=0, vmax=NUM_CLASSES - 1)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    # Shared legend for class colors
    patches = [
        plt.matplotlib.patches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES // 2, fontsize=8)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


def show_batch_predictions(images, masks, predictions, num_samples: int = 3) -> None:
    """
    Show Original / Ground Truth / Prediction for the first `num_samples`
    items of a batch (as produced by predict_dataset).
    """
    num_samples = min(num_samples, images.size(0))
    for i in range(num_samples):
        show_prediction(images[i], masks[i], predictions[i], title=f"Sample {i}")


if __name__ == "__main__":
    from dataset.loader import loaders
    from evaluation.predict import load_unet, predict_dataset

    model = load_unet()
    test_loader = loaders["test"]

    images, masks, predictions = next(predict_dataset(model, test_loader))
    show_batch_predictions(images, masks, predictions, num_samples=3)
