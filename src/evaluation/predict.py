"""
Run a trained U-Net model on a single image or on a full dataset split,
producing predicted segmentation masks. No metrics are computed here.
"""

import sys
from pathlib import Path

import torch
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_ROOT))

from models.unet import UNet
from preprocessing.transforms import NUM_CLASSES, val_image_transform, val_joint_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = SRC_ROOT.parent / "outputs" / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "unet_best.pth"


def load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT, device: torch.device = DEVICE) -> UNet:
    """
    Build a UNet and load trained weights from a checkpoint.
    """
    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model: UNet, image_path: Path, device: torch.device = DEVICE) -> torch.Tensor:
    """
    Run inference on a single image file.

    Returns:
        Tensor of shape (H, W) with predicted class indices (0-7).
    """
    image = Image.open(image_path).convert("RGB")

    # Use a dummy mask so the joint (resize) transform can run identically
    # to training/eval, then discard it — we only need the image here.
    dummy_mask = Image.new("L", image.size)
    image, _ = val_joint_transform(image, dummy_mask)
    image = val_image_transform(image)

    image = image.unsqueeze(0).to(device)  # add batch dimension: (1, C, H, W)
    logits = model(image)
    prediction = torch.argmax(logits, dim=1).squeeze(0).cpu()  # (H, W)

    return prediction


@torch.no_grad()
def predict_dataset(model: UNet, loader, device: torch.device = DEVICE):
    """
    Run inference over every batch in a DataLoader.

    Yields:
        (images, masks, predictions) tuples, one per batch, all on CPU.
        images:      (B, C, H, W) normalized input tensors
        masks:       (B, H, W) ground-truth class indices
        predictions: (B, H, W) predicted class indices
    """
    for images, masks in loader:
        images_device = images.to(device)
        logits = model(images_device)
        predictions = torch.argmax(logits, dim=1).cpu()
        yield images, masks, predictions


if __name__ == "__main__":
    model = load_model()
    print(f"Model loaded on {DEVICE} from {DEFAULT_CHECKPOINT}")