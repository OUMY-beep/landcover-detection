"""
SegFormer-B0 adapted for LoveDA semantic segmentation (8 classes).

Uses the official pretrained Hugging Face backbone and replaces the
classification head to output 8 classes instead of the original 150
(ADE20K). The pretrained head weights are discarded and reinitialized
(ignore_mismatched_sizes=True) since the number of classes differs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormer(nn.Module):
    """
    SegFormer-B0 adapted for LoveDA (8 classes).
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, num_classes, H, W) — logits upsampled to
            match the input resolution.
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (B, num_classes, H/4, W/4)

        # SegFormer's decoder head outputs at a reduced resolution;
        # upsample back to the input size so the logits align pixel-for-pixel
        # with the ground-truth masks (needed for CrossEntropyLoss and IoU).
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits


if __name__ == "__main__":
    model = SegFormer(num_classes=8)

    x = torch.randn(2, 3, 512, 512)
    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)  # expected: (2, 8, 512, 512)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {params:,}")