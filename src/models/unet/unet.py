import torch
import torch.nn as nn

from .blocks import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """
    U-Net for semantic segmentation.

    Encoder (contracting path): DoubleConv + 4x Down (channels double at each stage).
    Decoder (expansive path): 4x Up (channels halve at each stage) + skip connections.
    Head: OutConv maps the final feature map to per-pixel class logits.
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 8, bilinear: bool = True):
        """
        Args:
            n_channels (int): Number of input image channels (3 for RGB).
            n_classes (int): Number of segmentation classes to predict.
            bilinear (bool): If True, use bilinear upsampling in the decoder;
                if False, use learned transposed convolutions.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Head
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, n_classes, H, W)
        """
        # ---------- Encoder ----------
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ---------- Decoder ----------
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # ---------- Output ----------
        return self.outc(x)


if __name__ == "__main__":
    model = UNet(n_channels=3, n_classes=8, bilinear=True)

    x = torch.randn(2, 3, 512, 512)
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)  # expected: (2, 8, 512, 512)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")