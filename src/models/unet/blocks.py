import torch
import torch.nn as nn
import torch.nn.functional as func_pad

class DoubleConv(nn.Module):
    """
    Double Convolution Block:
    (Conv2d -> BatchNorm2d -> ReLU) * 2
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of channels after first conv.
                                          Defaults to out_channels if None.
        """
        super(DoubleConv, self).__init__()
        
        if mid_channels is None:
            mid_channels = out_channels

        # Sequential block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DoubleConv block.
        """
        return self.double_conv(x)
    
class Down(nn.Module):
    """
    Downscaling block: MaxPool2d followed by DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Down block.
        """
        return self.maxpool_conv(x)
    
import torch.nn.functional as func_pad


class Up(nn.Module):
    """
    Upscaling block: upsample, concatenate with the skip connection,
    then DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """
        Args:
            in_channels (int): Number of channels of the concatenated tensor
                (upsampled feature map + skip connection).
            out_channels (int): Number of output channels.
            bilinear (bool): If True, use bilinear upsampling followed by a
                1x1 conv to halve the channels. If False, use a learned
                transposed convolution (ConvTranspose2d) instead.
        """
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.Tensor): Feature map coming from the previous (deeper)
                decoder stage — to be upsampled.
            x2 (torch.Tensor): Skip connection feature map coming from the
                corresponding encoder stage.
        """
        x1 = self.up(x1)

        # Pad x1 so its spatial size matches x2, in case input dimensions
        # are not perfectly divisible by 2 at every downsampling step.
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = func_pad.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    """
    Final 1x1 convolution mapping the decoder's feature channels to the
    number of segmentation classes.
    """
    def __init__(self, in_channels: int, num_classes: int):
        """
        Args:
            in_channels (int): Number of input channels (from the last Up block).
            num_classes (int): Number of segmentation classes to predict.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the OutConv block.
        """
        return self.conv(x)