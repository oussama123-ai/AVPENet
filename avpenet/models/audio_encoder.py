"""
Audio Encoder for AVPENet.

Modified ResNet-34 architecture adapted for mel-spectrogram processing.
Input:  (B, 1, 128, 300)  — mel-spectrogram
Output: (B, 512)           — audio embedding

Reference: Section "Audio Encoder Architecture" of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ResidualBlock(nn.Module):
    """Standard ResNet residual block with identity shortcut.

    Implements: h_{l+1} = ReLU(h_l + F(h_l, {W_l}))   (Eq. 2)
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class AudioEncoder(nn.Module):
    """ResNet-34 based audio encoder for mel-spectrogram processing.

    Architecture:
        - Initial 7×7 conv (stride 2, 64 ch) → BN → ReLU → MaxPool
        - 4 residual block groups: [64, 128, 256, 512] channels
        - Global average pooling
        - Dropout (p=0.3)
        - Output: 512-d embedding

    The first conv layer is modified to accept single-channel
    mel-spectrograms instead of 3-channel RGB images.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        """
        Args:
            embed_dim: Output embedding dimension (default 512).
            dropout:   Dropout probability before output (default 0.3).
            pretrained: Initialize from ImageNet weights where compatible.
        """
        super().__init__()

        # Load ResNet-34 backbone
        if pretrained:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet34(weights=None)

        # --- Modify first conv to accept 1-channel mel-spectrograms ---
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Modified: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        original_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            # Average RGB channel weights to initialise the single input channel
            with torch.no_grad():
                self.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Residual block groups
        self.layer1 = backbone.layer1  # 64  ch, 2 blocks
        self.layer2 = backbone.layer2  # 128 ch, 2 blocks
        self.layer3 = backbone.layer3  # 256 ch, 2 blocks
        self.layer4 = backbone.layer4  # 512 ch, 2 blocks

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

        # Projection head (only if embed_dim ≠ 512)
        self.projector: nn.Module
        if embed_dim != 512:
            self.projector = nn.Linear(512, embed_dim)
        else:
            self.projector = nn.Identity()

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel-spectrogram tensor of shape (B, 1, 128, 300).

        Returns:
            Audio embedding of shape (B, embed_dim).
        """
        # Initial block  — Eq. 1: h0 = ReLU(BN(Conv_{7×7,s=2}(M)))
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)

        # Residual groups  — Eq. 2
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        # Global average pooling  — Eq. 3
        h = self.global_avg_pool(h)          # (B, 512, 1, 1)
        h = h.flatten(1)                     # (B, 512)
        h = self.dropout(h)

        # Optional projection
        ea = self.projector(h)               # (B, embed_dim)
        return ea


def build_audio_encoder(cfg: dict) -> AudioEncoder:
    """Factory function that builds AudioEncoder from config dict."""
    return AudioEncoder(
        embed_dim=cfg.get("embed_dim", 512),
        dropout=cfg.get("dropout_audio", 0.3),
        pretrained=cfg.get("pretrained", True),
    )
