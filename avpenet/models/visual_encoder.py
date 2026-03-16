"""
Visual Encoder for AVPENet.

ResNet-50 with spatial attention augmentation.
Input:  (B, 3, 224, 224)  — RGB face image
Output: (B, 512)           — visual embedding

Reference: Section "Visual Encoder Architecture" of the paper.
Implements Equations 4–8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class SpatialAttentionModule(nn.Module):
    """Spatial attention module applied after the 4th residual group.

    Implements Equations 4–7:
        M_max = max_c(h4_vis)                          (Eq. 4)
        M_avg = mean_c(h4_vis)                         (Eq. 5)
        M_spatial = σ(Conv_{7×7}([M_max; M_avg]))      (Eq. 6)
        h4_att = h4_vis ⊙ M_spatial                   (Eq. 7)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        # Takes 2-channel input: [max-pool, avg-pool] along channel dim
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
            Attention-weighted feature map of shape (B, C, H, W).
        """
        # Channel-wise max and average pooling
        m_max, _ = torch.max(x, dim=1, keepdim=True)   # (B, 1, H, W) — Eq. 4
        m_avg = torch.mean(x, dim=1, keepdim=True)      # (B, 1, H, W) — Eq. 5

        # Concatenate and convolve
        concat = torch.cat([m_max, m_avg], dim=1)        # (B, 2, H, W)
        m_spatial = self.sigmoid(self.conv(concat))       # (B, 1, H, W) — Eq. 6

        # Element-wise multiplication — Eq. 7
        return x * m_spatial                             # (B, C, H, W)


class VisualEncoder(nn.Module):
    """ResNet-50 visual encoder with spatial attention.

    Architecture:
        - ResNet-50 backbone (pretrained on ImageNet / VGGFace2)
        - Spatial attention module after the 4th residual group
        - Global average pooling
        - Linear projection: 2048 → embed_dim
        - Dropout (p=0.4)

    Implements Equations 4–8 from the paper.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        dropout: float = 0.4,
        pretrained: bool = True,
    ):
        """
        Args:
            embed_dim: Output embedding dimension (default 512).
            dropout:   Dropout probability after projection (default 0.4).
            pretrained: Initialize from ImageNet weights.
        """
        super().__init__()

        # ResNet-50 backbone
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = resnet50(weights=None)

        # Extract layers
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1   # 256  ch
        self.layer2 = backbone.layer2   # 512  ch
        self.layer3 = backbone.layer3   # 1024 ch
        self.layer4 = backbone.layer4   # 2048 ch

        # Spatial attention after layer4
        self.spatial_attention = SpatialAttentionModule(kernel_size=7)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection: 2048 → embed_dim  (W_proj in Eq. 8)
        self.projector = nn.Linear(2048, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB face image tensor of shape (B, 3, 224, 224).

        Returns:
            Visual embedding of shape (B, embed_dim).
        """
        # Standard ResNet stem
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)

        # Residual groups
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)   # (B, 2048, H, W)

        # Spatial attention — Eqs. 4–7
        h = self.spatial_attention(h)   # (B, 2048, H, W)

        # Global average pooling
        h = self.global_avg_pool(h)     # (B, 2048, 1, 1)
        h = h.flatten(1)                # (B, 2048)

        # Projection + dropout — Eq. 8
        ev = self.dropout(self.projector(h))   # (B, embed_dim)
        return ev


def build_visual_encoder(cfg: dict) -> VisualEncoder:
    """Factory function that builds VisualEncoder from config dict."""
    return VisualEncoder(
        embed_dim=cfg.get("embed_dim", 512),
        dropout=cfg.get("dropout_visual", 0.4),
        pretrained=cfg.get("pretrained", True),
    )
