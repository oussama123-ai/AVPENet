"""
Regression Head for AVPENet.

Transforms the 512-d fused representation into a continuous
pain score in [0, 10].

Architecture: 512 → 256 → 128 → 1 (scaled sigmoid output)

Reference: Section "Pain Regression Head", Equations 21–23.
"""

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Three-layer MLP regression head.

    Implements Equations 21–23:
        h1 = Dropout(ReLU(W_r1 * f_fused + b_r1))   W_r1 ∈ R^{256×512}
        h2 = Dropout(ReLU(W_r2 * h1 + b_r2))        W_r2 ∈ R^{128×256}
        y_hat = 10 × σ(W_r3 * h2 + b_r3)            W_r3 ∈ R^{1×128}
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ):
        """
        Args:
            embed_dim: Input dimension (fused representation, default 512).
            hidden1:   First hidden layer dimension (default 256).
            hidden2:   Second hidden layer dimension (default 128).
            dropout1:  Dropout after first layer (default 0.3).
            dropout2:  Dropout after second layer (default 0.2).
        """
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, hidden1)    # W_r1
        self.drop1 = nn.Dropout(p=dropout1)

        self.fc2 = nn.Linear(hidden1, hidden2)      # W_r2
        self.drop2 = nn.Dropout(p=dropout2)

        self.fc3 = nn.Linear(hidden2, 1)            # W_r3
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_fused: Fused representation of shape (B, embed_dim).

        Returns:
            Pain score tensor of shape (B,) in range [0, 10].
        """
        # Eq. 21
        h1 = self.drop1(torch.relu(self.fc1(f_fused)))   # (B, 256)
        # Eq. 22
        h2 = self.drop2(torch.relu(self.fc2(h1)))         # (B, 128)
        # Eq. 23
        y_hat = 10.0 * self.sigmoid(self.fc3(h2))         # (B, 1)
        return y_hat.squeeze(-1)                           # (B,)


def build_regression_head(cfg: dict) -> RegressionHead:
    """Factory function that builds RegressionHead from config dict."""
    return RegressionHead(
        embed_dim=cfg.get("embed_dim", 512),
        hidden1=cfg.get("head_hidden1", 256),
        hidden2=cfg.get("head_hidden2", 128),
        dropout1=cfg.get("head_dropout1", 0.3),
        dropout2=cfg.get("head_dropout2", 0.2),
    )
