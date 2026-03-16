"""
Loss Functions for AVPENet.

Implements the composite loss function from Section "Loss Function Design",
Equations 24–27:

    L_total = α · L_MSE + β · L_ordinal + γ · L_smooth

where:
    L_MSE     = mean squared error (primary metric)
    L_ordinal = ordinal consistency loss (encourages rank ordering)
    L_smooth  = boundary smoothness loss (prevents extreme predictions)
    α=1.0, β=0.3, γ=0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MSELoss(nn.Module):
    """Mean Squared Error loss — Eq. 24."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class OrdinalConsistencyLoss(nn.Module):
    """Ordinal consistency loss — Eq. 25.

    Penalises violations of the ordinal ranking between pairs:
        L_ord = (1/|P|) Σ_{(i,j) ∈ P} max(0, m − sign(y_i − y_j)(ŷ_i − ŷ_j))²

    where P = {(i,j) : |y_i − y_j| > m}  (pairs with meaningful difference).
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Predicted pain scores (B,).
            target: Ground-truth pain scores (B,).

        Returns:
            Scalar ordinal consistency loss.
        """
        B = pred.size(0)
        if B < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Build pairwise differences
        # target_diff[i,j] = target[i] - target[j]
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)   # (B, B)
        pred_diff   = pred.unsqueeze(1)   - pred.unsqueeze(0)     # (B, B)

        # Valid pairs: |y_i − y_j| > margin
        valid_mask = target_diff.abs() > self.margin

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Ordinal violation: sign mismatch
        sign_target = torch.sign(target_diff)
        violation   = self.margin - sign_target * pred_diff
        loss_pairs  = torch.clamp(violation, min=0.0) ** 2

        return loss_pairs[valid_mask].mean()


class BoundarySmoothnessLoss(nn.Module):
    """Boundary smoothness loss — Eq. 26.

    Penalises predictions that fall below 1 or above 9,
    preventing the model from predicting extreme boundary values.

    L_smooth = (1/B) Σ_i [max(0, 1 − ŷ_i)² + max(0, ŷ_i − 9)²]
    """

    def __init__(self, lower_bound: float = 1.0, upper_bound: float = 9.0):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        lower_violation = torch.clamp(self.lower_bound - pred, min=0.0) ** 2
        upper_violation = torch.clamp(pred - self.upper_bound, min=0.0) ** 2
        return (lower_violation + upper_violation).mean()


class CompositePainLoss(nn.Module):
    """Composite loss combining MSE, ordinal, and smoothness terms.

    L_total = α · L_MSE + β · L_ordinal + γ · L_smooth   (Eq. 27)

    Default weights from paper: α=1.0, β=0.3, γ=0.1
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta:  float = 0.3,
        gamma: float = 0.1,
        ordinal_margin: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.mse_loss      = MSELoss()
        self.ordinal_loss  = OrdinalConsistencyLoss(margin=ordinal_margin)
        self.smooth_loss   = BoundarySmoothnessLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        Args:
            pred:   Predicted pain scores (B,).
            target: Ground-truth pain scores (B,).

        Returns:
            Dict with keys:
                'total':    Composite loss scalar.
                'mse':      MSE component.
                'ordinal':  Ordinal consistency component.
                'smooth':   Boundary smoothness component.
        """
        l_mse     = self.mse_loss(pred, target)
        l_ordinal = self.ordinal_loss(pred, target)
        l_smooth  = self.smooth_loss(pred)

        total = (
            self.alpha * l_mse
            + self.beta  * l_ordinal
            + self.gamma * l_smooth
        )

        return {
            "total":   total,
            "mse":     l_mse.detach(),
            "ordinal": l_ordinal.detach(),
            "smooth":  l_smooth.detach(),
        }


def build_loss(cfg: dict) -> CompositePainLoss:
    """Factory function that builds CompositePainLoss from config dict."""
    loss_cfg = cfg.get("loss", cfg)
    return CompositePainLoss(
        alpha=loss_cfg.get("alpha", 1.0),
        beta=loss_cfg.get("beta",  0.3),
        gamma=loss_cfg.get("gamma", 0.1),
        ordinal_margin=loss_cfg.get("margin", 0.5),
    )
