"""
Cross-Modal Fusion Module for AVPENet.

Transformer-based bidirectional cross-modal attention fusion.

Given audio embedding ea ∈ R^512 and visual embedding ev ∈ R^512:
  1. Project into shared space with learnable positional encodings
  2. Audio-to-visual attention: queries from audio, keys/values from visual
  3. Visual-to-audio attention: queries from visual, keys/values from audio
  4. Concatenate: [ea; o_av; ev; o_va] → R^2048
  5. Feed-forward: 2048 → 1024 → 512

Reference: Section "Cross-Modal Fusion Module", Equations 9–20.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Multi-head scaled dot-product attention.

    Implements:
        A = softmax(Q K^T / sqrt(d_k))   (Eq. 12 / 15)
        o = A V
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads   # key dimension per head (64 for 512/8)

        # Projection matrices for Q, K, V
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, embed_dim)  — queries
            key:   (B, embed_dim)  — keys
            value: (B, embed_dim)  — values

        Returns:
            output:  (B, embed_dim)   — attended output
            weights: (B, num_heads, 1, 1)  — attention weights for visualization
        """
        B = query.size(0)

        # For single-token (segment-level) embeddings, add sequence dimension
        # Shape: (B, 1, embed_dim) to use standard multi-head attention
        Q = self.W_Q(query).unsqueeze(1)   # (B, 1, embed_dim)
        K = self.W_K(key).unsqueeze(1)     # (B, 1, embed_dim)
        V = self.W_V(value).unsqueeze(1)   # (B, 1, embed_dim)

        # Reshape for multi-head: (B, num_heads, 1, d_k)
        Q = Q.view(B, 1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, 1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, 1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # (B, h, 1, 1)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum of values
        attended = torch.matmul(weights, V)   # (B, h, 1, d_k)
        attended = attended.transpose(1, 2).contiguous().view(B, 1, self.embed_dim)
        output = self.W_O(attended).squeeze(1)   # (B, embed_dim)

        return output, weights


class CrossModalFusion(nn.Module):
    """Bidirectional cross-modal attention fusion module.

    Architecture:
        1. Linear projection + positional encoding for each modality
        2. Audio-to-visual cross-attention  (o_av)
        3. Visual-to-audio cross-attention  (o_va)
        4. Concatenate: f_concat = [ea; o_av; ev; o_va] ∈ R^2048
        5. Feed-forward network: 2048 → 1024 → 512 → LayerNorm

    Implements Equations 9–20 from the paper.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
        ff_dropout: float = 0.3,
    ):
        """
        Args:
            embed_dim:   Dimension of input embeddings (default 512).
            num_heads:   Number of attention heads (default 8 → d_k = 64).
            dropout:     Attention dropout probability.
            ff_dropout:  Feed-forward dropout probability.
        """
        super().__init__()

        self.embed_dim = embed_dim

        # Linear projections into shared representational space — Eqs. 9–10
        self.W_a = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Learnable positional encodings — Eqs. 9–10
        self.p_a = nn.Parameter(torch.zeros(embed_dim))
        self.p_v = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.p_a, std=0.02)
        nn.init.trunc_normal_(self.p_v, std=0.02)

        # Audio-to-visual attention: Q=audio, K/V=visual — Eqs. 11–13
        self.audio_to_visual_attn = ScaledDotProductAttention(
            embed_dim, num_heads, dropout
        )

        # Visual-to-audio attention: Q=visual, K/V=audio — Eqs. 14–15
        self.visual_to_audio_attn = ScaledDotProductAttention(
            embed_dim, num_heads, dropout
        )

        # Feed-forward network — Eqs. 17–20
        # Input: concatenated [ea; o_av; ev; o_va] ∈ R^{4 * embed_dim}
        concat_dim = 4 * embed_dim   # 2048

        self.ff = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),      # W1: 2048 → 1024  Eq. 17
            nn.ReLU(inplace=True),
            nn.Dropout(p=ff_dropout),                    # Eq. 18
            nn.Linear(concat_dim // 2, embed_dim),       # W2: 1024 → 512   Eq. 19
        )
        self.layer_norm = nn.LayerNorm(embed_dim)         # Eq. 20

    def forward(
        self,
        ea: torch.Tensor,
        ev: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            ea: Audio embedding    (B, embed_dim)
            ev: Visual embedding   (B, embed_dim)

        Returns:
            f_fused:  Fused representation (B, embed_dim)
            attn_weights: Dict with 'audio_to_visual' and 'visual_to_audio' weights
                          for interpretability / visualization.
        """
        # Project + add positional encodings — Eqs. 9–10
        za = self.W_a(ea) + self.p_a   # (B, embed_dim)
        zv = self.W_v(ev) + self.p_v   # (B, embed_dim)

        # Audio-to-visual attention — Eqs. 11–13
        # Q = audio queries, K/V = visual keys/values
        o_av, w_av = self.audio_to_visual_attn(
            query=za, key=zv, value=zv
        )   # o_av: (B, embed_dim)

        # Visual-to-audio attention — Eqs. 14–15
        # Q = visual queries, K/V = audio keys/values
        o_va, w_va = self.visual_to_audio_attn(
            query=zv, key=za, value=za
        )   # o_va: (B, embed_dim)

        # Concatenate four representations — Eq. 16
        f_concat = torch.cat([ea, o_av, ev, o_va], dim=-1)   # (B, 2048)

        # Feed-forward — Eqs. 17–20
        f3 = self.ff(f_concat)
        f_fused = self.layer_norm(f3)   # (B, embed_dim)

        attn_weights = {
            "audio_to_visual": w_av,
            "visual_to_audio": w_va,
        }
        return f_fused, attn_weights


def build_fusion(cfg: dict) -> CrossModalFusion:
    """Factory function that builds CrossModalFusion from config dict."""
    return CrossModalFusion(
        embed_dim=cfg.get("embed_dim", 512),
        num_heads=cfg.get("num_heads", 8),
        dropout=cfg.get("dropout_attn", 0.0),
        ff_dropout=cfg.get("dropout_ff", 0.3),
    )
