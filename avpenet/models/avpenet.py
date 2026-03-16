"""
AVPENet: Audio-Visual Pain Estimation Network.

Main model class that assembles all four components:
    1. AudioEncoder   — ResNet-34 on mel-spectrograms
    2. VisualEncoder  — ResNet-50 + spatial attention on face images
    3. CrossModalFusion — Bidirectional cross-attention
    4. RegressionHead — FC layers → pain score [0, 10]

Reference: Section "Proposed AVPENet Architecture", Figure 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from avpenet.models.audio_encoder import AudioEncoder, build_audio_encoder
from avpenet.models.visual_encoder import VisualEncoder, build_visual_encoder
from avpenet.models.fusion import CrossModalFusion, build_fusion
from avpenet.models.regression_head import RegressionHead, build_regression_head


class AVPENet(nn.Module):
    """Complete Audio-Visual Pain Estimation Network.

    Processes mel-spectrograms and face images in parallel,
    fuses them through bidirectional cross-modal attention,
    and outputs a continuous pain score on the 0–10 scale.

    Input shapes:
        audio:  (B, 1, 128, 300)  — mel-spectrogram
        visual: (B, 3, 224, 224)  — RGB face image (mean across 30 frames)
                                     or (B, T, 3, 224, 224) for multi-frame

    Output:
        pain_score: (B,)  — continuous score in [0, 10]
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout_audio: float = 0.3,
        dropout_visual: float = 0.4,
        dropout_ff: float = 0.3,
        head_dropout1: float = 0.3,
        head_dropout2: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()

        cfg = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_audio=dropout_audio,
            dropout_visual=dropout_visual,
            dropout_ff=dropout_ff,
            head_dropout1=head_dropout1,
            head_dropout2=head_dropout2,
            pretrained=pretrained,
        )

        self.audio_encoder  = build_audio_encoder(cfg)
        self.visual_encoder = build_visual_encoder(cfg)
        self.fusion         = build_fusion(cfg)
        self.head           = build_regression_head(cfg)

        self.embed_dim = embed_dim

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        return_embeddings: bool = False,
        return_attention: bool = False,
    ) -> dict:
        """
        Args:
            audio:  Mel-spectrogram (B, 1, 128, 300).
            visual: Face image(s). Accepts:
                    - (B, 3, 224, 224)  — single frame or pre-averaged
                    - (B, T, 3, 224, 224) — T frames; average over T
            return_embeddings: If True, include ea and ev in output dict.
            return_attention:  If True, include cross-attention weights.

        Returns:
            dict with keys:
                'pain_score'  (B,)           — required
                'ea'          (B, embed_dim) — if return_embeddings
                'ev'          (B, embed_dim) — if return_embeddings
                'attn_weights' dict          — if return_attention
        """
        # Handle multi-frame visual input
        if visual.dim() == 5:
            B, T, C, H, W = visual.shape
            visual = visual.view(B * T, C, H, W)
            ev_all = self.visual_encoder(visual)          # (B*T, embed_dim)
            ev = ev_all.view(B, T, self.embed_dim).mean(dim=1)   # average over frames
        else:
            ev = self.visual_encoder(visual)              # (B, embed_dim)

        # Audio pathway
        ea = self.audio_encoder(audio)                    # (B, embed_dim)

        # Bidirectional cross-modal fusion
        f_fused, attn_weights = self.fusion(ea, ev)       # (B, embed_dim)

        # Regression to pain score
        pain_score = self.head(f_fused)                   # (B,)

        out = {"pain_score": pain_score}
        if return_embeddings:
            out["ea"] = ea
            out["ev"] = ev
        if return_attention:
            out["attn_weights"] = attn_weights
        return out

    def predict(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method — returns pain score tensor only."""
        with torch.no_grad():
            return self.forward(audio, visual)["pain_score"]

    def freeze_encoders(self):
        """Freeze audio and visual encoder parameters (Stage 1 training)."""
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze all parameters (Stage 2 training)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self,
        lr_encoder: float = 1e-5,
        lr_fusion: float = 1e-4,
    ) -> list[dict]:
        """Return parameter groups with differential learning rates.

        Implements the two-stage learning rate schedule from the paper:
            - Encoders: lr = 1e-5  (fine-tuning pretrained features)
            - Fusion + Head: lr = 1e-4  (training from scratch / fast convergence)
        """
        encoder_params = (
            list(self.audio_encoder.parameters())
            + list(self.visual_encoder.parameters())
        )
        fusion_params = (
            list(self.fusion.parameters())
            + list(self.head.parameters())
        )
        return [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": fusion_params,  "lr": lr_fusion},
        ]

    @classmethod
    def from_config(cls, cfg: dict) -> "AVPENet":
        """Build model from a configuration dictionary."""
        model_cfg = cfg.get("model", cfg)
        return cls(
            embed_dim=model_cfg.get("embed_dim", 512),
            num_heads=model_cfg.get("num_heads", 8),
            dropout_audio=model_cfg.get("dropout_audio", 0.3),
            dropout_visual=model_cfg.get("dropout_visual", 0.4),
            dropout_ff=model_cfg.get("dropout_ff", 0.3),
            head_dropout1=model_cfg.get("head_dropout1", 0.3),
            head_dropout2=model_cfg.get("head_dropout2", 0.2),
            pretrained=model_cfg.get("pretrained", True),
        )

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, map_location: str = "cpu") -> "AVPENet":
        """Load a pretrained AVPENet from a checkpoint file.

        Args:
            checkpoint_path: Path to .pth checkpoint saved by train.py.
            map_location:    Device string for loading (default 'cpu').

        Returns:
            AVPENet model with loaded weights.
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        cfg  = ckpt.get("config", {})
        model = cls.from_config(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def count_parameters(self) -> dict:
        """Count trainable parameters per component."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {
            "audio_encoder":  count(self.audio_encoder),
            "visual_encoder": count(self.visual_encoder),
            "fusion":         count(self.fusion),
            "head":           count(self.head),
            "total":          count(self),
        }
