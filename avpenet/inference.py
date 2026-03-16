"""
Inference module for AVPENet.

Provides a high-level PainEstimator class for running the model
on raw audio/video files without needing to manage preprocessing manually.
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional

from avpenet.data.audio_preprocessing import AudioPreprocessor
from avpenet.data.visual_preprocessing import VisualPreprocessor


class PainEstimator:
    """High-level interface for pain estimation from audio-visual input.

    Usage:
        estimator = PainEstimator.from_pretrained("avpenet_base.pth")
        score = estimator.predict(audio_path="clip.wav", video_path="clip.mp4")
        print(f"Pain score: {score:.2f} / 10")
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        n_visual_frames: int = 30,
    ):
        self.model   = model.to(device).eval()
        self.device  = device
        self.audio_preprocessor  = AudioPreprocessor()
        self.visual_preprocessor = VisualPreprocessor(n_frames=n_visual_frames)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> "PainEstimator":
        """Load a pretrained AVPENet checkpoint."""
        from avpenet.models.avpenet import AVPENet
        model = AVPENet.from_pretrained(checkpoint_path, map_location=device)
        return cls(model=model, device=device)

    @torch.no_grad()
    def predict(
        self,
        audio_path: Union[str, Path],
        video_path: Union[str, Path, list],
    ) -> float:
        """Estimate pain score from an audio-visual segment.

        Args:
            audio_path: Path to a 3-second WAV/MP3 audio file.
            video_path: Path to video directory (frames) or list of frame paths.

        Returns:
            Pain score as a float in [0, 10].
        """
        mel    = self.audio_preprocessor(audio_path).unsqueeze(0).to(self.device)
        frames = self.visual_preprocessor(video_path).unsqueeze(0).to(self.device)
        visual = frames.mean(dim=1)   # average frames → (1, 3, 224, 224)

        out = self.model(mel, visual)
        return float(out["pain_score"].item())

    @torch.no_grad()
    def predict_batch(
        self,
        audio_tensors: torch.Tensor,
        visual_tensors: torch.Tensor,
    ) -> torch.Tensor:
        """Run batch inference on pre-processed tensors.

        Args:
            audio_tensors:  (B, 1, 128, 300)
            visual_tensors: (B, 3, 224, 224)

        Returns:
            Pain score tensor of shape (B,).
        """
        audio_tensors  = audio_tensors.to(self.device)
        visual_tensors = visual_tensors.to(self.device)
        out = self.model(audio_tensors, visual_tensors)
        return out["pain_score"].cpu()

    @torch.no_grad()
    def predict_with_attention(
        self,
        audio_path: Union[str, Path],
        video_path: Union[str, Path, list],
    ) -> dict:
        """Run inference and return attention weights for interpretability.

        Returns:
            Dict with 'pain_score', 'audio_to_visual_attention',
            'visual_to_audio_attention'.
        """
        mel    = self.audio_preprocessor(audio_path).unsqueeze(0).to(self.device)
        frames = self.visual_preprocessor(video_path).unsqueeze(0).to(self.device)
        visual = frames.mean(dim=1)

        out = self.model(mel, visual, return_attention=True)
        return {
            "pain_score":              float(out["pain_score"].item()),
            "audio_to_visual_attention": out["attn_weights"]["audio_to_visual"].cpu(),
            "visual_to_audio_attention": out["attn_weights"]["visual_to_audio"].cpu(),
        }
