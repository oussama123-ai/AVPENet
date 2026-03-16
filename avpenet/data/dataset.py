"""
PyTorch Dataset for AVPENet.

Loads audio-visual segments and pain labels from a CSV manifest.

CSV format:
    segment_id, audio_path, video_dir, pain_score, subject_id, age_group
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional

from avpenet.data.audio_preprocessing import AudioPreprocessor, AudioAugmenter
from avpenet.data.visual_preprocessing import VisualPreprocessor, VisualAugmenter


class PainDataset(Dataset):
    """Audio-visual pain assessment dataset.

    Each sample is a 3-second segment with:
        - audio:       mel-spectrogram (1, 128, 300)
        - visual:      face frames     (N_FRAMES, 3, 224, 224) or averaged (3, 224, 224)
        - pain_score:  float in [0, 10]
        - subject_id:  str
        - age_group:   'neonate' or 'adult'
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str = ".",
        split: str = "train",
        n_visual_frames: int = 30,
        augment: bool = False,
        average_frames: bool = True,
        label_smoothing: float = 0.0,
        cache_audio: bool = False,
    ):
        """
        Args:
            csv_path:       Path to manifest CSV file.
            data_root:      Root directory prepended to relative paths in CSV.
            split:          'train', 'val', or 'test' (for logging only).
            n_visual_frames: Number of frames to sample per segment.
            augment:        Apply data augmentation (training only).
            average_frames: If True, return mean over frames → (3, 224, 224).
                            If False, return all frames → (N, 3, 224, 224).
            label_smoothing: Epsilon for soft labels (Eq. 29 from paper).
            cache_audio:    Cache preprocessed mel-spectrograms in memory.
        """
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.split = split
        self.augment = augment
        self.average_frames = average_frames
        self.label_smoothing = label_smoothing

        self.audio_preprocessor  = AudioPreprocessor()
        self.visual_preprocessor = VisualPreprocessor(n_frames=n_visual_frames)

        self.audio_augmenter  = AudioAugmenter() if augment else None
        self.visual_augmenter = VisualAugmenter() if augment else None

        self._audio_cache: dict = {} if cache_audio else None

        # Validate required columns
        required = {"segment_id", "audio_path", "video_dir", "pain_score"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        pain_score = float(row["pain_score"])

        # Label smoothing — Eq. 29
        if self.label_smoothing > 0 and self.augment:
            pain_score = (1 - self.label_smoothing) * pain_score + \
                         self.label_smoothing * 5.0   # uniform centre = 5

        # ── Audio ──────────────────────────────────────────────────────
        audio_path = self.data_root / row["audio_path"]
        seg_id = str(row["segment_id"])

        if self._audio_cache is not None and seg_id in self._audio_cache:
            mel = self._audio_cache[seg_id]
        else:
            mel = self.audio_preprocessor(audio_path)   # (1, 128, 300)
            if self._audio_cache is not None:
                self._audio_cache[seg_id] = mel

        # ── Visual ─────────────────────────────────────────────────────
        video_dir = self.data_root / row["video_dir"]
        frames = self.visual_preprocessor(video_dir)   # (N, 3, 224, 224)

        if self.average_frames:
            visual = frames.mean(dim=0)   # (3, 224, 224)
        else:
            visual = frames               # (N, 3, 224, 224)

        # ── Metadata ───────────────────────────────────────────────────
        subject_id = str(row.get("subject_id", "unknown"))
        age_group  = str(row.get("age_group",  "unknown"))

        return {
            "audio":       mel,
            "visual":      visual,
            "pain_score":  torch.tensor(pain_score, dtype=torch.float32),
            "subject_id":  subject_id,
            "age_group":   age_group,
            "segment_id":  seg_id,
        }


def build_dataloader(
    csv_path: str,
    data_root: str = ".",
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    """Build DataLoader for a given split."""
    is_train = split == "train"
    dataset = PainDataset(
        csv_path=csv_path,
        data_root=data_root,
        split=split,
        augment=is_train,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )


def mixup_batch(
    batch: dict,
    alpha: float = 0.2,
) -> tuple[dict, torch.Tensor]:
    """Apply Mixup augmentation to a batch.

    Implements Equations 30–31 from the paper:
        x̃ = λ x_i + (1−λ) x_j
        ỹ = λ y_i + (1−λ) y_j
        λ ~ Beta(α, α)

    Returns:
        mixed_batch: Dict with mixed audio/visual tensors.
        mixed_labels: Mixed pain score tensor.
    """
    lam = np.random.beta(alpha, alpha)
    B = batch["audio"].size(0)
    idx = torch.randperm(B)

    mixed_audio  = lam * batch["audio"]  + (1 - lam) * batch["audio"][idx]
    mixed_visual = lam * batch["visual"] + (1 - lam) * batch["visual"][idx]
    mixed_labels = lam * batch["pain_score"] + (1 - lam) * batch["pain_score"][idx]

    mixed_batch = dict(batch)
    mixed_batch["audio"]      = mixed_audio
    mixed_batch["visual"]     = mixed_visual
    mixed_batch["pain_score"] = mixed_labels

    return mixed_batch, mixed_labels
