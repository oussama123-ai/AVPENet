"""
Audio Preprocessing Pipeline for AVPENet.

Implements Algorithm 1 from the paper:
    Stage 1: Noise reduction (spectral subtraction)
    Stage 2: Resampling (48 kHz → 16 kHz)
    Stage 3: Voice Activity Detection
    Stage 4: Normalisation
    Stage 5: Mel-spectrogram extraction

Output: Mel-spectrogram M ∈ R^{128 × 300}
"""

from __future__ import annotations

import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Union


# ─────────────────────────── Constants ───────────────────────────────────────
TARGET_SR        = 16_000    # target sampling rate after resampling
N_MELS           = 128       # number of mel filterbank channels
N_FFT            = 400       # 25 ms window at 16 kHz
HOP_LENGTH       = 160       # 10 ms hop at 16 kHz
FMIN             = 50        # lowest mel filter frequency
FMAX             = 8_000     # highest mel filter frequency
TARGET_FRAMES    = 300       # fixed time dimension (3 s × 100 frames/s)
NOISE_ALPHA      = 1.0       # spectral subtraction: suppression factor
NOISE_BETA       = 0.1       # spectral subtraction: spectral floor factor
VAD_ENERGY_THR   = 0.02      # VAD energy threshold θ_E
VAD_SF_THR       = 0.5       # VAD spectral flatness threshold θ_SF


# ─────────────────────────── Stage 1: Noise Reduction ────────────────────────

def spectral_subtraction(
    signal: np.ndarray,
    sr: int,
    alpha: float = NOISE_ALPHA,
    beta: float = NOISE_BETA,
    noise_frames: int = 20,
) -> np.ndarray:
    """Spectral subtraction for stationary noise removal.

    Algorithm 1, Stage 1:
        N(f) = FFT(x_silent)
        X'(f) = max(|X(f)| - α|N(f)|, β|X(f)|)

    Args:
        signal:       1-D audio signal.
        sr:           Sampling rate.
        alpha:        Noise suppression factor (default 1.0).
        beta:         Spectral floor factor (default 0.1).
        noise_frames: Number of initial frames assumed to be noise.

    Returns:
        Denoised signal of same length as input.
    """
    # STFT
    stft = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise spectrum from initial frames
    noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Spectral subtraction
    subtracted = magnitude - alpha * noise_mag
    floored = np.maximum(subtracted, beta * magnitude)

    # Reconstruct signal
    stft_denoised = floored * np.exp(1j * phase)
    signal_denoised = librosa.istft(stft_denoised, hop_length=HOP_LENGTH, length=len(signal))
    return signal_denoised.astype(np.float32)


# ─────────────────────────── Stage 3: VAD ─────────────────────────────────────

def voice_activity_detection(
    signal: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 160,
    energy_thr: float = VAD_ENERGY_THR,
    sf_thr: float = VAD_SF_THR,
) -> np.ndarray:
    """Simple energy + spectral flatness VAD.

    Algorithm 1, Stage 3:
        E_i = sum(x_r(t)^2)  for t in frame_i
        SF_i = geometric_mean / arithmetic_mean of power spectrum
        VAD_i = (E_i > θ_E) ∧ (SF_i < θ_SF)

    Returns:
        Boolean mask of length n_frames.
    """
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames ** 2, axis=0)

    # Spectral flatness per frame
    power = frames ** 2
    eps = 1e-10
    geo_mean = np.exp(np.mean(np.log(power + eps), axis=0))
    arith_mean = np.mean(power, axis=0) + eps
    sf = geo_mean / arith_mean

    vad_mask = (energy > energy_thr) & (sf < sf_thr)
    return vad_mask


# ─────────────────────────── Stage 5: Mel-Spectrogram ─────────────────────────

def extract_mel_spectrogram(
    signal: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmin: float = FMIN,
    fmax: float = FMAX,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    """Extract log-mel spectrogram and normalise to [0, 1].

    Algorithm 1, Stages 4–5.

    Returns:
        Normalised mel-spectrogram of shape (n_mels, target_frames).
    """
    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    # Log scale
    log_mel = np.log(mel + 1e-6)

    # Normalise to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

    # Pad or trim to fixed size (target_frames)
    current_frames = log_mel.shape[1]
    if current_frames < target_frames:
        pad = target_frames - current_frames
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant")
    else:
        log_mel = log_mel[:, :target_frames]

    return log_mel.astype(np.float32)


# ─────────────────────────── Full Pipeline ────────────────────────────────────

class AudioPreprocessor:
    """Full audio preprocessing pipeline (Algorithm 1 from paper).

    Usage:
        preprocessor = AudioPreprocessor()
        mel = preprocessor(audio_path)  # Returns (1, 128, 300) torch.Tensor
    """

    def __init__(
        self,
        target_sr: int = TARGET_SR,
        n_mels: int = N_MELS,
        target_frames: int = TARGET_FRAMES,
        apply_denoising: bool = True,
        apply_vad: bool = True,
    ):
        self.target_sr      = target_sr
        self.n_mels         = n_mels
        self.target_frames  = target_frames
        self.apply_denoising = apply_denoising
        self.apply_vad      = apply_vad

    def __call__(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sr: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_input: Path to audio file, or numpy array.
            sr:          Sampling rate (required if audio_input is array).

        Returns:
            Mel-spectrogram tensor of shape (1, 128, 300), dtype float32.
        """
        # Load audio
        if isinstance(audio_input, (str, Path)):
            signal, sr = librosa.load(str(audio_input), sr=None, mono=True)
        else:
            assert sr is not None, "sr must be provided when passing numpy array"
            signal = audio_input.astype(np.float32)

        # Stage 1: Noise reduction
        if self.apply_denoising:
            signal = spectral_subtraction(signal, sr)

        # Stage 2: Resample to 16 kHz
        if sr != self.target_sr:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.target_sr)

        # Stage 4: Peak normalisation
        max_val = np.max(np.abs(signal)) + 1e-8
        signal = signal / max_val

        # Stage 5: Mel-spectrogram
        mel = extract_mel_spectrogram(
            signal,
            sr=self.target_sr,
            n_mels=self.n_mels,
            target_frames=self.target_frames,
        )

        # Return as (1, H, W) tensor
        return torch.from_numpy(mel).unsqueeze(0)   # (1, 128, 300)


# ─────────────────────────── Data Augmentation ────────────────────────────────

class AudioAugmenter:
    """Audio augmentation for training (Algorithm 1 note in paper).

    Applies:
        - Time-stretching (0.9–1.1×)
        - Pitch-shifting (±2 semitones)
        - Additive white noise (SNR 20–30 dB)
    """

    def __init__(
        self,
        time_stretch_range: tuple = (0.9, 1.1),
        pitch_shift_range: tuple = (-2, 2),
        noise_snr_range: tuple = (20, 30),
        p: float = 0.5,
    ):
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range  = pitch_shift_range
        self.noise_snr_range    = noise_snr_range
        self.p = p

    def __call__(self, signal: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        rng = np.random.default_rng()

        # Time-stretching
        if rng.random() < self.p:
            rate = rng.uniform(*self.time_stretch_range)
            signal = librosa.effects.time_stretch(signal, rate=rate)

        # Pitch-shifting
        if rng.random() < self.p:
            n_steps = rng.uniform(*self.pitch_shift_range)
            signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)

        # Additive white noise
        if rng.random() < self.p:
            snr_db = rng.uniform(*self.noise_snr_range)
            signal_power = np.mean(signal ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = rng.normal(0, np.sqrt(noise_power), len(signal)).astype(np.float32)
            signal = signal + noise

        return signal.astype(np.float32)
