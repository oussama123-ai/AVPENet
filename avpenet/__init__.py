"""
AVPENet: Audio-Visual Pain Estimation Network

Official implementation of:
"AVPENet: Pain Estimation from Audio-Visual Fusion of Non-Speech Sounds"
PLOS Digital Health, 2026

Authors: Sami Naouali, Oussama El Othmani
"""

from avpenet.models.avpenet import AVPENet
from avpenet.inference import PainEstimator

__version__ = "1.0.0"
__all__ = ["AVPENet", "PainEstimator"]
