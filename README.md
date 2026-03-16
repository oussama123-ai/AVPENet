# AVPENet: Audio-Visual Pain Estimation Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PLOS Digital Health](https://img.shields.io/badge/PLOS%20Digital%20Health-2026-green.svg)]()

Official implementation of **"AVPENet: Pain Estimation from Audio-Visual Fusion of Non-Speech Sounds"**, published in PLOS Digital Health (2026).

> **Authors:** Sami Naouali¹, Oussama El Othmani²·³  
> ¹ King Faisal University, Al Ahsa, Saudi Arabia  
> ² Military Academy of Fondouk Jedid, Nabeul, Tunisia  
> ³ Military Research Center, Aouina, Tunisia

---

## Overview

AVPENet is a multimodal deep learning framework that estimates continuous pain intensity by fusing:
- **Non-speech audio cues** (cries, moans, vocalizations) via mel-spectrogram + ResNet-34
- **Facial expressions** via facial landmark CNN + ResNet-50 with spatial attention
- **Bidirectional cross-modal attention** fusion (transformer-based)

### Key Results

| Method | MAE ↓ | PCC ↑ | ICC ↑ | Accuracy (%) ↑ |
|--------|-------|-------|-------|----------------|
| Audio-Only | 1.47 | 0.71 | 0.68 | 64.3 |
| Visual-Only | 1.23 | 0.78 | 0.74 | 69.8 |
| **AVPENet (Ours)** | **0.89** | **0.89** | **0.86** | **81.4** |

- **39% improvement** over audio-only baseline
- **28% improvement** over visual-only baseline
- Validated on **3,247 recordings** from **428 subjects** (215 neonates + 213 adults)
- Cross-age MAE: **0.94** (neonates), **0.84** (adults)

---

## Architecture

```
Audio Input (Mel-Spectrogram 128×300)
        │
   ResNet-34 Encoder
        │
   Audio Embedding (512-d)
        │                    ┌─────────────────────────┐
        └────────────────────►  Cross-Modal Fusion      │
                             │  (Bidirectional Attention)│──► Regression Head ──► Pain Score (0-10)
        ┌────────────────────►                          │
        │                    └─────────────────────────┘
   Visual Embedding (512-d)
        │
   ResNet-50 + Spatial Attention
        │
Visual Input (224×224×3 RGB)
```

---

## Installation

```bash
git clone https://github.com/oussama123-ai/AVPENet.git
cd AVPENet
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.7 (for GPU training)
- See `requirements.txt` for full list

---

## Quick Start

### Inference on a Single Audio-Visual Segment

```python
from avpenet import AVPENet
from avpenet.inference import PainEstimator

# Load pretrained model
estimator = PainEstimator.from_pretrained("avpenet_base")

# Run on a 3-second audio-visual segment
pain_score = estimator.predict(
    audio_path="sample_audio.wav",
    video_path="sample_video.mp4"
)
print(f"Estimated pain score: {pain_score:.2f} / 10")
```

### Training

```bash
# Single GPU
python scripts/train.py --config configs/avpenet_base.yaml

# Multi-GPU (4× A100)
torchrun --nproc_per_node=4 scripts/train.py --config configs/avpenet_base.yaml
```

### Evaluation

```bash
python scripts/evaluate.py \
    --config configs/avpenet_base.yaml \
    --checkpoint checkpoints/avpenet_best.pth \
    --test_csv data/test.csv
```

---

## Data Preparation

### Dataset Structure

Organize your data as follows:

```
data/
├── train.csv
├── val.csv
├── test.csv
├── audio/
│   ├── subject_001_seg_001.wav
│   └── ...
└── video/
    ├── subject_001_seg_001/
    │   ├── frame_000.jpg
    │   └── ...
    └── ...
```

### CSV Format

Each CSV file should contain:

```
segment_id,audio_path,video_dir,pain_score,subject_id,age_group
subject_001_seg_001,audio/subject_001_seg_001.wav,video/subject_001_seg_001,4.5,001,adult
...
```

### Preprocessing

```bash
# Preprocess audio (noise reduction, mel-spectrogram extraction)
python scripts/preprocess_audio.py --input_dir raw_audio/ --output_dir data/audio/

# Preprocess video (face detection, alignment, landmark extraction)
python scripts/preprocess_video.py --input_dir raw_video/ --output_dir data/video/
```

---

## Model Components

### Audio Encoder (`avpenet/models/audio_encoder.py`)
- Modified ResNet-34 adapted for mel-spectrogram processing
- Input: `(B, 1, 128, 300)` mel-spectrogram
- Output: `(B, 512)` embedding

### Visual Encoder (`avpenet/models/visual_encoder.py`)
- ResNet-50 with spatial attention augmentation
- Input: `(B, 3, 224, 224)` RGB face image
- Output: `(B, 512)` embedding

### Cross-Modal Fusion (`avpenet/models/fusion.py`)
- Transformer-based bidirectional cross-attention
- 8 attention heads, key dimension 64
- Concatenates 4 representations: `[ea; o_av; ev; o_va]`

### Regression Head (`avpenet/models/regression_head.py`)
- 3 FC layers: `512 → 256 → 128 → 1`
- Output scaled to [0, 10] via sigmoid

---

## Training Details

### Two-Stage Protocol

| Stage | Epochs | Learning Rate | Trainable |
|-------|--------|--------------|-----------|
| Stage 1 | 1–30 | 1e-3 | Fusion + Head only |
| Stage 2 | 31–100 | Encoders: 1e-5, Fusion: 1e-4 | All parameters |

### Loss Function

```
L_total = 1.0 × L_MSE + 0.3 × L_ordinal + 0.1 × L_smooth
```

### Data Augmentation
- **Audio**: time-stretching (0.9–1.1×), pitch-shifting (±2 semitones), additive noise (SNR 20–30 dB)
- **Video**: rotation (±15°), translation (±10px), brightness/contrast (±20%)

---

## Configuration

Edit `configs/avpenet_base.yaml` to customize:

```yaml
model:
  audio_encoder: resnet34
  visual_encoder: resnet50
  embed_dim: 512
  num_heads: 8
  dropout_audio: 0.3
  dropout_visual: 0.4

training:
  epochs: 100
  batch_size: 32
  gradient_accumulation: 4
  lr_encoder: 1.0e-5
  lr_fusion: 1.0e-4
  weight_decay: 0.01
  label_smoothing: 0.1

loss:
  alpha: 1.0   # MSE weight
  beta: 0.3    # Ordinal weight
  gamma: 0.1   # Smoothness weight
  margin: 0.5  # Ordinal margin
```

---

## Pretrained Weights

Pretrained model weights are available at:

| Model | MAE | Download |
|-------|-----|----------|
| AVPENet Base | 0.89 | [avpenet_base.pth](https://github.com/oussama123-ai/AVPENet/releases) |

---

## Reproducibility

All experiments used:
- PyTorch 2.0, CUDA 11.7
- 4× NVIDIA A100 GPUs (40GB)
- Random seed: 42
- Mixed precision (AMP) training

```bash
python scripts/train.py --config configs/avpenet_base.yaml --seed 42
```

---

## Citation

If you use AVPENet in your research, please cite:

```bibtex
@article{naouali2026avpenet,
  title={{AVPENet}: Pain Estimation from Audio-Visual Fusion of Non-Speech Sounds},
  author={Naouali, Sami and El Othmani, Oussama},
  journal={PLOS Digital Health},
  year={2026},
  publisher={Public Library of Science}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Military Research Center for GPU compute resources (NVIDIA A100)
- Clinical collaborators at all three participating sites
- This work is dedicated to vulnerable populations whose pain often goes underrecognized

---

## Contact

- **Sami Naouali** — salnawali@kfu.edu.sa  
- **Oussama El Othmani** — GitHub Issues
