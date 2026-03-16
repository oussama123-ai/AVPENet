# AVPENet Architecture Details

This document provides a detailed description of every architectural component,
mirroring the mathematical notation used in the paper.

---

## Overview

AVPENet is a hierarchical multimodal network with four primary components:

```
Audio Input M ∈ R^{1×128×300}
    │
    ▼
Audio Encoder (ResNet-34)
    │  ea ∈ R^512
    │
    ├──────────────────────────────────────┐
    │                                      │
    ▼                                      ▼
Cross-Modal Fusion (Bidirectional Attention)
    │  f_fused ∈ R^512
    ▼
Regression Head
    │  ŷ ∈ [0, 10]
    ▼
Pain Score

Visual Input I ∈ R^{3×224×224}
    │
    ▼
Visual Encoder (ResNet-50 + Spatial Attention)
    │  ev ∈ R^512
    └──────────────────────────────────────┘
```

---

## Audio Encoder

**Architecture:** Modified ResNet-34  
**Input:** M ∈ R^{1×128×300}  
**Output:** ea ∈ R^512

### Initial Block (Eq. 1)
```
h0 = ReLU(BN(Conv_{7×7,s=2}(M)))
```

### Residual Blocks (Eq. 2)
```
h_{l+1} = ReLU(h_l + F(h_l, {W_l}))
```
Groups: [64, 128, 256, 512] channels × [2, 2, 2, 2] blocks each.

### Global Average Pooling (Eq. 3)
```
ea = Dropout_{p=0.3}(1/(HW) · Σ_{i,j} h4(i,j))
```

---

## Visual Encoder

**Architecture:** ResNet-50 + Spatial Attention  
**Input:** I ∈ R^{3×224×224}  
**Output:** ev ∈ R^512

### Spatial Attention (Eqs. 4–7)
```
M_max     = max_c(h4_vis(:,c,:,:))          ∈ R^{1×H×W}
M_avg     = mean_c(h4_vis(:,c,:,:))         ∈ R^{1×H×W}
M_spatial = σ(Conv_{7×7}([M_max; M_avg]))   ∈ R^{1×H×W}
h4_att    = h4_vis ⊙ M_spatial
```

### Projection (Eq. 8)
```
ev = Dropout_{p=0.4}(W_proj · (1/(HW) · Σ_{i,j} h4_att(i,j)))
```
where W_proj ∈ R^{512×2048}.

---

## Cross-Modal Fusion

**Architecture:** Bidirectional Cross-Attention (Transformer-based)  
**Input:** ea, ev ∈ R^512  
**Output:** f_fused ∈ R^512

### Projections + Positional Encodings (Eqs. 9–10)
```
za = Wa·ea + pa
zv = Wv·ev + pv
```

### Audio-to-Visual Attention (Eqs. 11–13)
```
Qa = W^Q_{av}·za,   Kv = W^K_{av}·zv,   Vv = W^V_{av}·zv
A_{av} = softmax(Qa·Kv^T / √dk)
o_{av} = A_{av}·Vv
```
where dk = 64 (= 512 / 8 heads).

### Visual-to-Audio Attention (Eqs. 14–15)
```
Qv = W^Q_{va}·zv,   Ka = W^K_{va}·za,   Va = W^V_{va}·za
o_{va} = softmax(Qv·Ka^T / √dk)·Va
```

### Concatenation + Feed-Forward (Eqs. 16–20)
```
f_concat = [ea; o_av; ev; o_va]   ∈ R^2048
f1       = ReLU(W1·f_concat + b1)    W1 ∈ R^{1024×2048}
f2       = Dropout_{p=0.3}(f1)
f3       = W2·f2 + b2                W2 ∈ R^{512×1024}
f_fused  = LayerNorm(f3)
```

---

## Regression Head

**Architecture:** 3-layer MLP with sigmoid output  
**Input:** f_fused ∈ R^512  
**Output:** ŷ ∈ [0, 10]

```
h1   = Dropout_{p=0.3}(ReLU(W_r1·f_fused + b_r1))   W_r1 ∈ R^{256×512}   (Eq. 21)
h2   = Dropout_{p=0.2}(ReLU(W_r2·h1 + b_r2))        W_r2 ∈ R^{128×256}   (Eq. 22)
ŷ    = 10·σ(W_r3·h2 + b_r3)                          W_r3 ∈ R^{1×128}     (Eq. 23)
```

---

## Loss Function

**Composite loss** (Eq. 27):
```
L_total = α·L_MSE + β·L_ordinal + γ·L_smooth
        = 1.0·L_MSE + 0.3·L_ordinal + 0.1·L_smooth
```

**MSE** (Eq. 24):
```
L_MSE = (1/B) Σ_i (yi − ŷi)²
```

**Ordinal Consistency** (Eq. 25):
```
L_ord = (1/|P|) Σ_{(i,j)∈P} max(0, m − sign(yi−yj)(ŷi−ŷj))²
where P = {(i,j) : |yi−yj| > m},  m = 0.5
```

**Boundary Smoothness** (Eq. 26):
```
L_smooth = (1/B) Σ_i [max(0, 1−ŷi)² + max(0, ŷi−9)²]
```

---

## Training Protocol

| Stage | Epochs | Trainable Params | Learning Rate |
|-------|--------|-----------------|--------------|
| 1 | 1–30 | Fusion + Head | 1e-3 |
| 2 | 31–100 | All | Encoders: 1e-5, Fusion: 1e-4 |

**Optimiser:** AdamW, weight_decay=0.01  
**Scheduler:** Cosine annealing (Eq. 28)  
**Label smoothing** (Eq. 29): ε=0.1  
**Mixup** (Eqs. 30–31): α=0.2  
**Gradient clipping:** L2 norm ≤ 1.0  
**Hardware:** 4× NVIDIA A100 (40GB), effective batch size = 128  
**Training time:** ~18 hours
