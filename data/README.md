# Data Directory

This directory contains the CSV manifests for dataset splits.  
The actual audio/video files are **not** distributed here due to patient privacy.

## Directory Structure

```
data/
├── README.md          ← this file
├── train.csv
├── val.csv
├── test.csv
├── audio/             ← preprocessed mel-spectrograms (.pt tensors)
│   ├── subject_001_seg_001.pt
│   └── ...
└── video/             ← extracted face frame directories
    ├── subject_001_seg_001/
    │   ├── frame_0000.jpg
    │   └── ...
    └── ...
```

## CSV Format

Each manifest CSV must contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `segment_id` | str | Unique segment identifier |
| `audio_path` | str | Relative path to `.wav` file (or `.pt` if pre-processed) |
| `video_dir` | str | Relative path to directory containing frame images |
| `pain_score` | float | Ground truth pain intensity [0, 10] |
| `subject_id` | str | Subject identifier (for leave-one-subject-out splits) |
| `age_group` | str | `"neonate"` or `"adult"` |

### Example

```csv
segment_id,audio_path,video_dir,pain_score,subject_id,age_group
N001_seg001,audio/N001_seg001.wav,video/N001_seg001,7.14,N001,neonate
N001_seg002,audio/N001_seg002.wav,video/N001_seg002,8.57,N001,neonate
A045_seg001,audio/A045_seg001.wav,video/A045_seg001,4.5,A045,adult
```

## Dataset Splits

The dataset was split at the **subject level** to prevent data leakage:

| Split | Subjects | Segments |
|-------|----------|----------|
| Train | 342 (80%) | 2,597 |
| Val   | 43 (10%)  | 325   |
| Test  | 43 (10%)  | 325   |
| **Total** | **428** | **3,247** |

## Pain Score Normalisation

- **Adults**: Self-reported NRS scores (0–10) used directly.
- **Neonates**: NIPS scores (0–7) normalised to 0–10 scale:
  ```
  NRS_normalised = NIPS × (10/7)
  ```
  Final score = median of 3 independent rater scores.

## Site Information

| Site | Type | n_subjects | Pain type |
|------|------|-----------|-----------|
| Site A | Neonatal ICU | 215 | Heel-stick, venipuncture, IM injections |
| Site B | Emergency Dept | 143 | Wound dressing, acute pain |
| Site C | Pain Clinic | 70 | Therapeutic procedures |

## Ethics

Data was collected under IRB approval at all three sites in accordance with
the Declaration of Helsinki. Patient privacy is protected; identifiable
recordings are stored securely and are not publicly distributed.
