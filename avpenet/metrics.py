"""
Evaluation Metrics for AVPENet.

Implements all metrics from the paper:
    - MAE  (primary metric)
    - RMSE
    - PCC  (Pearson correlation coefficient)
    - ICC  (intraclass correlation coefficient)
    - Accuracy, F1, Kappa (3-class discretised)

Reference: Section "Evaluation Metrics".
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)
from typing import Union


# ─────────────────────────── Helpers ──────────────────────────────────────────

ArrayLike = Union[np.ndarray, torch.Tensor, list]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def discretise_pain(scores: np.ndarray) -> np.ndarray:
    """Map continuous pain scores to 3-class labels.

    Classes (from the paper):
        0 — Low pain    : [0,  3]
        1 — Moderate    : (3,  6]
        2 — High pain   : (6, 10]
    """
    labels = np.zeros_like(scores, dtype=int)
    labels[(scores > 3) & (scores <= 6)] = 1
    labels[scores > 6] = 2
    return labels


# ─────────────────────────── Regression Metrics ───────────────────────────────

def compute_mae(pred: ArrayLike, target: ArrayLike) -> float:
    """Mean Absolute Error — primary metric."""
    return float(mean_absolute_error(_to_numpy(target), _to_numpy(pred)))


def compute_rmse(pred: ArrayLike, target: ArrayLike) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(_to_numpy(target), _to_numpy(pred))))


def compute_pcc(pred: ArrayLike, target: ArrayLike) -> float:
    """Pearson Correlation Coefficient."""
    p, t = _to_numpy(pred), _to_numpy(target)
    r, _ = stats.pearsonr(p, t)
    return float(r)


def compute_icc(pred: ArrayLike, target: ArrayLike) -> float:
    """Intraclass Correlation Coefficient (two-way random, absolute agreement).

    ICC(2,1) — measures absolute agreement between rater and reference.
    """
    try:
        import pingouin as pg
        import pandas as pd
        p, t = _to_numpy(pred), _to_numpy(target)
        n = len(p)
        data = pd.DataFrame({
            "targets":  np.tile(np.arange(n), 2),
            "raters":   ["pred"] * n + ["true"] * n,
            "scores":   np.concatenate([p, t]),
        })
        icc_result = pg.intraclass_corr(
            data=data,
            targets="targets",
            raters="raters",
            ratings="scores",
        )
        # Return ICC(2,1) absolute agreement
        row = icc_result[icc_result["Type"] == "ICC2"]
        if len(row) > 0:
            return float(row.iloc[0]["ICC"])
    except Exception:
        pass

    # Fallback: simplified ICC computation
    p, t = _to_numpy(pred), _to_numpy(target)
    n = len(p)
    grand_mean = (p.mean() + t.mean()) / 2.0
    ss_total   = np.sum((p - grand_mean) ** 2) + np.sum((t - grand_mean) ** 2)
    ss_within  = np.sum((p - t) ** 2) / 2.0
    ms_between = (ss_total - 2 * ss_within) / (n - 1)
    ms_within  = ss_within / n
    icc = (ms_between - ms_within) / (ms_between + ms_within)
    return float(icc)


# ─────────────────────────── Classification Metrics ──────────────────────────

def compute_classification_metrics(
    pred: ArrayLike,
    target: ArrayLike,
) -> dict:
    """Compute 3-class pain categorisation metrics.

    Returns:
        Dict with keys: accuracy, f1, kappa, confusion_matrix.
    """
    p, t = _to_numpy(pred), _to_numpy(target)
    pred_labels   = discretise_pain(p)
    target_labels = discretise_pain(t)

    acc   = float(accuracy_score(target_labels, pred_labels))
    f1    = float(f1_score(target_labels, pred_labels, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(target_labels, pred_labels))
    cm    = confusion_matrix(target_labels, pred_labels, labels=[0, 1, 2])

    return {
        "accuracy":         acc,
        "f1":               f1,
        "kappa":            kappa,
        "confusion_matrix": cm,
    }


# ─────────────────────────── Comprehensive Evaluation ─────────────────────────

def evaluate(
    pred: ArrayLike,
    target: ArrayLike,
    age_groups: ArrayLike = None,
) -> dict:
    """Compute all evaluation metrics reported in the paper.

    Args:
        pred:       Predicted pain scores.
        target:     Ground-truth pain scores.
        age_groups: Optional array of 'neonate'/'adult' labels
                    for age-stratified evaluation.

    Returns:
        Nested dict with all metrics.
    """
    p, t = _to_numpy(pred), _to_numpy(target)

    results = {
        "mae":  compute_mae(p, t),
        "rmse": compute_rmse(p, t),
        "pcc":  compute_pcc(p, t),
        "icc":  compute_icc(p, t),
    }
    results.update(compute_classification_metrics(p, t))

    # Age-stratified evaluation
    if age_groups is not None:
        groups = _to_numpy(age_groups) if not isinstance(age_groups, np.ndarray) else age_groups
        for group in ["neonate", "adult"]:
            mask = groups == group
            if mask.sum() > 0:
                sub = {
                    "mae":  compute_mae(p[mask], t[mask]),
                    "rmse": compute_rmse(p[mask], t[mask]),
                    "pcc":  compute_pcc(p[mask], t[mask]),
                    "n":    int(mask.sum()),
                }
                sub.update(compute_classification_metrics(p[mask], t[mask]))
                results[group] = sub

    return results


def print_results(results: dict, prefix: str = "") -> None:
    """Pretty-print evaluation results."""
    pad = f"[{prefix}] " if prefix else ""
    print(f"\n{'='*60}")
    print(f"{pad}Evaluation Results")
    print(f"{'='*60}")
    print(f"  MAE:      {results['mae']:.4f}")
    print(f"  RMSE:     {results['rmse']:.4f}")
    print(f"  PCC:      {results['pcc']:.4f}")
    print(f"  ICC:      {results['icc']:.4f}")
    print(f"  Accuracy: {results['accuracy']*100:.1f}%")
    print(f"  F1:       {results['f1']:.4f}")
    print(f"  Kappa:    {results['kappa']:.4f}")

    for group in ["neonate", "adult"]:
        if group in results:
            g = results[group]
            print(f"\n  [{group.capitalize()} (n={g['n']})]")
            print(f"    MAE:      {g['mae']:.4f}")
            print(f"    PCC:      {g['pcc']:.4f}")
            print(f"    Accuracy: {g['accuracy']*100:.1f}%")
    print(f"{'='*60}\n")
