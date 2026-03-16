"""
Evaluation script for AVPENet.

Runs full evaluation on a test set and reports all metrics from the paper.
Supports age-stratification, per-intensity analysis, and bootstrap CI.

Usage:
    python scripts/evaluate.py \
        --config configs/avpenet_base.yaml \
        --checkpoint outputs/checkpoints/avpenet_best.pth \
        --test_csv data/test.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from avpenet.models.avpenet import AVPENet
from avpenet.data.dataset import build_dataloader
from avpenet.metrics import evaluate, print_results, compute_mae, discretise_pain


def bootstrap_ci(pred, target, metric_fn, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(42)
    scores = []
    n = len(pred)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(pred[idx], target[idx]))
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))
    return lower, upper


def mae_by_intensity(pred, target):
    """Compute MAE broken down by pain intensity range."""
    results = {}
    for label, low, high in [("low (0-3)", 0, 3), ("moderate (4-6)", 3, 6), ("high (7-10)", 6, 10)]:
        mask = (target >= low) & (target <= high)
        if mask.sum() > 0:
            results[label] = {
                "mae": compute_mae(pred[mask], target[mask]),
                "n":   int(mask.sum()),
            }
    return results


@torch.no_grad()
def run_evaluation(model, dataloader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    all_groups = []

    for batch in dataloader:
        audio  = batch["audio"].to(device, non_blocking=True)
        visual = batch["visual"].to(device, non_blocking=True)
        labels = batch["pain_score"]

        with autocast():
            out = model(audio, visual)

        all_preds.append(out["pain_score"].cpu().numpy())
        all_labels.append(labels.numpy())
        all_groups.extend(batch.get("age_group", ["unknown"] * len(labels)))

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.array(all_groups),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate AVPENet")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv",   type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bootstrap",  type=int, default=1000,
                        help="Bootstrap iterations for CI (0 to skip)")
    parser.add_argument("--save_preds", type=str, default=None,
                        help="Save predictions to CSV path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = AVPENet.from_pretrained(args.checkpoint, map_location=str(device))
    model = model.to(device)

    # Load test data
    data_cfg = cfg.get("data", {})
    test_loader = build_dataloader(
        csv_path=args.test_csv,
        data_root=data_cfg.get("root", "."),
        split="test",
        batch_size=args.batch_size,
        num_workers=data_cfg.get("num_workers", 4),
    )

    # Run inference
    print("Running evaluation...")
    preds, labels, groups = run_evaluation(model, test_loader, device)

    # Full metrics
    results = evaluate(preds, labels, groups)
    print_results(results, prefix="Test Set")

    # Bootstrap confidence intervals
    if args.bootstrap > 0:
        print(f"\nBootstrap CIs (n={args.bootstrap}):")
        lo, hi = bootstrap_ci(preds, labels, compute_mae, args.bootstrap)
        print(f"  MAE 95% CI: [{lo:.4f}, {hi:.4f}]")

    # MAE by pain intensity
    print("\nMAE by pain intensity range:")
    intensity_results = mae_by_intensity(preds, labels)
    for label, d in intensity_results.items():
        print(f"  {label:20s}: MAE={d['mae']:.4f}  (n={d['n']})")

    # Save predictions
    if args.save_preds:
        import pandas as pd
        df = pd.DataFrame({
            "pred":       preds,
            "true":       labels,
            "age_group":  groups,
            "pred_class": discretise_pain(preds),
            "true_class": discretise_pain(labels),
        })
        df.to_csv(args.save_preds, index=False)
        print(f"\nPredictions saved to {args.save_preds}")


if __name__ == "__main__":
    main()
