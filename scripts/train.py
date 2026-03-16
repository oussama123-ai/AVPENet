"""
Training script for AVPENet.

Implements the two-stage training protocol from the paper:
    Stage 1 (epochs 1–30):  Freeze encoders, train fusion + head only
    Stage 2 (epochs 31–100): Unfreeze all, differential learning rates

Usage:
    python scripts/train.py --config configs/avpenet_base.yaml
    torchrun --nproc_per_node=4 scripts/train.py --config configs/avpenet_base.yaml
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from avpenet.models.avpenet import AVPENet
from avpenet.data.dataset import build_dataloader, mixup_batch
from avpenet.losses import build_loss
from avpenet.metrics import evaluate, print_results

import yaml


# ─────────────────────────── Utilities ────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    val_mae: float,
    cfg: dict,
    path: str,
):
    """Save model checkpoint including config for reproducibility."""
    state = {
        "epoch":            epoch,
        "val_mae":          val_mae,
        "config":           cfg,
        "model_state_dict": (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, path)


def cosine_lr(
    optimizer,
    epoch: int,
    total_epochs: int,
    lr_min: float,
    lr_max: float,
):
    """Cosine annealing LR schedule — Eq. 28 from paper."""
    import math
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / total_epochs))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * (pg.get("lr_scale", 1.0))
    return lr


# ─────────────────────────── Training Loop ────────────────────────────────────

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch: int,
    cfg: dict,
    writer=None,
    rank: int = 0,
) -> dict:
    model.train()
    training_cfg = cfg.get("training", {})
    use_mixup    = training_cfg.get("mixup", True)
    grad_clip    = training_cfg.get("grad_clip", 1.0)
    accum_steps  = training_cfg.get("gradient_accumulation", 4)

    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        audio  = batch["audio"].to(device, non_blocking=True)
        visual = batch["visual"].to(device, non_blocking=True)
        labels = batch["pain_score"].to(device, non_blocking=True)

        # Mixup augmentation — Eqs. 30–31
        if use_mixup and model.training:
            batch["audio"]      = audio
            batch["visual"]     = visual
            batch["pain_score"] = labels
            batch, labels = mixup_batch(batch, alpha=0.2)
            audio  = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            labels = labels.to(device)

        # Forward pass with AMP
        with autocast():
            out  = model(audio, visual)
            pred = out["pain_score"]
            loss_dict = criterion(pred, labels)
            loss = loss_dict["total"] / accum_steps

        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_dict["total"].item()
        n_batches  += 1

        if rank == 0 and step % 20 == 0:
            global_step = epoch * len(dataloader) + step
            if writer:
                writer.add_scalar("train/loss",         loss_dict["total"].item(), global_step)
                writer.add_scalar("train/loss_mse",     loss_dict["mse"].item(),   global_step)
                writer.add_scalar("train/loss_ordinal", loss_dict["ordinal"].item(), global_step)
                writer.add_scalar("train/loss_smooth",  loss_dict["smooth"].item(), global_step)

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device,
    epoch: int,
    writer=None,
    rank: int = 0,
) -> dict:
    model.eval()
    all_preds  = []
    all_labels = []
    all_groups = []
    total_loss = 0.0

    for batch in dataloader:
        audio  = batch["audio"].to(device, non_blocking=True)
        visual = batch["visual"].to(device, non_blocking=True)
        labels = batch["pain_score"].to(device, non_blocking=True)

        out  = model(audio, visual)
        pred = out["pain_score"]
        loss_dict = criterion(pred, labels)

        all_preds.append(pred.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_groups.extend(batch.get("age_group", []))
        total_loss += loss_dict["total"].item()

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    groups_arr = np.array(all_groups) if all_groups else None

    results = evaluate(all_preds, all_labels, groups_arr)
    results["loss"] = total_loss / len(dataloader)

    if rank == 0 and writer:
        writer.add_scalar("val/mae",      results["mae"],  epoch)
        writer.add_scalar("val/pcc",      results["pcc"],  epoch)
        writer.add_scalar("val/accuracy", results["accuracy"], epoch)
        writer.add_scalar("val/loss",     results["loss"], epoch)

    return results


# ─────────────────────────── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train AVPENet")
    parser.add_argument("--config",   type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--resume",   type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    # ── Distributed setup ───────────────────────────────────────────
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank       = 0
        world_size = 1
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Directories ─────────────────────────────────────────────────
    output_dir = Path(cfg.get("output_dir", "outputs"))
    ckpt_dir   = output_dir / "checkpoints"
    log_dir    = output_dir / "logs"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir)) if rank == 0 else None

    # ── Data ────────────────────────────────────────────────────────
    data_cfg    = cfg.get("data", {})
    train_cfg   = cfg.get("training", {})
    batch_size  = train_cfg.get("batch_size", 32) // world_size

    train_loader = build_dataloader(
        csv_path=data_cfg["train_csv"],
        data_root=data_cfg.get("root", "."),
        split="train",
        batch_size=batch_size,
        num_workers=data_cfg.get("num_workers", 4),
        label_smoothing=train_cfg.get("label_smoothing", 0.1),
    )
    val_loader = build_dataloader(
        csv_path=data_cfg["val_csv"],
        data_root=data_cfg.get("root", "."),
        split="val",
        batch_size=batch_size,
        num_workers=data_cfg.get("num_workers", 4),
    )

    # ── Model ───────────────────────────────────────────────────────
    model = AVPENet.from_config(cfg).to(device)

    if rank == 0:
        param_counts = model.count_parameters()
        print(f"\n{'='*60}")
        print(f"AVPENet Parameters:")
        for k, v in param_counts.items():
            print(f"  {k:20s}: {v:,}")
        print(f"{'='*60}\n")

    if distributed:
        model = DDP(model, device_ids=[rank])

    # ── Loss & Optimiser ────────────────────────────────────────────
    criterion = build_loss(cfg).to(device)
    scaler    = GradScaler()

    lr_enc = train_cfg.get("lr_encoder", 1e-5)
    lr_fus = train_cfg.get("lr_fusion",  1e-4)
    base_model = model.module if distributed else model
    param_groups = base_model.get_parameter_groups(lr_encoder=lr_enc, lr_fusion=lr_fus)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # ── Resume ──────────────────────────────────────────────────────
    start_epoch = 1
    best_mae    = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        base_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_mae    = ckpt.get("val_mae", best_mae)
        if rank == 0:
            print(f"Resumed from epoch {ckpt['epoch']}, best MAE = {best_mae:.4f}")

    # ── Training loop ───────────────────────────────────────────────
    total_epochs   = train_cfg.get("epochs", 100)
    stage2_epoch   = train_cfg.get("stage2_start", 31)
    patience       = train_cfg.get("early_stopping_patience", 15)
    epochs_no_improve = 0

    for epoch in range(start_epoch, total_epochs + 1):

        # Stage 1 → Stage 2 transition
        if epoch == stage2_epoch:
            if rank == 0:
                print(f"\n>>> Entering Stage 2 (epoch {epoch}): unfreezing all parameters")
            base_model.unfreeze_encoders()
        elif epoch < stage2_epoch:
            base_model.freeze_encoders()

        # LR schedule
        lr = cosine_lr(
            optimizer, epoch, total_epochs,
            lr_min=train_cfg.get("lr_min", 1e-6),
            lr_max=lr_fus,
        )

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, epoch, cfg, writer, rank,
        )
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer, rank)
        elapsed = time.time() - t0

        if rank == 0:
            print(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"lr={lr:.2e} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_pcc={val_metrics['pcc']:.4f} | "
                f"val_acc={val_metrics['accuracy']*100:.1f}% | "
                f"time={elapsed:.1f}s"
            )

            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                epochs_no_improve = 0
                save_checkpoint(
                    base_model, optimizer, epoch, best_mae, cfg,
                    path=str(ckpt_dir / "avpenet_best.pth"),
                )
                print(f"  ✓ New best MAE = {best_mae:.4f} — checkpoint saved")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping after {patience} epochs without improvement.")
                    break

        # Periodic checkpoint
        if rank == 0 and epoch % 10 == 0:
            save_checkpoint(
                base_model, optimizer, epoch, val_metrics["mae"], cfg,
                path=str(ckpt_dir / f"avpenet_epoch{epoch:03d}.pth"),
            )

    if rank == 0:
        print(f"\nTraining complete. Best validation MAE = {best_mae:.4f}")
        if writer:
            writer.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
