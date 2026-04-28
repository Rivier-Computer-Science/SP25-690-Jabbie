"""
src/train.py
============
End-to-end training pipeline with:
  - validation after every epoch
  - early stopping based on validation balanced accuracy
  - checkpoint saving (best model only)
  - CSV logging of all metrics per epoch

Usage examples:
    python src/train.py --config configs/config.yaml --model resnet18
    python src/train.py --config configs/config.yaml --model resnet18 --no-dropout
    python src/train.py --config configs/config.yaml --model resnet18 --no-augmentation
    python src/train.py --config configs/config.yaml --model resnet18 --loss focal
    python src/train.py --config configs/config.yaml --model simple_cnn
    python src/train.py --config configs/config.yaml --model logistic
"""

import os
import copy
import json
import argparse
import random
import time
import csv
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score
import yaml

from dataset import get_dataloaders
from models  import build_model
from losses  import build_loss


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── One-epoch train ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, log_interval=10):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping (Ch. 7 — prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}", end="\r")

    avg_loss = total_loss / len(loader.dataset)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc


# ─── Main training loop ───────────────────────────────────────────────────────

def train(cfg, model_name, run_name=None, no_dropout=False, no_augmentation=False, loss_type=None):
    """
    Full training loop.
    
    Args:
        cfg:             parsed config dict
        model_name:      'logistic', 'simple_cnn', or 'resnet18'
        run_name:        experiment identifier for file naming
        no_dropout:      if True, disables dropout (ablation A)
        no_augmentation: if True, disables augmentation (ablation B)
        loss_type:       overrides cfg['loss']['type'] if provided (ablation C)
    
    Returns:
        best_val_bal_acc (float)
    """
    seed = cfg["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Experiment: {run_name or model_name}")
    print(f"Device:     {device}")
    print(f"{'='*60}")

    # Apply ablation flags to config copies
    if no_dropout:
        cfg["model"]["dropout_enabled"] = False
    if no_augmentation:
        cfg["augmentation"]["enabled"] = False
    if loss_type:
        cfg["loss"]["type"] = loss_type

    # Data
    train_loader, val_loader, _, class_weights = get_dataloaders(cfg, seed=seed)

    # Model
    model = build_model(model_name, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Loss
    criterion = build_loss(cfg, class_weights=class_weights)

    # Optimizer (Ch. 6 — Adam optimizer)
    train_cfg = cfg["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )

    # LR scheduler (CosineAnnealingLR — Ch. 6)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["scheduler_T_max"],
        eta_min=1e-6
    )

    # Output dirs
    out_dir  = Path(cfg["output"]["results_dir"])
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_id   = run_name or model_name
    log_path = out_dir / f"{run_id}_log.csv"
    ckpt_path = ckpt_dir / f"{run_id}_best.pth"

    # CSV logger
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_bal_acc", "val_loss", "val_bal_acc", "lr"])

    # Early stopping state
    es_cfg       = train_cfg["early_stopping"]
    best_val_acc = -float("inf")
    best_epoch   = 0
    best_weights  = None
    patience_counter = 0

    # Training loop
    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            log_interval=cfg["output"]["log_interval"]
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        # CSV log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                             f"{val_loss:.6f}", f"{val_acc:.6f}", f"{lr:.8f}"])

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_weights  = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "state_dict": best_weights,
                "val_bal_acc": best_val_acc,
                "cfg": cfg,
            }, ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if es_cfg["enabled"] and patience_counter >= es_cfg["patience"]:
            print(f"\nEarly stopping at epoch {epoch}. Best was epoch {best_epoch} "
                  f"with val_bal_acc={best_val_acc:.4f}")
            break

    print(f"\nTraining complete. Best val_bal_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Log:        {log_path}")
    return best_val_acc


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a skin lesion classifier")
    parser.add_argument("--config",           default="configs/config.yaml")
    parser.add_argument("--model",            default="resnet18",
                        choices=["logistic", "simple_cnn", "resnet18"])
    parser.add_argument("--no-dropout",       action="store_true",
                        help="Ablation A: disable dropout")
    parser.add_argument("--no-augmentation",  action="store_true",
                        help="Ablation B: disable data augmentation")
    parser.add_argument("--loss",             default=None,
                        choices=["cross_entropy", "focal"],
                        help="Ablation C: override loss function")
    parser.add_argument("--run-name",         default=None,
                        help="Experiment identifier (default: auto-generated)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Auto-generate run name from flags
    if args.run_name is None:
        parts = [args.model]
        if args.no_dropout:      parts.append("no_dropout")
        if args.no_augmentation: parts.append("no_aug")
        if args.loss:            parts.append(f"loss_{args.loss}")
        args.run_name = "_".join(parts)

    train(
        cfg=cfg,
        model_name=args.model,
        run_name=args.run_name,
        no_dropout=args.no_dropout,
        no_augmentation=args.no_augmentation,
        loss_type=args.loss,
    )
