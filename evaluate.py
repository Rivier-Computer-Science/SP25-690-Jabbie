"""
src/evaluate.py
===============
Final evaluation on the held-out test set.

Computes:
  - Balanced accuracy (primary metric — macro-averaged recall)
  - Macro AUC-ROC
  - Per-class F1 score
  - Confusion matrix (saved as PNG)
  - Per-class AUC (saved as CSV)
  - Summary table across all experiments

Usage:
    # Evaluate a single checkpoint
    python src/evaluate.py --checkpoint results/checkpoints/resnet18_best.pth

    # Evaluate all checkpoints and generate summary table
    python src/evaluate.py --results-dir results/
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    f1_score, confusion_matrix, classification_report
)
import yaml

from dataset import get_dataloaders, CLASSES
from models  import build_model


# ─── Core evaluation function ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, loader, device, num_classes=8):
    """
    Runs the model on a DataLoader and returns predictions, labels, and probabilities.
    """
    model.eval()
    all_logits, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs  = F.softmax(all_logits, dim=1).numpy()
    all_preds  = all_logits.argmax(dim=1).numpy()

    return all_preds, all_labels, all_probs


def compute_metrics(preds, labels, probs, num_classes=8):
    """
    Computes all evaluation metrics and returns a dict.
    
    Primary metric: balanced_accuracy (macro-averaged recall)
    Rationale: ISIC 2019 is heavily imbalanced; accuracy would be dominated
    by the majority class (NV). Balanced accuracy treats each class equally,
    matching clinical importance (missing rare melanoma is costly).
    """
    metrics = {}

    # Primary: Balanced accuracy (Ch. 8 — Measuring Performance)
    metrics["balanced_accuracy"] = balanced_accuracy_score(labels, preds)

    # Macro AUC-ROC
    try:
        metrics["macro_auc"] = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["macro_auc"] = float("nan")

    # Macro F1
    metrics["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    for i, cls in enumerate(CLASSES):
        metrics[f"f1_{cls}"] = float(per_class_f1[i])

    # Per-class AUC
    for i, cls in enumerate(CLASSES):
        binary_labels = (labels == i).astype(int)
        try:
            metrics[f"auc_{cls}"] = roc_auc_score(binary_labels, probs[:, i])
        except ValueError:
            metrics[f"auc_{cls}"] = float("nan")

    return metrics


# ─── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, run_name, figures_dir):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Normalized Confusion Matrix — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = Path(figures_dir) / f"confusion_matrix_{run_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {out_path}")
    return out_path


# ─── Training curves plot ─────────────────────────────────────────────────────

def plot_training_curves(log_csv, run_name, figures_dir):
    """Plots train/val loss and balanced accuracy over epochs."""
    df = pd.read_csv(log_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(df["epoch"], df["train_loss"], label="Train", color="#2196F3")
    ax1.plot(df["epoch"], df["val_loss"],   label="Val",   color="#F44336")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training & Validation Loss — {run_name}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(df["epoch"], df["train_bal_acc"], label="Train", color="#2196F3")
    ax2.plot(df["epoch"], df["val_bal_acc"],   label="Val",   color="#F44336")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_title(f"Balanced Accuracy — {run_name}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(figures_dir) / f"training_curves_{run_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved: {out_path}")
    return out_path


# ─── Summary table ─────────────────────────────────────────────────────────────

def generate_summary_table(results_list, figures_dir):
    """Builds Table 1 from the paper: comparison across all experiments."""
    rows = []
    for r in results_list:
        rows.append({
            "Experiment":      r["run_name"],
            "Balanced Acc.":   f"{r['metrics']['balanced_accuracy']:.4f}",
            "Macro AUC":       f"{r['metrics']['macro_auc']:.4f}",
            "Macro F1":        f"{r['metrics']['macro_f1']:.4f}",
        })
    df = pd.DataFrame(rows)
    out_path = Path(figures_dir) / "summary_table.csv"
    df.to_csv(out_path, index=False)
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    print(f"Saved: {out_path}")
    return df


# ─── Single checkpoint evaluator ─────────────────────────────────────────────

def evaluate_checkpoint(ckpt_path, cfg_path=None, split="test"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("cfg") or yaml.safe_load(open(cfg_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = ckpt["model_name"]
    model = build_model(model_name, cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    _, val_loader, test_loader, _ = get_dataloaders(cfg)
    loader = test_loader if split == "test" else val_loader

    preds, labels, probs = evaluate_model(model, loader, device)
    metrics = compute_metrics(preds, labels, probs)

    run_name = Path(ckpt_path).stem.replace("_best", "")
    figures_dir = cfg["output"]["figures_dir"]
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(labels, preds, run_name, figures_dir)

    # Check for log file and plot curves
    log_path = Path(cfg["output"]["results_dir"]) / f"{run_name}_log.csv"
    if log_path.exists():
        plot_training_curves(str(log_path), run_name, figures_dir)

    print(f"\nTest metrics for {run_name}:")
    for k, v in metrics.items():
        if not k.startswith("f1_") and not k.startswith("auc_"):
            print(f"  {k}: {v:.4f}")

    return {"run_name": run_name, "metrics": metrics}


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default=None,
                        help="Single checkpoint to evaluate")
    parser.add_argument("--results-dir",  default="results/",
                        help="Evaluate all checkpoints in this directory")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--split",        default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    figures_dir = cfg["output"]["figures_dir"]
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        result = evaluate_checkpoint(args.checkpoint, cfg_path=args.config, split=args.split)
        generate_summary_table([result], figures_dir)
    else:
        # Evaluate all checkpoints in results/checkpoints/
        ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
        checkpoints = list(ckpt_dir.glob("*_best.pth"))
        if not checkpoints:
            print(f"No checkpoints found in {ckpt_dir}")
        else:
            results_list = []
            for ckpt_path in sorted(checkpoints):
                print(f"\nEvaluating: {ckpt_path.name}")
                result = evaluate_checkpoint(str(ckpt_path), cfg_path=args.config, split=args.split)
                results_list.append(result)
            generate_summary_table(results_list, figures_dir)
