"""
src/gradcam.py
==============
Grad-CAM visualization for failure analysis (Ch. 8 — Measuring Performance).

Grad-CAM (Selvaraju et al., 2017) produces a heatmap showing which spatial regions
of an input image most influenced the model's prediction. This allows us to:
  - Check whether the model attends to clinically relevant features
  - Identify systematic failure modes (e.g., attending to artifacts or rulers)
  - Support the failure analysis section of the report

Usage:
    python src/gradcam.py --checkpoint results/checkpoints/resnet18_best.pth
    python src/gradcam.py --checkpoint results/checkpoints/resnet18_best.pth --n-samples 20
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
import yaml

from dataset import get_dataloaders, CLASSES
from models  import build_model


# ─── Grad-CAM implementation ──────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    For a given input image and target class c:
      1. Forward pass: get logits and record feature map A at target layer
      2. Backward pass: compute dL_c/dA (gradient of loss w.r.t. feature map)
      3. Weight each channel by its global average gradient: alpha_k = mean(dL_c/dA_k)
      4. Cam = ReLU(sum_k alpha_k * A_k) — only positive influences
      5. Upsample to input size and overlay on image
    
    Target layer for ResNet-18: layer4[-1].conv2 (final convolutional layer,
    highest-level semantic features at 7×7 spatial resolution).
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, input_tensor, target_class=None):
        """
        Args:
            input_tensor: (1, C, H, W) float tensor on device
            target_class: int or None (uses predicted class if None)
        
        Returns:
            cam: (H, W) numpy array in [0, 1]
            pred_class: predicted class index
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        pred_class = logits.argmax(dim=1).item()

        if target_class is None:
            target_class = pred_class

        # Backprop on target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Compute channel weights (global average of gradients)
        grads  = self.gradients[0]        # (C, H, W)
        acts   = self.activations[0]      # (C, H, W)
        weights = grads.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input size
        h, w = input_tensor.shape[-2:]
        cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (w, h), resample=Image.BILINEAR)) / 255.0

        return cam, pred_class


# ─── Visualization helpers ────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Converts a normalized ImageNet tensor back to a viewable uint8 numpy image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_cam(image_np, cam, alpha=0.45):
    """Blends Grad-CAM heatmap onto the original image."""
    heatmap = cm.jet(cam)[:, :, :3]                         # (H, W, 3) RGB jet colormap
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


# ─── Main analysis ───────────────────────────────────────────────────────────

def run_gradcam_analysis(ckpt_path, cfg, n_samples=16, split="test", seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt["model_name"]
    model = build_model(model_name, cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if model_name != "resnet18":
        print(f"Grad-CAM is only implemented for resnet18 (got {model_name}).")
        return

    # Target layer: last conv in layer4
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    # Load data
    _, val_loader, test_loader, _ = get_dataloaders(cfg, seed=seed)
    loader = test_loader if split == "test" else val_loader

    # Collect images, labels, predictions
    all_images, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds  = model(images).argmax(dim=1).cpu()
            all_images.append(images.cpu())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            if len(all_labels) >= 500:
                break

    all_images = torch.cat(all_images, dim=0)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    # Separate correct and incorrect predictions
    correct_idx   = np.where(all_labels == all_preds)[0]
    incorrect_idx = np.where(all_labels != all_preds)[0]

    random.seed(seed)
    n_each  = min(n_samples // 2, len(correct_idx), len(incorrect_idx))
    sample_correct   = random.sample(correct_idx.tolist(),   n_each)
    sample_incorrect = random.sample(incorrect_idx.tolist(), n_each)

    figures_dir = Path(cfg["output"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Correct predictions ───────────────────────────────────────────
    _plot_gradcam_grid(
        all_images, all_labels, all_preds, sample_correct,
        gradcam, device, title="Grad-CAM: Correctly Classified Samples",
        out_path=figures_dir / "gradcam_correct.png"
    )

    # ── Plot 2: Incorrect predictions (failure analysis) ─────────────────────
    _plot_gradcam_grid(
        all_images, all_labels, all_preds, sample_incorrect,
        gradcam, device, title="Grad-CAM: Misclassified Samples (Failure Analysis)",
        out_path=figures_dir / "gradcam_failures.png"
    )

    gradcam.remove_hooks()
    print("Grad-CAM analysis complete.")


def _plot_gradcam_grid(images, labels, preds, indices, gradcam, device, title, out_path):
    n     = len(indices)
    ncols = 4
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.2))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for plot_i, data_i in enumerate(indices):
        img_tensor = images[data_i].unsqueeze(0).to(device)
        true_cls   = CLASSES[labels[data_i]]
        pred_cls   = CLASSES[preds[data_i]]
        correct    = labels[data_i] == preds[data_i]

        cam, _ = gradcam(img_tensor.clone().requires_grad_(True))
        img_np = denormalize(images[data_i])
        overlay = overlay_cam(img_np, cam)

        axes[plot_i].imshow(overlay)
        color = "#2ECC71" if correct else "#E74C3C"
        axes[plot_i].set_title(
            f"True: {true_cls}\nPred: {pred_cls}",
            fontsize=8, color=color, fontweight="bold"
        )
        axes[plot_i].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--n-samples",  type=int, default=16,
                        help="Number of samples to visualize (half correct, half wrong)")
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_gradcam_analysis(
        ckpt_path=args.checkpoint,
        cfg=cfg,
        n_samples=args.n_samples,
        split=args.split
    )
