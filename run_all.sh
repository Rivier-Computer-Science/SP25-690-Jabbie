#!/usr/bin/env bash
# scripts/run_all.sh
# ==================
# Runs the complete experiment pipeline in order.
# Estimated runtime: ~3-4 hours on a single GPU (T4/V100).

set -e  # Exit on any error

CONFIG="configs/config.yaml"

echo "============================================================"
echo " SKIN LESION CLASSIFICATION — FULL EXPERIMENT PIPELINE"
echo "============================================================"

# Step 1: Prepare data splits
echo ""
echo "[1/8] Preparing stratified data splits..."
python src/dataset.py --prepare --config $CONFIG

# Step 2: Baseline 1 — Logistic Regression
echo ""
echo "[2/8] Training Baseline 1: Logistic Regression..."
python src/train.py --config $CONFIG --model logistic --run-name baseline_logistic

# Step 3: Baseline 2 — Simple CNN
echo ""
echo "[3/8] Training Baseline 2: Simple CNN..."
python src/train.py --config $CONFIG --model simple_cnn --run-name baseline_simple_cnn

# Step 4: Main model — ResNet-18 (full regularization)
echo ""
echo "[4/8] Training Main Model: ResNet-18 (full regularization)..."
python src/train.py --config $CONFIG --model resnet18 --run-name resnet18_full

# Step 5: Ablation A — No Dropout
echo ""
echo "[5/8] Ablation A: ResNet-18 without Dropout..."
python src/train.py --config $CONFIG --model resnet18 --no-dropout --run-name resnet18_no_dropout

# Step 6: Ablation B — No Augmentation
echo ""
echo "[6/8] Ablation B: ResNet-18 without Data Augmentation..."
python src/train.py --config $CONFIG --model resnet18 --no-augmentation --run-name resnet18_no_aug

# Step 7: Ablation C — Focal Loss
echo ""
echo "[7/8] Ablation C: ResNet-18 with Focal Loss..."
python src/train.py --config $CONFIG --model resnet18 --loss focal --run-name resnet18_focal_loss

# Step 8: Evaluate all checkpoints and generate figures
echo ""
echo "[8/8] Evaluating all models on test set..."
python src/evaluate.py --results-dir results/ --config $CONFIG

# Step 9: Grad-CAM failure analysis (main model only)
echo ""
echo "[BONUS] Generating Grad-CAM visualizations..."
python src/gradcam.py \
    --checkpoint results/checkpoints/resnet18_full_best.pth \
    --config $CONFIG \
    --n-samples 16 \
    --split test

echo ""
echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " Results:  results/"
echo " Figures:  figures/"
echo "============================================================"
