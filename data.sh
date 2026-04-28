#!/usr/bin/env bash
# scripts/download_data.sh
# ========================
# Downloads the ISIC 2019 training dataset.
# Requires: wget or curl, ~9.5 GB free disk space.

set -e

DATA_DIR="data/ISIC_2019"
mkdir -p "$DATA_DIR"

echo "============================================================"
echo " ISIC 2019 Dataset Download"
echo " Target: $DATA_DIR"
echo "============================================================"

# Ground truth CSV
echo ""
echo "[1/2] Downloading ground truth labels..."
wget -nc -O "$DATA_DIR/ISIC_2019_Training_GroundTruth.csv" \
  "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"

# Training images
echo ""
echo "[2/2] Downloading training images (~9.1 GB)..."
echo "This may take a while depending on your internet connection."
wget -nc -O "$DATA_DIR/ISIC_2019_Training_Input.zip" \
  "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"

# Unzip
echo ""
echo "Extracting images..."
unzip -n "$DATA_DIR/ISIC_2019_Training_Input.zip" -d "$DATA_DIR/"

echo ""
echo "Done. Dataset ready at: $DATA_DIR"
echo ""
echo "Verify structure:"
echo "  $DATA_DIR/"
echo "    ISIC_2019_Training_Input/   (contains .jpg files)"
echo "    ISIC_2019_Training_GroundTruth.csv"
echo ""
echo "Next: python src/dataset.py --prepare"
