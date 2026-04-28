"""
src/dataset.py
==============
Data loading, stratified splitting, and augmentation pipeline for ISIC 2019.

Usage (prepare splits):
    python src/dataset.py --prepare --config configs/config.yaml
"""

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import yaml


# ─── Globals ──────────────────────────────────────────────────────────────────

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(cfg, split="train"):
    """
    Returns torchvision transforms for the given split.
    Augmentation is applied only during training.
    Normalization uses ImageNet statistics (suitable for ResNet-18 pretrained init).
    """
    aug_cfg = cfg["augmentation"]
    mean = aug_cfg["normalize"]["mean"]
    std  = aug_cfg["normalize"]["std"]
    size = cfg["data"]["image_size"]

    if split == "train" and aug_cfg["enabled"]:
        cj = aug_cfg["color_jitter"]
        rc = aug_cfg["random_resized_crop"]
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=tuple(rc["scale"])),
            transforms.RandomHorizontalFlip(p=aug_cfg["random_horizontal_flip"]),
            transforms.RandomVerticalFlip(p=aug_cfg["random_vertical_flip"]),
            transforms.RandomRotation(degrees=aug_cfg["random_rotation"]),
            transforms.ColorJitter(
                brightness=cj["brightness"],
                contrast=cj["contrast"],
                saturation=cj["saturation"],
                hue=cj["hue"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Validation and test: deterministic resize + center crop only
        return transforms.Compose([
            transforms.Resize(int(size * 1.1)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ISICDataset(Dataset):
    """
    PyTorch Dataset for ISIC 2019.

    Expected directory layout:
        data/ISIC_2019/
            ISIC_2019_Training_Input/
                ISIC_0024306.jpg
                ...
            ISIC_2019_Training_GroundTruth.csv

    The ground-truth CSV has one-hot columns for each class.
    """

    def __init__(self, image_dir, gt_csv, indices, transform=None):
        """
        Args:
            image_dir: path to folder containing .jpg images
            gt_csv:    path to ISIC_2019_Training_GroundTruth.csv
            indices:   list of row indices (into the CSV) to include
            transform: torchvision transform pipeline
        """
        self.image_dir = image_dir
        self.transform = transform

        df = pd.read_csv(gt_csv)
        self.df = df.iloc[indices].reset_index(drop=True)

        # Convert one-hot to integer label
        self.labels = self.df[CLASSES].values.argmax(axis=1)
        self.image_ids = self.df["image"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])
        return image, label


# ─── Split generation ─────────────────────────────────────────────────────────

def make_splits(gt_csv, cfg, seed=42):
    """
    Generates stratified train/val/test splits and saves indices to JSON.
    Stratification ensures each class is proportionally represented in all splits.

    Returns: (train_idx, val_idx, test_idx) as lists of integers.
    """
    df = pd.read_csv(gt_csv)
    labels = df[CLASSES].values.argmax(axis=1)
    all_idx = np.arange(len(df))

    split_cfg = cfg["data"]["splits"]
    test_frac = split_cfg["test"]
    val_frac  = split_cfg["val"]

    # First split: train+val vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(all_idx, labels))

    # Second split: train vs val (relative to trainval)
    val_rel = val_frac / (1.0 - test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
    train_idx_rel, val_idx_rel = next(sss2.split(trainval_idx, labels[trainval_idx]))

    train_idx = trainval_idx[train_idx_rel].tolist()
    val_idx   = trainval_idx[val_idx_rel].tolist()
    test_idx  = test_idx.tolist()

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    splits_file = cfg["data"]["splits_file"]
    os.makedirs(os.path.dirname(splits_file), exist_ok=True)
    with open(splits_file, "w") as f:
        json.dump(splits, f)

    print(f"Splits saved to {splits_file}")
    print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}  |  Test: {len(test_idx):,}")

    # Print class distribution for verification
    for split_name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        split_labels = labels[idx]
        counts = {CLASSES[i]: int((split_labels == i).sum()) for i in range(len(CLASSES))}
        print(f"  {split_name} class distribution: {counts}")

    return train_idx, val_idx, test_idx


def load_splits(cfg):
    """Loads previously saved split indices from JSON."""
    with open(cfg["data"]["splits_file"]) as f:
        splits = json.load(f)
    return splits["train"], splits["val"], splits["test"]


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(cfg, seed=42):
    """
    Returns (train_loader, val_loader, test_loader) and class_weights tensor.
    If splits.json exists, loads existing splits; otherwise creates them.
    """
    data_root = cfg["data"]["root"]
    image_dir = os.path.join(data_root, "ISIC_2019_Training_Input")
    gt_csv    = os.path.join(data_root, "ISIC_2019_Training_GroundTruth.csv")

    splits_file = cfg["data"]["splits_file"]
    if os.path.exists(splits_file):
        train_idx, val_idx, test_idx = load_splits(cfg)
    else:
        train_idx, val_idx, test_idx = make_splits(gt_csv, cfg, seed=seed)

    train_ds = ISICDataset(image_dir, gt_csv, train_idx, transform=get_transforms(cfg, "train"))
    val_ds   = ISICDataset(image_dir, gt_csv, val_idx,   transform=get_transforms(cfg, "val"))
    test_ds  = ISICDataset(image_dir, gt_csv, test_idx,  transform=get_transforms(cfg, "test"))

    bs  = cfg["data"]["batch_size"]
    nw  = cfg["data"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # Compute inverse-frequency class weights for imbalance handling (Ch. 5 — Loss Functions)
    labels = np.array(train_ds.labels)
    class_counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(CLASSES)  # normalize

    return train_loader, val_loader, test_loader, class_weights


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Generate splits.json")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.prepare:
        data_root = cfg["data"]["root"]
        gt_csv    = os.path.join(data_root, "ISIC_2019_Training_GroundTruth.csv")
        if not os.path.exists(gt_csv):
            print(f"ERROR: Ground truth CSV not found at {gt_csv}")
            print("Please download the dataset first. See scripts/download_data.sh")
        else:
            make_splits(gt_csv, cfg, seed=cfg["seed"])
