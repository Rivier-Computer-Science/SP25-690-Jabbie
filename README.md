# Skin Lesion Classification: Measuring the Impact of Regularization on CNN Generalization

**Course:** COMP-690-AH2 — Topics in Deep Learning (SP26)  
**Chapters covered:** 4 (Deep Neural Networks), 5 (Loss Functions), 6 (Fitting Models), 7 (Gradients & Initialization), 8 (Measuring Performance), 9 (Regularization)

---

## Project Overview

This project investigates how regularization strategies affect the generalization of a convolutional neural network trained on imbalanced medical image data (dermoscopy skin lesion images). Using the publicly available **ISIC 2019** dataset, we compare multiple models and ablations to isolate the effect of dropout, weight decay, data augmentation, and loss function choice on balanced accuracy and AUC-ROC across 8 lesion classes.

**Research question:** *Which combination of regularization strategies most improves CNN generalization on an imbalanced, multi-class medical image classification task?*

---

## Repository Structure

```
skin-lesion-project/
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
├── environment.yml             ← Conda environment spec
├── configs/
│   └── config.yaml             ← All hyperparameters and settings
├── src/
│   ├── dataset.py              ← Data loading, augmentation, stratified splits
│   ├── models.py               ← CNN architectures (baseline CNN, ResNet-18)
│   ├── train.py                ← Training loop with validation
│   ├── evaluate.py             ← Evaluation metrics, confusion matrix, AUC
│   ├── losses.py               ← Cross-entropy and focal loss implementations
│   └── gradcam.py              ← Grad-CAM visualization for failure analysis
├── scripts/
│   ├── download_data.sh        ← ISIC 2019 dataset download instructions
│   ├── run_baselines.sh        ← Run all baseline experiments
│   ├── run_ablations.sh        ← Run all ablation experiments
│   └── run_all.sh              ← Full pipeline: baselines + main + ablations
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory data analysis
│   ├── 02_results_analysis.ipynb ← Results tables and figures
│   └── 03_failure_analysis.ipynb ← Grad-CAM and error analysis
└── results/
    └── .gitkeep                ← Experiment outputs saved here (CSVs, checkpoints)
```

---

## Dataset

**Source:** [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/)  
**Size:** ~25,331 dermoscopy images across 8 skin lesion classes  
**Format:** JPEG images + CSV metadata with ground truth labels

### Classes
| Label | Disease |
|-------|---------|
| MEL | Melanoma |
| NV | Melanocytic nevus |
| BCC | Basal cell carcinoma |
| AK | Actinic keratosis |
| BKL | Benign keratosis |
| DF | Dermatofibroma |
| VASC | Vascular lesion |
| SCC | Squamous cell carcinoma |

### Download Instructions

```bash
# Option 1: Direct download via ISIC Archive API
bash scripts/download_data.sh

# Option 2: Manual download
# 1. Go to: https://challenge.isic-archive.com/data/#2019
# 2. Download "ISIC_2019_Training_Input.zip" (~9.1 GB)
# 3. Download "ISIC_2019_Training_GroundTruth.csv"
# 4. Extract to: data/ISIC_2019/
```

### Data Split (Stratified)
- **Train:** 70% (17,732 images)
- **Validation:** 15% (3,800 images)
- **Test:** 15% (3,799 images)

Splits are stratified by class to preserve class distribution. Split indices are saved to `results/splits.json` for exact reproducibility.

---

## Setup

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate skin-lesion
```

### Option B: pip

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.1+ with CUDA (or CPU fallback)
- ~10 GB disk space for dataset
- GPU recommended (experiments run in ~2–4 hours on a single T4)

---

## Running the Experiments

### Step 1: Download and prepare data

```bash
bash scripts/download_data.sh
python src/dataset.py --prepare  # generates splits.json and validates structure
```

### Step 2: Run all experiments (recommended)

```bash
bash scripts/run_all.sh
```

### Or run individually:

```bash
# Baseline 1: Logistic regression
python src/train.py --config configs/config.yaml --model logistic

# Baseline 2: Simple 3-layer CNN
python src/train.py --config configs/config.yaml --model simple_cnn

# Main model: ResNet-18
python src/train.py --config configs/config.yaml --model resnet18

# Ablation A: ResNet-18 without dropout
python src/train.py --config configs/config.yaml --model resnet18 --no-dropout

# Ablation B: ResNet-18 without augmentation
python src/train.py --config configs/config.yaml --model resnet18 --no-augmentation

# Ablation C: ResNet-18 with focal loss
python src/train.py --config configs/config.yaml --model resnet18 --loss focal
```

### Step 3: Evaluate and generate figures

```bash
python src/evaluate.py --results-dir results/
```

### Step 4: Failure analysis (Grad-CAM)

```bash
python src/gradcam.py --checkpoint results/resnet18_best.pth --split test
```

---

## Key Hyperparameters

All hyperparameters are documented in `configs/config.yaml`. Key values:

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| LR scheduler | CosineAnnealingLR |
| Weight decay | 1e-4 |
| Dropout rate | 0.5 |
| Epochs | 50 |
| Early stopping patience | 7 |
| Focal loss gamma | 2.0 |
| Random seed | 42 |

---

## Reproducing the Reported Results

1. Use the exact environment specified in `environment.yml`
2. Use seed 42 (set in `config.yaml`)
3. Use the split indices in `results/splits.json`
4. Pretrained weights are saved at `results/resnet18_best.pth` (download link in `results/README.md`)
5. Run `python src/evaluate.py --checkpoint results/resnet18_best.pth` to reproduce test metrics

---

## Evaluation Metrics

- **Primary:** Balanced accuracy (macro-averaged recall) — justified by class imbalance
- **Secondary:** Macro AUC-ROC, per-class F1, confusion matrix
- All metrics computed on the held-out test set only after all hyperparameter decisions are finalized

---

## Ethics and Limitations

See Section 8 of the written report for a detailed discussion. Key points:
- ISIC 2019 underrepresents darker skin tones (Fitzpatrick IV–VI), introducing potential demographic bias
- This model is a research tool, not a clinical diagnostic system
- False negatives (missed melanoma) carry higher risk than false positives

---

## License

Code: MIT License  
Dataset: ISIC 2019 — see [ISIC Archive Terms](https://www.isic-archive.com/terms-of-use)
