"""
src/models.py
=============
Model architectures used in all experiments:
  - LogisticRegression (Baseline 1 — non-deep)
  - SimpleCNN          (Baseline 2 — shallow deep)
  - ResNet18           (Main model — Ch. 4, residual connections)

All models share a common interface: forward(x) -> logits of shape (B, num_classes).

References:
  - Prince Ch. 4: Deep Neural Networks (residual connections, depth)
  - Prince Ch. 7: Gradients & Initialization (He init, BatchNorm)
  - Prince Ch. 9: Regularization (Dropout, weight decay via optimizer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Baseline 1: Logistic Regression ─────────────────────────────────────────

class LogisticRegression(nn.Module):
    """
    Flattens a 224x224x3 image to a vector and applies a single linear layer.
    Used as the non-deep baseline to establish a floor on performance.
    No hidden layers, no nonlinearities — purely linear classification.
    """
    def __init__(self, num_classes=8, image_size=224):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(image_size * image_size * 3, num_classes)

    def forward(self, x):
        return self.linear(self.flatten(x))


# ─── Baseline 2: Simple 3-layer CNN ──────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    A lightweight 3-block CNN:
        Conv → BN → ReLU → MaxPool  (×3)
        Global Average Pooling
        Dropout (if enabled)
        Linear classifier

    Motivated by Ch. 4 (convolutional networks) and Ch. 9 (dropout regularization).
    Serves as the shallow-deep baseline between logistic regression and ResNet-18.
    BatchNorm is included to stabilize gradients (Ch. 7).
    """
    def __init__(self, num_classes=8, dropout_rate=0.5, dropout_enabled=True):
        super().__init__()
        self.dropout_enabled = dropout_enabled

        self.features = nn.Sequential(
            # Block 1: 224 → 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112 → 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56 → 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Global Average Pooling collapses spatial dims → (B, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        """He (Kaiming) initialization for conv layers (Ch. 7 — Gradients & Init)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        if self.dropout_enabled:
            x = self.dropout(x)
        return self.classifier(x)


# ─── Main Model: ResNet-18 (from scratch) ────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Standard residual block with identity shortcut (He et al., 2016).
    Implements the skip connection from Prince Ch. 4 (residual networks):
        y = F(x, W) + x
    where F is two Conv→BN→ReLU layers.
    The shortcut projects x when channel dims or stride changes (option B in the paper).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # Skip connection (residual addition)
        return self.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 implemented from scratch (not pretrained / fine-tuned).
    Architecture: [Conv, BN, ReLU, MaxPool] → 4 stages × 2 residual blocks → GAP → Linear
    
    Trained from scratch rather than fine-tuned to:
      (a) demonstrate the complete training pipeline,
      (b) keep the experiment within reasonable compute budget,
      (c) allow fair comparison with SimpleCNN baseline.
    
    Dropout is applied before the final classifier (Ch. 9 — Regularization).
    Weight decay is handled by the optimizer (not defined here).
    He initialization applied throughout (Ch. 7).
    """

    def __init__(self, num_classes=8, dropout_rate=0.5, dropout_enabled=True):
        super().__init__()
        self.dropout_enabled = dropout_enabled

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages (2 blocks each)
        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc      = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """He initialization for conv layers, Xavier for linear (Ch. 7)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        if self.dropout_enabled:
            x = self.dropout(x)
        return self.fc(x)


# ─── Factory function ─────────────────────────────────────────────────────────

def build_model(model_name, cfg):
    """
    Instantiates the requested model from the config.
    
    Args:
        model_name: one of 'logistic', 'simple_cnn', 'resnet18'
        cfg: parsed YAML config dict
    
    Returns:
        nn.Module
    """
    num_classes     = cfg["data"]["num_classes"]
    dropout_rate    = cfg["model"]["dropout_rate"]
    dropout_enabled = cfg["model"]["dropout_enabled"]
    image_size      = cfg["data"]["image_size"]

    if model_name == "logistic":
        return LogisticRegression(num_classes=num_classes, image_size=image_size)
    elif model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes,
                         dropout_rate=dropout_rate,
                         dropout_enabled=dropout_enabled)
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes,
                        dropout_rate=dropout_rate,
                        dropout_enabled=dropout_enabled)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from logistic, simple_cnn, resnet18.")


if __name__ == "__main__":
    # Smoke test: verify all models produce the correct output shape
    dummy = torch.randn(4, 3, 224, 224)
    for name in ["logistic", "simple_cnn", "resnet18"]:
        cfg = {
            "data": {"num_classes": 8, "image_size": 224},
            "model": {"dropout_rate": 0.5, "dropout_enabled": True}
        }
        model = build_model(name, cfg)
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:15s}  output: {tuple(out.shape)}  params: {params:,}")
