"""EfficientNet-B0 pneumonia classifier (M3, Drop 3).

Backbone : torchvision EfficientNet-B0, ImageNet weights (transfer learning)
Head     : Dropout(0.3) -> Linear(1280 -> 2)       classes = [NORMAL, PNEUMONIA]
Grad-CAM : target layer = model.features[-1]  (last conv block, 7x7x1280 for 224 input)

Checkpoint format (best_cnn.pth) - a plain dict so it loads anywhere:
    {"state_dict", "arch", "classes", "img_size", "epoch", "val_metrics"}
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

ARCH = "efficientnet_b0"
CLASSES = ["NORMAL", "PNEUMONIA"]


class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_f = net.classifier[1].in_features  # 1280
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        self.net = net

    # convenience handles used by Grad-CAM / feature extraction
    @property
    def features(self):
        return self.net.features

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.net.features[-1]

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def embed(self, x):
        """1280-d pooled feature vector (re-used by M4's CNN+LSTM phase recogniser)."""
        f = self.net.avgpool(self.net.features(x))
        return torch.flatten(f, 1)

    def freeze_backbone(self, freeze: bool = True):
        for p in self.net.features.parameters():
            p.requires_grad = not freeze


def build_classifier(num_classes: int = 2, pretrained: bool = True) -> PneumoniaClassifier:
    return PneumoniaClassifier(num_classes, pretrained)


def save_checkpoint(model, path, classes=CLASSES, img_size=224, epoch=None, val_metrics=None):
    torch.save({"state_dict": model.state_dict(), "arch": ARCH, "classes": list(classes),
                "img_size": img_size, "epoch": epoch, "val_metrics": val_metrics or {}}, path)


def load_classifier(path, device="cpu"):
    """Load best_cnn.pth WITHOUT downloading ImageNet weights (important inside Docker,
    where the container may have no internet). Returns (model.eval(), checkpoint_dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = PneumoniaClassifier(len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
