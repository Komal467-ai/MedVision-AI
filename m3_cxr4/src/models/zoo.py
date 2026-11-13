"""Model zoo: six ImageNet-pretrained backbones from three design families, one interface.

| key            | family              | year | params (4-class head) | Grad-CAM layer            |
|----------------|---------------------|------|-----------------------|---------------------------|
| resnet50       | residual CNN        | 2016 | ~23.5 M               | layer4                    |
| densenet121    | dense CNN (CheXNet) | 2017 | ~7.0 M                | features.denseblock4      |
| efficientnet_b0| scaled mobile CNN   | 2019 | ~4.0 M                | features[-1]              |
| convnext_tiny  | modernised CNN      | 2022 | ~27.8 M               | features[-1]              |
| vit_b_16       | vision transformer  | 2021 | ~85.8 M               | encoder.layers[-1].ln_1   |
| swin_t         | windowed transformer| 2021 | ~27.5 M               | features[-1]              |

Every model gets the same new head: Dropout(0.3) -> Linear(features -> 4).
Checkpoints store {state_dict, arch, classes, img_size, recipe, val_metrics} and are rebuilt
with pretrained=False when loaded (no internet needed at inference / in Docker).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torchvision import models as tvm

from src.cxrdata.dataset import CLASSES


@dataclass
class Spec:
    family: str
    lr: float            # default peak learning rate (transformers prefer smaller)
    build: Callable      # (pretrained) -> nn.Module with ImageNet head
    replace_head: Callable  # (net, n_classes) -> None
    cam_layer: Callable  # net -> module whose output is explained
    to_maps: Callable    # activation tensor -> [B, K, h, w]


def _linear(in_f, n):
    return nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, n))


def _w(name):
    return getattr(tvm, name).DEFAULT


def _nchw(a):
    return a


def _nhwc(a):  # Swin keeps channels last
    return a.permute(0, 3, 1, 2)


def _vit_tokens(a):  # [B, 1+N, C] -> [B, C, sqrt(N), sqrt(N)] (drop the class token)
    t = a[:, 1:, :]
    s = int(t.shape[1] ** 0.5)
    return t.reshape(t.shape[0], s, s, t.shape[2]).permute(0, 3, 1, 2)


def _set(attr_path):
    def f(net, n):
        obj = net
        parts = attr_path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
        last = parts[-1]
        old = obj[int(last)] if last.isdigit() else getattr(obj, last)
        new = _linear(old.in_features, n)
        if last.isdigit():
            obj[int(last)] = new
        else:
            setattr(obj, last, new)
    return f


SPECS: dict[str, Spec] = {
    "resnet50": Spec("CNN", 3e-4, lambda p: tvm.resnet50(weights=_w("ResNet50_Weights") if p else None),
                     _set("fc"), lambda n: n.layer4, _nchw),
    "densenet121": Spec("CNN", 3e-4, lambda p: tvm.densenet121(weights=_w("DenseNet121_Weights") if p else None),
                        _set("classifier"), lambda n: n.features.denseblock4, _nchw),
    "efficientnet_b0": Spec("CNN", 3e-4, lambda p: tvm.efficientnet_b0(weights=_w("EfficientNet_B0_Weights") if p else None),
                            _set("classifier.1"), lambda n: n.features[-1], _nchw),
    "convnext_tiny": Spec("CNN", 1e-4, lambda p: tvm.convnext_tiny(weights=_w("ConvNeXt_Tiny_Weights") if p else None),
                          _set("classifier.2"), lambda n: n.features[-1], _nchw),
    "vit_b_16": Spec("Transformer", 5e-5, lambda p: tvm.vit_b_16(weights=_w("ViT_B_16_Weights") if p else None),
                     _set("heads.head"), lambda n: n.encoder.layers[-1].ln_1, _vit_tokens),
    "swin_t": Spec("Transformer", 1e-4, lambda p: tvm.swin_t(weights=_w("Swin_T_Weights") if p else None),
                   _set("head"), lambda n: n.features[-1], _nhwc),
}
ARCHS = list(SPECS)


class Classifier(nn.Module):
    def __init__(self, arch: str, n_classes: int = len(CLASSES), pretrained: bool = True):
        super().__init__()
        if arch not in SPECS:
            raise ValueError(f"unknown arch {arch}; choose from {ARCHS}")
        self.arch, self.spec = arch, SPECS[arch]
        self.net = self.spec.build(pretrained)
        self.spec.replace_head(self.net, n_classes)

    def forward(self, x):
        return self.net(x)

    @property
    def cam_layer(self) -> nn.Module:
        return self.spec.cam_layer(self.net)

    def to_maps(self, act):
        return self.spec.to_maps(act)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def save_checkpoint(model: Classifier, path, img_size=224, recipe=None, val_metrics=None):
    torch.save({"state_dict": model.state_dict(), "arch": model.arch, "classes": CLASSES,
                "img_size": img_size, "recipe": recipe or {}, "val_metrics": val_metrics or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = Classifier(ck["arch"], len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck
