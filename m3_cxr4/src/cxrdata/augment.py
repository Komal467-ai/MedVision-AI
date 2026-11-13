"""Albumentations transforms shared by every model (so architectures are compared fairly).

All backbones are ImageNet-pretrained, so inputs are resized to 224x224 and normalised with
ImageNet statistics. Training-only augmentation is mild and anatomically plausible:
no horizontal flips (they move the heart to the wrong side), small rotations/shifts only.
"""
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool = True, size: int = 224) -> A.Compose:
    ops = [A.Resize(size, size)]
    if train:
        ops += [
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(p=0.2),
        ]
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def denormalize(t) -> np.ndarray:
    img = t.detach().cpu().numpy().transpose(1, 2, 0)
    return (img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)
