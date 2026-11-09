"""Albumentations transforms for the diagnosis models (M3, Drop 2).

Two pipelines:
  * classification (chest X-ray, EfficientNet-B0, 224x224)
  * segmentation   (MRI slice + mask, U-Net, 256x256) - geometric ops are applied
                   to image AND mask together so they stay aligned.

Design notes (for the viva):
  * No horizontal flip on chest X-rays: the heart is on the left, flipping creates
    anatomically impossible images. Brain MRI slices are roughly symmetric, so
    flipping is fine there.
  * Augmentations are mild (+-10 deg, +-10 % scale, small brightness/contrast) because
    large distortions change clinically relevant appearance.
  * Normalisation uses ImageNet mean/std because both encoders start from (or are
    compared against) ImageNet-pretrained weights.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLS_SIZE = 224
SEG_SIZE = 256


def classification_transforms(train: bool = True, size: int = CLS_SIZE) -> A.Compose:
    """Input: HxWx3 uint8 RGB. Output: dict(image=FloatTensor[3,size,size])."""
    ops = [A.Resize(size, size)]
    if train:
        ops += [
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(p=0.2),
        ]
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def segmentation_transforms(train: bool = True, size: int = SEG_SIZE) -> A.Compose:
    """Input: image HxWx3 uint8 RGB + mask HxW float/uint8 {0,1}.
    Output: dict(image=FloatTensor[3,size,size], mask=Tensor[size,size])."""
    ops = [A.Resize(size, size)]
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ]
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def denormalize(t):
    """Tensor[3,H,W] (normalised) -> HxWx3 float in [0,1]; for plotting / Grad-CAM overlays."""
    import numpy as np

    img = t.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return img.clip(0, 1)
