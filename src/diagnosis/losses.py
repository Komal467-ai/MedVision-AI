"""Segmentation losses + metrics (M3, Drop 4).

Dice + BCE:
  * BCE  - per-pixel, well-behaved gradients, but dominated by the (huge) background
  * Dice - overlap-based, insensitive to class imbalance (tumour is often < 5 % of pixels)
  Combining both (Milletari et al. 2016 V-Net Dice; common practice in BraTS) gives stable
  training AND good overlap.   loss = w_bce * BCE + w_dice * (1 - softDice)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Differentiable Dice per sample, using probabilities (not thresholded). Shape [B]."""
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)


class DiceLoss(nn.Module):
    def forward(self, logits, target):
        return 1 - soft_dice(logits, target).mean()


class DiceBCELoss(nn.Module):
    def __init__(self, w_bce: float = 0.5, w_dice: float = 0.5, pos_weight: float | None = None):
        super().__init__()
        self.w_bce, self.w_dice = w_bce, w_dice
        self.pos_weight = None if pos_weight is None else torch.tensor([pos_weight])

    def forward(self, logits, target):
        pw = None if self.pos_weight is None else self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        dice = 1 - soft_dice(logits, target).mean()
        return self.w_bce * bce + self.w_dice * dice


# ----------------------------------------------------------------------- hard metrics
@torch.no_grad()
def dice_iou_per_image(logits, target, thr: float = 0.5, eps: float = 1e-7):
    """Thresholded Dice and IoU per image -> (dice[B], iou[B]).
    Convention: if BOTH prediction and ground truth are empty, the score is 1 (correct 'no tumour')."""
    p = (torch.sigmoid(logits) > thr).float().flatten(1)
    t = (target > 0.5).float().flatten(1)
    inter = (p * t).sum(1)
    ps, ts = p.sum(1), t.sum(1)
    union = ps + ts - inter
    empty = (ps + ts) == 0
    dice = torch.where(empty, torch.ones_like(inter), 2 * inter / (ps + ts + eps))
    iou = torch.where(empty, torch.ones_like(inter), inter / (union + eps))
    return dice, iou


class SegMeter:
    """Accumulates per-image Dice/IoU plus a global (dataset-level) Dice."""

    def __init__(self, thr=0.5):
        self.thr, self.d, self.i = thr, [], []
        self.inter = self.psum = self.tsum = 0.0
        self.d_fg = []  # dice on images that actually contain foreground

    def update(self, logits, target):
        d, i = dice_iou_per_image(logits, target, self.thr)
        self.d += d.tolist(); self.i += i.tolist()
        has_fg = (target.flatten(1).sum(1) > 0)
        self.d_fg += d[has_fg].tolist()
        p = (torch.sigmoid(logits) > self.thr).float()
        t = (target > 0.5).float()
        self.inter += (p * t).sum().item(); self.psum += p.sum().item(); self.tsum += t.sum().item()

    def compute(self) -> dict:
        import numpy as np
        g_union = self.psum + self.tsum - self.inter
        return {
            "dice_mean": round(float(np.mean(self.d)), 4) if self.d else 0.0,
            "iou_mean": round(float(np.mean(self.i)), 4) if self.i else 0.0,
            "dice_tumour_slices": round(float(np.mean(self.d_fg)), 4) if self.d_fg else float("nan"),
            "dice_global": round(2 * self.inter / (self.psum + self.tsum + 1e-7), 4),
            "iou_global": round(self.inter / (g_union + 1e-7), 4),
            "n_images": len(self.d),
        }
