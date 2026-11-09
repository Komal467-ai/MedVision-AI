"""Reference LSTM phase recogniser over CNN features (M3 support for M4, Drop 6).

M4 owns src/surgery/phase_recognizer.py - this is a tested building block to import or
copy from. Input: per-video feature matrices from src.shared.cnn_features ([T,1280]) and
per-frame phase labels (Cholec80: 7 phases, 1 fps).

Cholec80 phases:
 0 Preparation  1 CalotTriangleDissection  2 ClippingCutting  3 GallbladderDissection
 4 GallbladderPackaging  5 CleaningCoagulation  6 GallbladderRetraction

Training uses fixed-length windows (e.g. 64 s) cut from each video; at inference the
causal (uni-directional) LSTM is run over the whole video, so it can work online.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

CHOLEC80_PHASES = ["Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
                   "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]


class PhaseLSTM(nn.Module):
    def __init__(self, in_dim=1280, hidden=256, layers=2, n_classes=7, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)  # causal: bidirectional=False
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, feats, state=None):
        """feats [B,T,in_dim] -> logits [B,T,n_classes] (one phase prediction per frame)."""
        h, state = self.lstm(self.proj(feats), state)
        return self.head(h), state


class FeatureWindows(Dataset):
    """videos: list of (features np[T,D], labels np[T]) -> random windows of length `win`."""

    def __init__(self, videos, win=64, stride=32):
        self.items = []
        for vi, (f, y) in enumerate(videos):
            T = len(f)
            starts = range(0, max(T - win, 0) + 1, stride) if T >= win else [0]
            self.items += [(vi, s) for s in starts]
        self.videos, self.win = videos, win

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vi, s = self.items[i]
        f, y = self.videos[vi]
        f, y = f[s:s + self.win], y[s:s + self.win]
        pad = self.win - len(f)
        if pad:  # pad short videos; label -100 is ignored by CrossEntropyLoss
            f = np.concatenate([f, np.zeros((pad, f.shape[1]), f.dtype)])
            y = np.concatenate([y, np.full(pad, -100)])
        return torch.from_numpy(f).float(), torch.from_numpy(y).long()


def train_epoch(model, loader, opt, device, class_weights=None):
    model.train()
    crit = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    tot = n = 0
    for f, y in loader:
        f, y = f.to(device), y.to(device)
        opt.zero_grad()
        logits, _ = model(f)
        loss = crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(f); n += len(f)
    return tot / max(n, 1)


@torch.no_grad()
def predict_video(model, feats: np.ndarray, device="cpu", smooth: int = 5) -> np.ndarray:
    """Whole-video prediction [T] with optional majority-vote smoothing (removes 1-2 s flicker)."""
    model.eval()
    logits, _ = model(torch.from_numpy(feats).float()[None].to(device))
    pred = logits[0].argmax(-1).cpu().numpy()
    if smooth > 1:
        k = smooth // 2
        pred = np.array([np.bincount(pred[max(0, t - k):t + k + 1]).argmax() for t in range(len(pred))])
    return pred
