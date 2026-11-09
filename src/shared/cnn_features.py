"""CNN feature extraction for M4's phase recogniser (M3 support, Drop 5).

Phase recognition = CNN (per-frame spatial features) + LSTM (temporal context).
This module turns a surgical video (or a folder of extracted frames from M2's
src/surgery/extract_frames.py) into a [T, 1280] feature matrix using EfficientNet-B0.

    python -m src.shared.cnn_features --video data/cholec80/videos/video01.mp4 --fps 1 \
        --out data/features/video01.npy
    python -m src.shared.cnn_features --frames_dir data/frames/video01 --out data/features/video01.npy

Pre-computing features once means the LSTM trains in minutes on CPU/GPU instead of
re-running the CNN every epoch (standard 2-stage approach, e.g. EndoNet / SV-RCNet style).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from src.preprocessing.augment import classification_transforms


class FrameEncoder(nn.Module):
    """EfficientNet-B0 trunk -> global average pool -> 1280-d vector per frame."""

    out_dim = 1280

    def __init__(self, pretrained: bool = True, ckpt: str | None = None):
        super().__init__()
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained and not ckpt else None)
        if ckpt:  # optionally start from a fine-tuned backbone (e.g. a tool-presence CNN)
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            sd = {k.replace("net.", "", 1): v for k, v in sd.items() if "classifier" not in k}
            net.load_state_dict(sd, strict=False)
        self.features, self.pool = net.features, net.avgpool

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


def iter_video_frames(video_path: str, fps: float = 1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(int(round(src_fps / fps)), 1)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def iter_frame_dir(frames_dir: str):
    for p in sorted(Path(frames_dir).glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


@torch.no_grad()
def extract_features(frames, encoder: FrameEncoder | None = None, batch_size: int = 64,
                     device: str | None = None) -> np.ndarray:
    """frames: iterable of HxWx3 RGB uint8. Returns float32 array [T, 1280]."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = (encoder or FrameEncoder()).to(device).eval()
    tf = classification_transforms(train=False, size=224)
    feats, batch = [], []
    for fr in frames:
        batch.append(tf(image=fr)["image"])
        if len(batch) == batch_size:
            feats.append(encoder(torch.stack(batch).to(device)).cpu()); batch = []
    if batch:
        feats.append(encoder(torch.stack(batch).to(device)).cpu())
    return torch.cat(feats).numpy().astype(np.float32) if feats else np.zeros((0, FrameEncoder.out_dim), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--frames_dir")
    ap.add_argument("--fps", type=float, default=1.0, help="Cholec80 phase labels are given at 1 fps")
    ap.add_argument("--ckpt", help="optional fine-tuned backbone checkpoint")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frames = iter_video_frames(a.video, a.fps) if a.video else iter_frame_dir(a.frames_dir)
    f = extract_features(frames, FrameEncoder(ckpt=a.ckpt), a.bs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, f)
    print(f"saved {f.shape} -> {a.out}")


if __name__ == "__main__":
    main()
