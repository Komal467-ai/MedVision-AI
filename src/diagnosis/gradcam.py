"""Grad-CAM for the EfficientNet-B0 classifier (M3, Drop 3).

Selvaraju et al., 2017.  For a target class c and the last conv feature maps A^k:
    alpha_k = mean_{i,j}( d y^c / d A^k_ij )          (global-average-pooled gradients)
    L^c     = ReLU( sum_k alpha_k * A^k )              (coarse 7x7 map, upsampled to 224x224)
Implemented directly with forward/backward hooks (no extra library), hooks are removed
after every call so it is safe to use inside the API.

    python -m src.diagnosis.gradcam --ckpt models/best_cnn.pth --data_root <...> --out results/gradcam_samples.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model, self.layer = model, target_layer

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: [1,3,H,W] normalised. Returns (cam HxW float in [0,1], class_idx, probs np[C])."""
        acts, grads = {}, {}

        def fwd_hook(module, inp, out):
            acts["a"] = out
            out.register_hook(lambda g: grads.__setitem__("g", g))  # gradient w.r.t. feature maps

        h1 = self.layer.register_forward_hook(fwd_hook)
        was_training = self.model.training
        self.model.eval()
        try:
            x = x.clone().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
                if class_idx is None:
                    class_idx = int(logits.argmax(1).item())
                self.model.zero_grad(set_to_none=True)
                logits[0, class_idx].backward()
            a, g = acts["a"].detach()[0], grads["g"].detach()[0]       # [K,h,w]
            weights = g.mean(dim=(1, 2), keepdim=True)                  # alpha_k
            cam = F.relu((weights * a).sum(0))                          # [h,w]
            cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = cam.cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        finally:
            h1.remove()
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)
        return cam, class_idx, probs


def overlay_cam(rgb_01: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """rgb_01: HxWx3 float [0,1]; cam: HxW [0,1]. Returns HxWx3 uint8 RGB overlay."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    out = (1 - alpha) * rgb_01.astype(np.float32) + alpha * heat
    return np.uint8(255 * out.clip(0, 1))


def gradcam_figure(model, samples, out_path, classes=("NORMAL", "PNEUMONIA"), device="cpu"):
    """samples: list of (image_path, true_label_idx). Saves a 3-column figure per sample:
    original | heat-map | overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.dataset import load_xray
    from src.preprocessing.augment import classification_transforms

    tf = classification_transforms(train=False)
    gc = GradCAM(model, model.gradcam_layer)
    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3.4 * len(samples)), squeeze=False)
    for r, (path, y) in enumerate(samples):
        img = load_xray(path)
        x = tf(image=img)["image"].unsqueeze(0).to(device)
        cam, c, p = gc(x)
        rgb = cv2.resize(img, (x.shape[-1], x.shape[-2])) / 255.0
        axes[r, 0].imshow(rgb); axes[r, 0].set_title(f"{Path(path).name}\ntrue: {classes[y]}", fontsize=9)
        axes[r, 1].imshow(cam, cmap="jet"); axes[r, 1].set_title("Grad-CAM heat-map", fontsize=9)
        axes[r, 2].imshow(overlay_cam(rgb, cam))
        axes[r, 2].set_title(f"pred: {classes[c]} ({p[c]*100:.1f}%)", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle("Grad-CAM - regions driving the EfficientNet-B0 prediction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    from src.data.dataset import CLASSES, load_cxr_table
    from src.diagnosis.classifier import load_classifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/best_cnn.pth")
    ap.add_argument("--data_root")
    ap.add_argument("--csv")
    ap.add_argument("--images", nargs="*", help="explicit image paths (label inferred from folder name)")
    ap.add_argument("--n_per_class", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gradcam_samples.png")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_classifier(a.ckpt, dev)
    if a.images:
        samples = [(p, CLASSES.index(Path(p).parent.name.upper()) if Path(p).parent.name.upper() in CLASSES else 0)
                   for p in a.images]
    else:  # one test image per class (2 samples, as required)
        df = load_cxr_table(a.data_root, a.csv)
        test = df[df["split"] == "test"]
        samples = []
        for ci, cname in enumerate(CLASSES):
            for p in test[test["label"] == cname].sample(a.n_per_class, random_state=a.seed)["path"]:
                samples.append((p, ci))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    gradcam_figure(model, samples, a.out, device=dev)
    print("saved", a.out)


if __name__ == "__main__":
    main()
