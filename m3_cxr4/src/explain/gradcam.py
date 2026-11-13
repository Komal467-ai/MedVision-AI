"""Grad-CAM for every architecture in the zoo, and a quantitative lung-focus audit.

Grad-CAM (Selvaraju et al., 2017): weights = spatial mean of d(class score)/d(feature maps);
map = ReLU(sum_k w_k * A_k), upsampled to the input size and scaled to [0, 1].
CNNs explain their last conv block; ViT/Swin token maps are reshaped to a grid first
(see zoo.Spec.to_maps), the usual way Grad-CAM is applied to transformers.

Lung-focus audit (uses the dataset's lung masks):
  lung_focus = share of total heat-map energy that falls inside the lungs   (1.0 = all inside)
  pointing   = 1 if the hottest pixel is inside the lungs, else 0           (pointing game)
  lung_area  = share of the image that is lung (the score of a heat-map spread uniformly)

    python -m src.explain.gradcam --runs runs/resnet50_group_none_s0 runs/vit_b_16_group_none_s0 \
        --data_root ... --n_audit 400 --out results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.cxrdata.augment import build_transforms
from src.cxrdata.dataset import CLASSES, SHORT, load_image


class GradCAM:
    def __init__(self, model):
        self.model = model

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        store = {}

        def hook(_m, _i, out):
            store["a"] = out
            out.register_hook(lambda g: store.__setitem__("g", g))

        h = self.model.cam_layer.register_forward_hook(hook)
        self.model.eval()
        try:
            with torch.enable_grad():
                logits = self.model(x)
                probs = torch.softmax(logits.float(), 1)[0].detach().cpu().numpy()
                c = int(logits.argmax(1)) if class_idx is None else class_idx
                self.model.zero_grad(set_to_none=True)
                logits[0, c].backward()
            a = self.model.to_maps(store["a"].detach())[0]
            g = self.model.to_maps(store["g"].detach())[0]
            cam = F.relu((g.mean(dim=(1, 2), keepdim=True) * a).sum(0))
            cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = cam.cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        finally:
            h.remove()
            self.model.zero_grad(set_to_none=True)
        return cam, c, probs


def lung_scores(cam: np.ndarray, mask: np.ndarray) -> dict:
    m = cv2.resize(mask, cam.shape[::-1], interpolation=cv2.INTER_NEAREST) > 127
    total = cam.sum() + 1e-8
    yx = np.unravel_index(cam.argmax(), cam.shape)
    return {"lung_focus": float(cam[m].sum() / total), "pointing": float(m[yx]), "lung_area": float(m.mean())}


def overlay(rgb01, cam, alpha=0.4):
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
    return np.uint8(255 * ((1 - alpha) * rgb01 + alpha * heat).clip(0, 1))


def audit(model, df_test: pd.DataFrame, n: int, size: int, dev: str, seed: int = 0) -> pd.DataFrame:
    tf = build_transforms(train=False, size=size)
    gc = GradCAM(model)
    sample = df_test[df_test["mask_path"].notna()]
    sample = sample.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(n // len(CLASSES), 1)), random_state=seed))
    rows = []
    for _, r in sample.iterrows():
        img = load_image(r["path"])
        x = tf(image=img)["image"][None].to(dev)
        cam, c, p = gc(x)
        mask = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
        rows.append({"path": r["path"], "y": int(r["y"]), "pred": c, "conf": float(p[c]), **lung_scores(cam, mask)})
    return pd.DataFrame(rows)


def gallery(models: dict, paths: list[tuple[str, int, str | None]], out_png: str, size: int, dev: str):
    """Rows = images, columns = original + one heat-map per model (lung outline drawn in white)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf = build_transforms(train=False, size=size)
    fig, axes = plt.subplots(len(paths), len(models) + 1, figsize=(2.6 * (len(models) + 1), 2.7 * len(paths)),
                             squeeze=False)
    for r, (p, y, mpath) in enumerate(paths):
        img = load_image(p)
        rgb = cv2.resize(img, (size, size)) / 255.0
        contour = None
        if mpath:
            m = cv2.resize(cv2.imread(mpath, cv2.IMREAD_GRAYSCALE), (size, size), interpolation=cv2.INTER_NEAREST)
            contour, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        axes[r, 0].imshow(rgb); axes[r, 0].set_title(f"true: {SHORT[y]}", fontsize=9)
        x = tf(image=img)["image"][None].to(dev)
        for c_, (name, mdl) in enumerate(models.items(), start=1):
            cam, c, pr = GradCAM(mdl)(x)
            ov = overlay(rgb, cam)
            if contour is not None:
                ov = cv2.drawContours(ov.copy(), contour, -1, (255, 255, 255), 1)
            axes[r, c_].imshow(ov)
            axes[r, c_].set_title(f"{name}\n{SHORT[c]} {pr[c]*100:.0f}%", fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle("Grad-CAM per model (white = lung boundary)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main(argv=None):
    from src.cxrdata.dataset import prepare_table
    from src.models.zoo import load_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--n_audit", type=int, default=400)
    ap.add_argument("--n_gallery", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--no_dedup", action="store_true")
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Path(a.out).mkdir(parents=True, exist_ok=True)

    models, size, summary_rows = {}, 224, []
    for r in map(Path, a.runs):
        m, ck = load_checkpoint(r / "best.pth", dev)
        cfg = json.loads((r / "config.json").read_text())
        size = ck.get("img_size", 224)
        df = prepare_table(a.data_root, cfg["split_mode"], seed=cfg["split_seed"],
                           cache_dir=str(Path(a.runs_dir) / "cache"), dedup=not a.no_dedup)
        test = df[df["split"] == "test"]
        res = audit(m, test, a.n_audit, size, dev)
        res.to_csv(r / "cam_audit.csv", index=False)
        s = {"run": r.name, "arch": ck["arch"], "n": len(res), "lung_focus_mean": round(float(res["lung_focus"].mean()), 4),
             "lung_focus_sd": round(float(res["lung_focus"].std()), 4), "pointing_game": round(float(res["pointing"].mean()), 4),
             "lung_area_mean": round(float(res["lung_area"].mean()), 4)}
        summary_rows.append(s)
        print(s)
        models[ck["arch"]] = m
    pd.DataFrame(summary_rows).to_csv(Path(a.out) / "cam_audit_summary.csv", index=False)

    picks = test[test["mask_path"].notna()].groupby("label", group_keys=False).apply(
        lambda g: g.sample(1, random_state=1)).head(a.n_gallery)
    gallery(models, [(r["path"], int(r["y"]), r["mask_path"]) for _, r in picks.iterrows()],
            str(Path(a.out) / "gradcam_gallery.png"), size, dev)
    print("saved", Path(a.out) / "gradcam_gallery.png")


if __name__ == "__main__":
    main()
