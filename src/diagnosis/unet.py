"""U-Net with skip connections (M3, Drop 4) - written from scratch in PyTorch.

Ronneberger et al., 2015.  Encoder-decoder:
    enc1 (32) -> enc2 (64) -> enc3 (128) -> enc4 (256) -> bottleneck (512)
    dec4 (256) <- dec3 (128) <- dec2 (64) <- dec1 (32) -> 1x1 conv -> 1 logit/pixel
Each decoder stage = up-sample x2, CONCATENATE the encoder feature map of the same
resolution (the skip connection), then two 3x3 conv-BN-ReLU blocks.

Differences from the 2015 paper (state these in the viva):
  * 'same' padding (padding=1) so output size == input size -> no cropping of skips
  * BatchNorm after every conv -> faster, more stable training
  * base width 32 instead of 64 -> ~7.8 M params, trains in minutes on a Kaggle T4
  * Dropout in the bottleneck for regularisation on small medical datasets
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout, dropout=0.0):
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class Down(nn.Sequential):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__(nn.MaxPool2d(2), DoubleConv(cin, cout, dropout))


class Up(nn.Module):
    def __init__(self, cin, cout, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                    nn.Conv2d(cin, cin // 2, 1))
        else:
            self.up = nn.ConvTranspose2d(cin, cin // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(cin, cout)  # cin = cin//2 (upsampled) + cin//2 (skip)

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dy or dx:  # odd input sizes
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))  # <-- skip connection


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32, bilinear=False, dropout=0.2):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]  # 32,64,128,256,512
        self.inc = DoubleConv(in_channels, c[0])
        self.down1, self.down2, self.down3 = Down(c[0], c[1]), Down(c[1], c[2]), Down(c[2], c[3])
        self.down4 = Down(c[3], c[4], dropout)  # bottleneck
        self.up1, self.up2 = Up(c[4], c[3], bilinear), Up(c[3], c[2], bilinear)
        self.up3, self.up4 = Up(c[2], c[1], bilinear), Up(c[1], c[0], bilinear)
        self.outc = nn.Conv2d(c[0], out_channels, 1)
        self.config = dict(in_channels=in_channels, out_channels=out_channels, base=base,
                           bilinear=bilinear, dropout=dropout)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)  # raw logits; apply sigmoid for probabilities


def save_unet(model: UNet, path, img_size=256, epoch=None, val_metrics=None):
    torch.save({"state_dict": model.state_dict(), "arch": "unet", "config": model.config,
                "img_size": img_size, "epoch": epoch, "val_metrics": val_metrics or {}}, path)


def load_unet(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = UNet(**ck.get("config", {}))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
