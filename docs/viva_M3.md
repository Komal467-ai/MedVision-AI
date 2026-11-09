# Mock viva — M3 (CNN + U-Net + Grad-CAM): likely questions & short answers

## Classification (EfficientNet-B0)
**Why EfficientNet-B0 and not ResNet/DenseNet?** Compound scaling gives ImageNet accuracy similar to much bigger ResNets with ~5.3 M parameters (≈4 M with our 2-class head) → trains in ~30 min on a free Kaggle GPU and runs on CPU in Docker.

**What is transfer learning here?** Start from ImageNet weights (edges, textures, shapes already learned), replace the 1000-class layer with a 2-class head, fine-tune all layers with a small LR (3e-4).

**Why CrossEntropy with class weights?** Data is ~3:1 PNEUMONIA:NORMAL. Weights w_c = N/(2·N_c) make a NORMAL mistake cost ~3× more, so the model doesn't just predict "pneumonia" for everything.

**Why AdamW?** Adam with *decoupled* weight decay — regularisation works properly, standard for fine-tuning.

**Why choose the best epoch by validation AUROC, not accuracy?** AUROC doesn't depend on the 0.5 threshold and is less noisy on a small validation set.

**Why not use the official Kaggle val folder?** Only 16 images — one wrong prediction = 6 % accuracy change. We re-split 70/15/15 **by patient**.

**What is data leakage and how did you prevent it?** Same patient's images in train and test → model "recognises the patient", inflating accuracy. Patient ID is parsed from file names, whole patient goes to one split, `assert_no_patient_leak` fails the run otherwise.

**Which metric matters most clinically?** Recall/sensitivity — a missed pneumonia (FN) is worse than a false alarm (FP). We report specificity alongside so the trade-off is visible.

**Why no horizontal flip?** Heart is on the left; flipped X-rays are anatomically impossible.

**Why CLAHE?** Local contrast equalisation makes lung opacities more visible; applied identically at training and inference.

## Grad-CAM
**How does it work?** Take gradients of the class score w.r.t. the last conv feature maps (7×7×1280), average them spatially → one weight per channel, weighted sum of the maps, ReLU, upsample to 224×224, overlay.

**Why ReLU?** Keep only regions that *increase* the class score.

**Why the last conv layer?** Best trade-off: highest-level semantics while still keeping spatial layout.

**Limitations?** Coarse (7×7), can highlight wrong regions; not a validated lesion localiser — we use it to check the model looks at the lungs, not at text/markers.

## U-Net
**What are skip connections for?** Pooling loses fine detail; concatenating the encoder map of the same resolution gives the decoder precise boundary information.

**Why Dice + BCE?** BCE: stable per-pixel gradients but dominated by background. Dice: measures overlap, robust to imbalance (tumour ≈ <5 % of pixels). Together: stable and overlap-focused.

**Dice vs IoU?** Dice = 2|P∩G|/(|P|+|G|), IoU = |P∩G|/|P∪G|; IoU = Dice/(2−Dice), so IoU is always ≤ Dice.

**What if a slice has no tumour?** If prediction is also empty → score 1. Because that can inflate averages, we also report Dice on tumour slices only and global Dice.

**What did you change from the original U-Net?** 'Same' padding (no cropping), BatchNorm, base width 32 (7.8 M params), bottleneck dropout.

**2D vs 3D?** 2D slices: far less GPU memory, more training samples; loses inter-slice context (3D U-Net/nnU-Net would be the next step).

## Deployment / integration
**How does the API call your models?** `predict_classification(image)` and `predict_segmentation(image)` in `src/diagnosis/inference.py` — accept path / bytes / array, return JSON with base64 PNGs.

**Model loading problems in Docker and fixes?** No internet → build networks with `pretrained=False` before loading weights; no GPU → `map_location` CPU; weights path from `MODEL_DIR` env (volume mount); load once + cache + lock for threads.

## Limitations to state honestly
Single-centre paediatric dataset (generalisation to adults/other hospitals unknown), binary labels only, 2D segmentation, Grad-CAM is qualitative, research prototype — not a medical device.
