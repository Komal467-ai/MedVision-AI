# Report — M3 sections (Diagnosis): Section 2 Related Work · Section 3 Methodology

> Written by M3 (ML Engineer — Diagnosis) for `docs/report.pdf` (compiled by M1).
> Items in **[square brackets]** are filled from the Kaggle run (`results/*.json`) — do not invent numbers.
> A Section 4 (Results — diagnosis) template is included at the end so M1 can drop it in next to M4's surgery results.

---

## 2. Related Work (diagnosis)

### 2.1 Deep learning for chest radiograph classification
Large public chest X-ray collections made deep learning practical for radiograph analysis. Wang et al. (2017) released ChestX-ray8/14 with more than 100,000 frontal radiographs and text-mined labels for 14 thoracic findings. Rajpurkar et al. (2017) trained a 121-layer DenseNet (CheXNet) on this data and reported pneumonia detection comparable to practising radiologists on their test set. Kermany et al. (2018) introduced the paediatric chest X-ray pneumonia dataset used in this project (5,863 images labelled NORMAL or PNEUMONIA) and showed that an ImageNet-pretrained CNN, fine-tuned by transfer learning, reaches roughly 93 % accuracy. These works establish two points we build on: (i) ImageNet transfer learning is effective even though radiographs look very different from natural images, and (ii) a binary pneumonia task is a realistic, well-benchmarked starting point for a student-scale diagnosis module.

### 2.2 Efficient CNN backbones
Earlier medical work typically used VGG, ResNet or DenseNet backbones with tens of millions of parameters. Tan and Le (2019) proposed EfficientNet, which scales network depth, width and input resolution together with a single *compound coefficient*; the base model, EfficientNet-B0, is built from mobile inverted-bottleneck (MBConv) blocks with squeeze-and-excitation and has about 5.3 M parameters while matching much larger ResNets on ImageNet. Its small size suits our constraints (free Kaggle GPU, ~30 min training budget, CPU inference inside Docker), which is why we chose B0 over DenseNet-121.

### 2.3 Explainability: CAM and Grad-CAM
Clinicians need to see *why* a model predicts a disease. Class Activation Mapping (CAM; Zhou et al., 2016) highlights discriminative regions but requires a specific global-average-pooling architecture. Grad-CAM (Selvaraju et al., 2017) generalises CAM to any CNN by weighting the last convolutional feature maps with the spatially averaged gradients of the class score, producing a coarse localisation map without retraining. Variants such as Grad-CAM++ (Chattopadhay et al., 2018) refine localisation for multiple objects. Studies that benchmarked saliency maps against radiologist annotations (e.g. Saporta et al., 2022) found they often localise pathology worse than experts, so in this project Grad-CAM is used as a **sanity check** that the model looks at the lung fields (and not at text markers or borders), not as a lesion detector.

### 2.4 Medical image segmentation
Fully Convolutional Networks (Long et al., 2015) made dense, pixel-wise prediction possible. U-Net (Ronneberger et al., 2015) added a symmetric expanding path and **skip connections** that concatenate high-resolution encoder features with up-sampled decoder features, recovering fine boundaries from very few annotated images; it remains the dominant baseline in medical segmentation. Extensions include 3D U-Net (Çiçek et al., 2016), V-Net with a Dice-based loss for class-imbalanced volumes (Milletari et al., 2016), Attention U-Net with gating on skip connections (Oktay et al., 2018), and nnU-Net (Isensee et al., 2021), which automatically configures U-Net pipelines and wins many challenges. For brain tumours, the BraTS benchmark (Menze et al., 2015) provides multi-modal MRI with expert masks; Buda et al. (2019) released a smaller FLAIR lower-grade-glioma dataset (110 patients) that we use as a backup when BraTS preparation is not ready.

### 2.5 Evaluation pitfalls: leakage and shortcut learning
Zech et al. (2018) showed that CNNs trained on chest X-rays can exploit hospital-specific confounders rather than pathology, and that results drop on external data. A related, very common error is splitting by *image* instead of by *patient*, which lets near-identical images of the same person appear in both training and test sets and inflates accuracy. Following these lessons, all our splits are **patient-level**, threshold tuning uses only the validation set, and Grad-CAM is inspected for shortcut behaviour.

### 2.6 Gap addressed by this project
Most published systems treat diagnosis models in isolation. Our contribution is engineering-oriented: a lightweight, explainable diagnosis module (EfficientNet-B0 + Grad-CAM, U-Net) packaged behind a stable inference interface that the FastAPI/Streamlit system and the surgical-assistance module share.

---

## 3. Methodology (diagnosis parts)

### 3.1 Datasets
| Task | Dataset | Size | Labels |
|---|---|---|---|
| Classification | Chest X-Ray Images (Pneumonia), Kermany et al. 2018 (Kaggle) | 5,863 JPEG images, paediatric, anterior-posterior | NORMAL / PNEUMONIA (≈ 1 : 2.9) |
| Segmentation (primary) | BraTS slices prepared by M2 | [n slices / n patients] | binary whole-tumour mask |
| Segmentation (backup) | LGG MRI Segmentation, Buda et al. 2019 (Kaggle) | 3,929 FLAIR slices, 110 patients | binary tumour mask |

### 3.2 Pre-processing
1. **Grayscale loading and CLAHE** (clip limit 2.0, 8×8 tiles) to equalise local contrast in radiographs — identical to M2's `image_preprocess.py` and re-applied at inference so training and deployment see the same input distribution.
2. Replication to **3 channels** so ImageNet-pretrained weights can be used.
3. **Resize** to 224×224 (classifier) or 256×256 (U-Net).
4. **Normalisation** with ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).

### 3.3 Patient-level data split
The official Kaggle split has a 16-image validation folder, which is too small to select models. All images were pooled and re-split **70 / 15 / 15 by patient**: the patient identifier is parsed from the file name (e.g. `person1_bacteria_1.jpeg` → `person1`, `IM-0115-0001.jpeg` → `IM-0115`), patients are shuffled with a fixed seed separately per class (stratification), and every image of a patient goes to the same split. An automatic check (`assert_no_patient_leak`) fails the run if any patient appears in two splits. Final counts: **[train / val / test images and patients]**. The same procedure (by case / patient folder) is used for segmentation.

### 3.4 Data augmentation (training only, Albumentations)
| Classifier | U-Net (applied jointly to image and mask) |
|---|---|
| Affine: rotation ±10°, scale 0.9–1.1, shift ±5 % (p = 0.7) | Horizontal flip (p = 0.5) |
| Brightness/contrast ±15 % (p = 0.5) | Affine: rotation ±15°, scale 0.9–1.1, shift ±5 % |
| Gaussian noise (p = 0.2) | Brightness/contrast ±10 % |

Horizontal flips are deliberately **not** used for chest X-rays because they create anatomically impossible images (heart on the right).

### 3.5 Classification model — EfficientNet-B0
* **Backbone:** torchvision EfficientNet-B0 initialised with ImageNet-1k weights; all layers fine-tuned.
* **Head:** the 1000-class layer is replaced by Dropout(0.3) → Linear(1280 → 2).
* **Parameters:** ≈ 4.0 M (all trainable).
* **Loss:** class-weighted cross-entropy, with inverse-frequency weights computed on the training split,
  w_c = N / (2 · N_c), to counter the ≈ 3 : 1 PNEUMONIA : NORMAL imbalance.
* **Optimiser:** AdamW (lr 3×10⁻⁴, weight decay 10⁻⁴) with a one-cycle schedule (15 % warm-up, cosine decay).
* **Training:** 5 epochs, batch 32, mixed precision (FP16) on a Kaggle [T4/P100] GPU, seed 42.
* **Model selection:** checkpoint with the highest **validation AUROC** (threshold-independent, stable on small validation sets) is saved as `best_cnn.pth`.
* **Decision threshold:** 0.5 by default; optionally the threshold maximising Youden's J (sensitivity + specificity − 1) on the **validation** set. The test set is never used for any choice.

### 3.6 Evaluation metrics (classification)
With PNEUMONIA as the positive class and TP, FP, TN, FN from the confusion matrix:
* Accuracy = (TP + TN) / N
* Precision = TP / (TP + FP)
* Recall (sensitivity) = TP / (TP + FN) — the most important metric clinically, a missed pneumonia is costlier than a false alarm
* Specificity = TN / (TN + FP)
* F1 = 2 · Precision · Recall / (Precision + Recall)
* AUROC — area under the ROC curve, threshold-independent ranking quality.

### 3.7 Explainability — Grad-CAM
For the predicted class c and the feature maps Aᵏ of the last convolutional block (`features[-1]`, 7×7×1280 for a 224×224 input):

  αₖᶜ = (1/Z) Σᵢⱼ ∂yᶜ/∂Aᵏᵢⱼ   and   L_Grad-CAMᶜ = ReLU( Σₖ αₖᶜ Aᵏ )

The 7×7 map is bilinearly up-sampled to 224×224, min-max normalised and blended (α = 0.4, JET colour map) with the input. It is implemented with forward hooks and a tensor gradient hook that are removed after every call, so it is safe to call from the multi-threaded API. Grad-CAM always explains the class that is shown to the user.

### 3.8 Segmentation model — U-Net
* **Architecture:** encoder of four down-sampling stages (2× [3×3 conv → BatchNorm → ReLU] + 2×2 max-pool) with 32-64-128-256 channels, a 512-channel bottleneck with Dropout2d(0.2), and a symmetric decoder using 2×2 transposed convolutions. At each decoder stage the up-sampled map is **concatenated** with the encoder map of the same resolution (skip connection) before two 3×3 conv blocks. A final 1×1 convolution gives one logit per pixel. ≈ 7.8 M parameters.
* **Changes from Ronneberger et al.:** 'same' padding (output size = input size, no cropping), batch normalisation, half the channel width, bottleneck dropout.
* **Loss — Dice + BCE:** L = 0.5 · BCE(ŷ, y) + 0.5 · (1 − softDice), with softDice = (2 Σ p·y + 1) / (Σ p + Σ y + 1) computed on sigmoid probabilities. BCE gives smooth per-pixel gradients; the Dice term directly optimises overlap and is insensitive to the heavy background imbalance (tumour is usually < 5 % of a slice).
* **Training:** AdamW (lr 10⁻³, weight decay 10⁻⁴), cosine annealing, 25 epochs, batch 16, 256×256, mixed precision; best checkpoint by validation Dice → `best_unet.pth`.

### 3.9 Evaluation metrics (segmentation)
Prediction mask P = 1[σ(logit) > 0.5], ground truth G:
* Dice = 2|P ∩ G| / (|P| + |G|)
* IoU (Jaccard) = |P ∩ G| / |P ∪ G|  (IoU = Dice / (2 − Dice))

Both are computed per slice and averaged. A slice where both P and G are empty scores 1 (correctly reports "no tumour"). Because this convention inflates the average on datasets with many empty slices, we additionally report **Dice on tumour-containing slices only** and a **global Dice** over all pixels of the test set.

### 3.10 Integration for deployment
`src/diagnosis/inference.py` exposes two functions used by the FastAPI service (M5): `predict_classification(image)` → label, confidence, class probabilities and a Grad-CAM overlay (base64 PNG); `predict_segmentation(image)` → binary mask, overlay, tumour area %, bounding box and number of regions. Models are loaded once, cached, and guarded by locks; weights are read from the `MODEL_DIR` directory, networks are constructed without downloading ImageNet weights, and checkpoints are mapped to CPU so the same code runs in a GPU-less Docker container. Every response carries a disclaimer that the system is a research prototype for decision support only.

### 3.11 Support for the surgical module (M4)
For phase recognition, `src/shared/cnn_features.py` converts each surgical video (1 fps) into a sequence of 1280-d EfficientNet-B0 frame embeddings, and `src/shared/phase_lstm.py` provides a two-layer uni-directional LSTM (hidden 256) that outputs one of the 7 Cholec80 phases per frame. Features are pre-computed once so that the temporal model trains quickly.

---

## 4. Results — diagnosis (template for the Results section)

**Table 4.x — Pneumonia classification, held-out test patients (n = [ ])**
| Model | Accuracy | Precision | Recall | Specificity | F1 | AUROC |
|---|---|---|---|---|---|---|
| EfficientNet-B0 (thr 0.5) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EfficientNet-B0 (val-tuned thr = [ ]) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

Figures: `training_curves.png`, `confusion_matrix.png`, `roc_curve.png`, `gradcam_samples.png`.

**Table 4.y — Tumour segmentation, held-out test patients (n = [ ] slices)**
| Model | Dice (all slices) | Dice (tumour slices) | IoU | Global Dice |
|---|---|---|---|---|
| U-Net (32 base, Dice+BCE) | [ ] | [ ] | [ ] | [ ] |

Figures: `unet_training_curves.png`, `segmentation_overlay.png`.

Targets set in the plan: accuracy ≥ 0.85, Dice ≥ 0.80 → **[met / not met]**.

---

## References (diagnosis)
1. Wang X. et al. ChestX-ray8: Hospital-scale chest X-ray database and benchmarks. *CVPR* 2017.
2. Rajpurkar P. et al. CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv:1711.05225*, 2017.
3. Kermany D. S. et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell* 172(5), 2018.
4. Tan M., Le Q. V. EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML* 2019.
5. Zhou B. et al. Learning deep features for discriminative localization. *CVPR* 2016.
6. Selvaraju R. R. et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV* 2017.
7. Chattopadhay A. et al. Grad-CAM++: Generalized gradient-based visual explanations. *WACV* 2018.
8. Saporta A. et al. Benchmarking saliency methods for chest X-ray interpretation. *Nature Machine Intelligence* 4, 2022.
9. Long J., Shelhamer E., Darrell T. Fully convolutional networks for semantic segmentation. *CVPR* 2015.
10. Ronneberger O., Fischer P., Brox T. U-Net: Convolutional networks for biomedical image segmentation. *MICCAI* 2015.
11. Çiçek Ö. et al. 3D U-Net: Learning dense volumetric segmentation from sparse annotation. *MICCAI* 2016.
12. Milletari F., Navab N., Ahmadi S.-A. V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3DV* 2016.
13. Oktay O. et al. Attention U-Net: Learning where to look for the pancreas. *MIDL* 2018.
14. Isensee F. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods* 18, 2021.
15. Menze B. H. et al. The multimodal brain tumor image segmentation benchmark (BRATS). *IEEE TMI* 34(10), 2015.
16. Buda M., Saha A., Mazurowski M. A. Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm. *Computers in Biology and Medicine* 109, 2019.
17. Zech J. R. et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. *PLOS Medicine* 15(11), 2018.
