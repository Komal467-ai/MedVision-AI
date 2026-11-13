# MedVision-CXR4 — 4-class chest X-ray diagnosis: CNNs vs Vision Transformers vs Ensembles

Research codebase for the X-ray part of MedVision-AI. Brain-tumour segmentation and surgical-video
code are **not** part of this study.

**Task:** classify a chest X-ray as **COVID-19, Lung Opacity, Normal or Viral Pneumonia**.
**Data:** COVID-19 Radiography Database, Kaggle `tawsifurrahman/covid19-radiography-database`
(21,165 images with lung masks; Chowdhury et al., IEEE Access 2020; Rahman et al., Comput. Biol. Med. 2021).
**Models (all ImageNet-pretrained, torchvision):** ResNet-50, DenseNet-121, EfficientNet-B0,
ConvNeXt-Tiny, ViT-B/16, Swin-T, and soft-voting ensembles.

## Experiments (see the research blueprint doc)
| ID | Question | How |
|---|---|---|
| E1 | Which architecture family works best? | 6 models x 3 seeds, duplicate-aware split |
| E2 | Does mixing models help? | all-6 and greedy (validation-selected) soft-voting ensembles |
| E3 | How much does leakage inflate scores? | `--split_mode image` vs duplicate-aware `group` |
| E4 | Do explanations look at the lungs? | Grad-CAM lung-focus ratio + pointing game with dataset masks |
| E5 | Shortcut learning? | `--mask_mode lungs_only / lungs_removed / lung_crop` |
| E6 | Are probabilities trustworthy? | ECE before/after temperature scaling; selective prediction |
| E7 | Which pipeline choices matter? | `--no_clahe / --no_augment / --no_class_weights / --no_pretrained` |

## Layout
```
src/cxrdata/dataset.py        index, near-duplicate groups (MD5 + dHash incl. mirrored), stratified split, mask modes
src/cxrdata/augment.py        identical transforms for every model
src/models/zoo.py          the six backbones, one interface, Grad-CAM layer per model
src/train.py               one run -> runs/<name>/{best.pth, metrics.json, preds_val/test.csv, history.json}
src/research/metrics.py    accuracy, balanced acc., macro-F1/AUROC, per-class sens/spec, ECE, temperature scaling, bootstrap CI
src/research/ensemble.py   soft-voting ensembles (all / greedy on validation)
src/research/aggregate.py  mean +/- SD over seeds, 95% CIs, McNemar with Holm correction -> results/table_*.csv/.md
src/research/plots.py      all paper figures -> results/fig_*.png
src/explain/gradcam.py     Grad-CAM for CNNs and transformers + lung-focus audit + gallery
src/inference.py           predict_xray() for the API (single model or ensemble, with Grad-CAM)
src/run_study.py           runs a whole stage (smoke | main | audits | analysis) with one command
notebooks/kaggle_run_study.ipynb   notebook front-end for run_study (Kaggle GPU, or DEMO=True on CPU)
tests/                     20 offline tests on synthetic data (pytest -q)
```

## Quick start
```bash
pip install -r requirements.txt
pytest -q                                                    # 20 tests, CPU, offline
python -m src.train --data_root <kaggle path> --arch swin_t --seed 0
python -m src.research.ensemble --pattern "*_group_none_s0" --greedy --name greedy_s0
python -m src.research.aggregate && python -m src.research.plots
```
## Running the full study (Kaggle, free T4 GPU)
Open `notebooks/kaggle_run_study.ipynb` on Kaggle and set GPU T4 x2 and Internet On. Run it once per stage with
`STAGE = "main"`, then `"audits"` (these two can run at the same time), then `"analysis"` with both earlier versions attached as inputs.
To check that the code works without a GPU or the dataset, set `DEMO = True` (synthetic data, CPU, a few minutes).

For the API: copy one or more `best.pth` files into `models/` and call `src.inference.predict_xray(image_bytes)`.

Research prototype - not a medical device.
