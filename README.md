# EE6483 Project: Cats vs Dogs and CIFAR-10

This repository contains a reproducible PyTorch pipeline for the EE6483 mini project. It covers:

- binary classification for the Kaggle Dogs vs Cats dataset
- prediction and submission export for the test split
- validation error analysis utilities
- CIFAR-10 baseline training
- CIFAR-10 class imbalance experiments with multiple mitigation strategies

Datasets and report deliverables are intentionally excluded from version control. Selected trained checkpoints are included through Git LFS.

## Repository Layout

```text
cat_vs_dog/
├── scripts/
│   ├── train_dogs_vs_cats.py
│   ├── predict_dogs_vs_cats.py
│   ├── analyze_dogs_vs_cats.py
│   ├── train_cifar10.py
│   └── run_project_pipeline.py
├── src/
│   ├── common.py
│   ├── datasets.py
│   ├── models.py
│   └── reporting.py
├── runs/
│   └── project_pipeline/
│       └── selected best checkpoints tracked with Git LFS
├── requirements.txt
└── README.md
```

## Environment

- Python 3.10 or newer is recommended
- PyTorch and torchvision are required
- CUDA is optional; the scripts accept `--device auto` and prefer GPU when available

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

This repository does not store the datasets.

- Dogs vs Cats should be available under `datasets/`
- CIFAR-10 is downloaded automatically to `data/cifar10/` when needed

## Main Scripts

Train Dogs vs Cats:

```bash
python scripts/train_dogs_vs_cats.py --device auto
```

Generate predictions or a Kaggle submission CSV:

```bash
python scripts/predict_dogs_vs_cats.py \
  --checkpoint runs/project_pipeline/dogs_resnet18_pretrained_aug/best.pt \
  --split test \
  --output-csv runs/project_pipeline/submission.csv
```

Export correct and incorrect validation examples:

```bash
python scripts/analyze_dogs_vs_cats.py \
  --checkpoint runs/project_pipeline/dogs_resnet18_pretrained_aug/best.pt
```

Train CIFAR-10:

```bash
python scripts/train_cifar10.py --device auto
```

Run the end-to-end project pipeline:

```bash
python scripts/run_project_pipeline.py --device auto --profile quick
```

Run the full experiment configuration:

```bash
python scripts/run_project_pipeline.py --device auto --profile report --include-focal
```

## Included Checkpoints

Selected `best.pt` checkpoints are stored with Git LFS under `runs/project_pipeline/`:

- `dogs_resnet18_pretrained_aug/best.pt`
- `dogs_resnet18_pretrained_noaug/best.pt`
- `dogs_smallcnn_aug/best.pt`
- `cifar10_baseline/best.pt`
- `cifar10_imbalance_class_weight/best.pt`
- `cifar10_imbalance_weighted_sampler/best.pt`
- `cifar10_imbalance_focal_loss/best.pt`

After cloning, fetch the large files with:

```bash
git lfs pull
```

## Notes

- `runs/` is mostly ignored to avoid committing transient training artifacts
- datasets, caches, and generated report files are not tracked
- only reusable checkpoints are versioned
