# Data Directory

Training, validation, patch, and inference data for WatNetv2.

## Directory structure

```text
data/
├── dset/
│   ├── train/
│   │   ├── scene/        # Normalized images
│   │   └── truth/        # Ground-truth masks
│   ├── val/
│   │   ├── scene/        # Normalized images
│   │   └── truth/        # Ground-truth masks
│   └── val_patch/
│       ├── patch_512/    # 512 × 512 validation patches
│       └── patch_1024_null/
│                         # Reserved 1024 × 1024 patches
├── result/               # Predictions and evaluation results
└── readme.md
```

## Data conventions

- `scene` files are multispectral GeoTIFF images.
- `truth` files are single-band binary masks (`0`: background, `1`: surface water).
- Validation patches contain image bands followed by one ground-truth band and
  are stored as PyTorch tensors (`.pth`).
- Paired images and masks must share identifiers, extents, resolutions, and
  coordinate reference systems.

## Version control

Large data files are excluded by `.gitignore`. Keep only documentation and
lightweight metadata in Git; use external storage or Git LFS for datasets.
