# TODO: Middleslice Prediction Experiments

## Current Status
- [x] Swin-UNETR baseline working
- [x] Basic U-Net model
- [x] UNETR model
- [x] Wavelet wrapper (haar, db2) for all architectures
- [x] Dataset loader working (256x256 slices)
- [x] Training loop with W&B logging
- [x] Qualitative visualization working
- [x] Evaluation with MSE/SSIM (global + per-modality)
- [x] Tumor-region-focused evaluation (MSE/SSIM within WT, TC, ET)
- [x] Linear interpolation baseline
- [x] Cubic interpolation baseline
- [x] Downstream segmentation evaluation (Dice for WT, TC, ET)
- [x] Preprocessor saves segmentation masks for downstream eval

## Model Configurations (Grid Search)

| Model Type | Wavelet | Training Required |
|------------|---------|-------------------|
| swin       | none    | Yes               |
| swin       | haar    | Yes               |
| swin       | db2     | Yes               |
| unet       | none    | Yes               |
| unet       | haar    | Yes               |
| unet       | db2     | Yes               |
| unetr      | none    | Yes               |
| unetr      | haar    | Yes               |
| unetr      | db2     | Yes               |
| interpolation       | N/A | No (eval only) |
| interpolation_cubic  | N/A | No (eval only) |

## Remaining Tasks

### Training
- [ ] Train all 9 learning-based model configurations
- [ ] Evaluate interpolation baselines (linear + cubic)

### Downstream Segmentation
- [ ] Obtain/train nnU-Net segmentation model for BraTS 2023
- [ ] Run `evaluate_segmentation.py` on all model predictions
- [ ] Report Dice scores (WT, TC, ET) in paper

### Paper Figure Generation
- [ ] Select 1-2 best examples (clear tumor, low MSE)
- [ ] Generate comparison figure: [Z-1][Z+1][UNet][Swin][UNETR][Wavelet][GT][Error]
- [ ] Export as high-res PDF

### Paper Writing
- [ ] Write Methods section (spatial + wavelet domain modeling)
- [ ] Write Results section with tables
- [ ] Generate Figure 1 (method diagram)
- [ ] Update abstract with final numbers

## File Organization

```
isbi26-slideprediction/
├── train.py                         - Main training script
├── evaluate.py                      - MSE/SSIM evaluation (global + tumor-region)
├── evaluate_segmentation.py         - Downstream segmentation Dice evaluation
├── preprocess_slices_to_tensors.py  - Data preprocessing (saves seg masks)
├── preprocessed_dataset.py          - Fast tensor dataset loader
├── transforms.py                    - MONAI preprocessing pipeline
├── utils.py                         - Utility functions
├── logging_utils.py                 - W&B logging utilities
├── generate_figure2.py              - Paper figure generation
├── models/
│   ├── __init__.py
│   ├── wavelet_diffusion.py         - Standalone wavelet U-Net (Fast-cWDM)
│   ├── wavelet_wrapper.py           - Wavelet wrapper for any base model
│   └── interpolation_wrapper.py     - Linear + cubic interpolation baselines
├── sweep.yml                        - W&B hyperparameter sweep config
└── requirements.txt                 - Python dependencies
```

## Notes
- Image size: 256x256 for all models (fair comparison)
- Primary metrics: MSE, SSIM (global + per-modality + tumor-region)
- Downstream: Dice scores via segmentation model
- BraTS 2023 labels: 1=NCR, 2=ED, 3=ET; WT={1,2,3}, TC={1,3}, ET={3}
