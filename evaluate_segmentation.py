#!/usr/bin/env python3
"""
Downstream segmentation evaluation for reconstructed volumes.

Evaluates the clinical impact of middle-slice reconstruction by running
tumor segmentation on reconstructed volumes and computing Dice scores
for whole tumor (WT), tumor core (TC), and enhancing tumor (ET) regions.

BraTS 2023 label convention:
    1 = necrotic core (NCR)
    2 = peritumoral edema (ED)
    3 = enhancing tumor (ET)

Derived regions:
    WT = labels {1, 2, 3}
    TC = labels {1, 3}
    ET = label  {3}

Usage:
    python evaluate_segmentation.py \
        --predictions_dir ./results/swin_baseline/predictions \
        --seg_model_checkpoint /path/to/nnunet_checkpoint.pth \
        --output_dir ./results/segmentation_eval
"""

import argparse
import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ──────────────────────────────────────────────────────────
# Dice helpers
# ──────────────────────────────────────────────────────────

def dice_coefficient(pred_mask, gt_mask, smooth=1e-5):
    """Compute Dice coefficient between two binary masks.

    Args:
        pred_mask: np.ndarray (bool or 0/1) - predicted binary mask.
        gt_mask:   np.ndarray (bool or 0/1) - ground-truth binary mask.
        smooth: Laplace smoothing to avoid division by zero.

    Returns:
        float: Dice score in [0, 1].
    """
    pred = pred_mask.astype(np.float32).ravel()
    gt = gt_mask.astype(np.float32).ravel()
    intersection = (pred * gt).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth))


def compute_brats_dice(pred_seg, gt_seg):
    """Compute Dice for the three standard BraTS regions.

    Args:
        pred_seg: np.ndarray [H, W] or [D, H, W] integer labels (0-3).
        gt_seg:   np.ndarray [H, W] or [D, H, W] integer labels (0-3).

    Returns:
        dict with keys 'dice_wt', 'dice_tc', 'dice_et'.
    """
    # Whole tumor
    pred_wt = pred_seg > 0
    gt_wt = gt_seg > 0

    # Tumor core = NCR (1) + ET (3)
    pred_tc = (pred_seg == 1) | (pred_seg == 3)
    gt_tc = (gt_seg == 1) | (gt_seg == 3)

    # Enhancing tumor
    pred_et = pred_seg == 3
    gt_et = gt_seg == 3

    return {
        'dice_wt': dice_coefficient(pred_wt, gt_wt),
        'dice_tc': dice_coefficient(pred_tc, gt_tc),
        'dice_et': dice_coefficient(pred_et, gt_et),
    }


# ──────────────────────────────────────────────────────────
# Segmentation model loading
# ──────────────────────────────────────────────────────────

def load_segmentation_model(checkpoint_path, device, img_size=256):
    """Load a segmentation model from checkpoint.

    Supports MONAI SwinUNETR segmentation models and generic PyTorch
    checkpoints with a 'model_state_dict' key.

    Args:
        checkpoint_path: path to .pth checkpoint file.
        device: torch device.
        img_size: input spatial size for the segmentation model.

    Returns:
        nn.Module: loaded segmentation model in eval mode.
    """
    from monai.networks.nets import SwinUNETR

    # Default: SwinUNETR for BraTS segmentation (4 input modalities -> 4 output classes)
    model = SwinUNETR(
        in_channels=4,
        out_channels=4,      # background + NCR + ED + ET
        feature_size=48,
        spatial_dims=2,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    print(f"Loaded segmentation model from {checkpoint_path}")
    return model


def run_simple_threshold_segmentation(prediction, gt_seg):
    """Fallback segmentation when no trained model is available.

    Uses a simple intensity-threshold heuristic on the reconstructed
    multi-modal slice to produce a rough segmentation.  This is NOT
    clinically meaningful but allows the pipeline to run end-to-end
    for development and debugging.

    Args:
        prediction: np.ndarray [4, H, W] - reconstructed modalities.
        gt_seg:     np.ndarray [1, H, W] or [H, W] - ground truth segmentation.

    Returns:
        np.ndarray [H, W] integer labels.
    """
    gt = gt_seg.squeeze()  # [H, W]

    # Simple threshold-based heuristic using T1c (channel 1) and T2f (channel 3)
    t1c = prediction[1]
    t2f = prediction[3]

    seg = np.zeros_like(gt, dtype=np.int64)

    # Rough heuristics (will NOT match real nnU-Net quality)
    brain_mask = prediction.mean(axis=0) > 0.05
    seg[brain_mask & (t2f > 0.4)] = 2        # Edema (high T2-FLAIR)
    seg[brain_mask & (t1c > 0.5)] = 1        # Necrotic core (high T1c)
    seg[brain_mask & (t1c > 0.7)] = 3        # Enhancing tumor (very high T1c)

    return seg


# ──────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────

def evaluate_segmentation_from_predictions(
    predictions_dir,
    output_dir,
    seg_model_checkpoint=None,
    device=None,
    img_size=256,
):
    """Evaluate downstream segmentation on reconstructed slices.

    Loads per-sample predictions and ground-truth segmentation masks
    from disk (as saved by evaluate.py), runs segmentation, and computes
    Dice scores.

    Args:
        predictions_dir: directory containing per-patient prediction folders
                         (each with prediction_middle_slice.npy, target_middle_slice.npy,
                          and optionally metadata.npz with seg info).
        output_dir: where to save Dice results.
        seg_model_checkpoint: path to segmentation model checkpoint (optional).
        device: torch device.
        img_size: spatial size for segmentation model.

    Returns:
        dict with aggregated Dice scores.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation model if checkpoint provided
    seg_model = None
    if seg_model_checkpoint and os.path.exists(seg_model_checkpoint):
        seg_model = load_segmentation_model(seg_model_checkpoint, device, img_size)

    # Find all patient prediction directories
    patient_dirs = sorted([d for d in predictions_dir.iterdir() if d.is_dir()])
    if not patient_dirs:
        print(f"No patient directories found in {predictions_dir}")
        return {}

    print(f"Found {len(patient_dirs)} patient prediction directories")
    if seg_model:
        print("Using trained segmentation model for downstream evaluation")
    else:
        print("No segmentation model provided - using threshold-based heuristic")
        print("(Provide --seg_model_checkpoint for meaningful Dice scores)")

    all_dice = defaultdict(list)
    all_results = []

    for patient_dir in tqdm(patient_dirs, desc="Segmentation evaluation"):
        pred_path = patient_dir / 'prediction_middle_slice.npy'
        target_path = patient_dir / 'target_middle_slice.npy'
        metadata_path = patient_dir / 'metadata.npz'

        if not pred_path.exists() or not target_path.exists():
            continue

        # Load prediction and target
        prediction = np.load(pred_path)  # [4, H, W]
        target = np.load(target_path)    # [4, H, W]

        # Load metadata to get patient_id and ground truth segmentation
        patient_id = patient_dir.name
        gt_seg = None

        if metadata_path.exists():
            metadata = np.load(metadata_path, allow_pickle=True)
            patient_id = str(metadata.get('patient_id', patient_dir.name))

        # Check for ground truth segmentation saved during preprocessing
        seg_path = patient_dir / 'seg_middle_slice.npy'
        if seg_path.exists():
            gt_seg = np.load(seg_path)  # [1, H, W] or [H, W]

        if gt_seg is None:
            # No ground truth segmentation available - skip
            continue

        gt_seg_squeezed = gt_seg.squeeze()  # [H, W]

        # Run segmentation on reconstructed slice
        if seg_model is not None:
            # Neural network segmentation
            with torch.no_grad():
                pred_tensor = torch.from_numpy(prediction).float().unsqueeze(0).to(device)
                seg_output = seg_model(pred_tensor)
                pred_seg = seg_output.argmax(dim=1).squeeze(0).cpu().numpy()

            # Also segment the ground truth for comparison
            with torch.no_grad():
                gt_tensor = torch.from_numpy(target).float().unsqueeze(0).to(device)
                gt_seg_output = seg_model(gt_tensor)
                gt_seg_from_model = gt_seg_output.argmax(dim=1).squeeze(0).cpu().numpy()
        else:
            # Threshold-based fallback
            pred_seg = run_simple_threshold_segmentation(prediction, gt_seg)
            gt_seg_from_model = None

        # Compute Dice: reconstructed slice seg vs ground truth seg
        dice_scores = compute_brats_dice(pred_seg, gt_seg_squeezed.astype(np.int64))

        # If we also segmented the original slice, compute reference Dice
        if gt_seg_from_model is not None:
            ref_dice = compute_brats_dice(gt_seg_from_model, gt_seg_squeezed.astype(np.int64))
            dice_scores['ref_dice_wt'] = ref_dice['dice_wt']
            dice_scores['ref_dice_tc'] = ref_dice['dice_tc']
            dice_scores['ref_dice_et'] = ref_dice['dice_et']

        dice_scores['patient_id'] = patient_id
        all_results.append(dice_scores)

        for key in ['dice_wt', 'dice_tc', 'dice_et']:
            all_dice[key].append(dice_scores[key])
        if 'ref_dice_wt' in dice_scores:
            for key in ['ref_dice_wt', 'ref_dice_tc', 'ref_dice_et']:
                all_dice[key].append(dice_scores[key])

    # Aggregate results
    if not all_dice:
        print("No samples with segmentation masks found. Skipping Dice evaluation.")
        return {}

    results = {}
    print("\n" + "=" * 60)
    print("DOWNSTREAM SEGMENTATION EVALUATION")
    print("=" * 60)

    region_labels = {
        'dice_wt': 'Whole Tumor (WT)',
        'dice_tc': 'Tumor Core (TC)',
        'dice_et': 'Enhancing Tumor (ET)',
    }

    print(f"\nSamples evaluated: {len(all_dice['dice_wt'])}")
    print("\nDice Scores (Reconstructed -> Segmentation vs GT Segmentation):")
    for key, label in region_labels.items():
        vals = all_dice[key]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        results[f'{key}_mean'] = mean_val
        results[f'{key}_std'] = std_val
        print(f"  {label}: {mean_val:.4f} +/- {std_val:.4f}")

    if 'ref_dice_wt' in all_dice:
        print("\nReference Dice (Original GT Slice -> Segmentation vs GT Segmentation):")
        for key in ['ref_dice_wt', 'ref_dice_tc', 'ref_dice_et']:
            label = region_labels[key.replace('ref_', '')]
            vals = all_dice[key]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            results[f'{key}_mean'] = mean_val
            results[f'{key}_std'] = std_val
            print(f"  {label}: {mean_val:.4f} +/- {std_val:.4f}")

    print("=" * 60)

    # Save results
    results['num_samples'] = len(all_dice['dice_wt'])

    # Save summary
    summary_path = output_dir / 'segmentation_dice_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Save per-sample results
    import csv
    details_path = output_dir / 'segmentation_dice_detailed.csv'
    if all_results:
        with open(details_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Detailed results saved to {details_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Downstream segmentation evaluation for reconstructed volumes"
    )
    parser.add_argument(
        '--predictions_dir', type=str, required=True,
        help='Directory containing per-patient prediction folders from evaluate.py'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results/segmentation_eval',
        help='Output directory for Dice results'
    )
    parser.add_argument(
        '--seg_model_checkpoint', type=str, default=None,
        help='Path to segmentation model checkpoint (optional; uses threshold heuristic if omitted)'
    )
    parser.add_argument(
        '--img_size', type=int, default=256,
        help='Image size for segmentation model'
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    evaluate_segmentation_from_predictions(
        predictions_dir=args.predictions_dir,
        output_dir=args.output_dir,
        seg_model_checkpoint=args.seg_model_checkpoint,
        device=device,
        img_size=args.img_size,
    )


if __name__ == '__main__':
    main()
