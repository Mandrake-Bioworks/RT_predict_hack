#!/usr/bin/env python3
"""
Official evaluation script for RT-Predict Benchmark.

Usage:
    python evaluate.py --predictions path/to/predictions.csv

Predictions CSV must have columns: rt_name, predicted_active, predicted_score

Computes:
    1. Leave-one-family-out metrics (primary)
    2. Overall metrics
    3. Per-family breakdown
    4. Ranking quality on active RTs
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from scipy.stats import spearmanr, kendalltau

DATA_DIR = Path(__file__).parent.parent / "data"


def load_ground_truth():
    gt = pd.read_csv(DATA_DIR / "rt_sequences.csv")
    return gt[['rt_name', 'active', 'pe_efficiency_pct', 'rt_family']]


def evaluate(predictions_path):
    gt = load_ground_truth()
    pred = pd.read_csv(predictions_path)

    # Validate
    required_cols = {'rt_name', 'predicted_active', 'predicted_score'}
    missing = required_cols - set(pred.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions: {missing}")

    # Merge
    merged = gt.merge(pred, on='rt_name', how='left')
    missing_preds = merged['predicted_active'].isna().sum()
    if missing_preds > 0:
        print(f"WARNING: {missing_preds} RTs have no prediction. Treating as inactive.")
        merged['predicted_active'] = merged['predicted_active'].fillna(0).astype(int)
        merged['predicted_score'] = merged['predicted_score'].fillna(0.0)

    y_true = merged['active'].values
    y_pred = merged['predicted_active'].values.astype(int)
    y_score = merged['predicted_score'].values.astype(float)

    print("=" * 70)
    print("RT-PREDICT BENCHMARK — EVALUATION RESULTS")
    print("=" * 70)

    # --- 1. Overall metrics ---
    print("\n--- OVERALL ---")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = float('nan')
    print(f"  F1 (positive class): {f1:.4f}")
    print(f"  AUC-ROC:            {auc:.4f}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Sensitivity (TP rate): {tp}/{tp+fn} = {tp/(tp+fn):.3f}" if (tp+fn) > 0 else "")
    print(f"  Precision:             {tp}/{tp+fp} = {tp/(tp+fp):.3f}" if (tp+fp) > 0 else "")

    # --- 2. Per-family breakdown ---
    print("\n--- PER-FAMILY BREAKDOWN (Leave-One-Family-Out perspective) ---")
    families = merged['rt_family'].values
    unique_fams = sorted(set(families))

    print(f"\n  {'Family':25s} {'n':>3s} {'A:I':>5s} {'TP':>4s} {'FP':>4s} "
          f"{'FN':>4s} {'TN':>4s} {'F1':>6s} {'AUC':>6s}")
    print("  " + "-" * 65)

    for fam in unique_fams:
        mask = families == fam
        yt = y_true[mask]
        yp = y_pred[mask]
        ys = y_score[mask]
        n = mask.sum()
        na = int(yt.sum())
        ni = n - na

        fam_tn, fam_fp, fam_fn, fam_tp = 0, 0, 0, 0
        for a, p in zip(yt, yp):
            if a == 1 and p == 1: fam_tp += 1
            elif a == 0 and p == 1: fam_fp += 1
            elif a == 1 and p == 0: fam_fn += 1
            else: fam_tn += 1

        fam_f1 = f1_score(yt, yp, zero_division=0)
        if na > 0 and ni > 0:
            try:
                fam_auc = roc_auc_score(yt, ys)
            except:
                fam_auc = float('nan')
            auc_s = f"{fam_auc:.3f}"
        else:
            auc_s = "  N/A"

        print(f"  {fam:25s} {n:3d} {na}:{ni:>2d}  {fam_tp:4d} {fam_fp:4d} "
              f"{fam_fn:4d} {fam_tn:4d} {fam_f1:6.3f} {auc_s:>6s}")

    # --- 3. Retroviral fold (the critical test) ---
    retro_mask = families == 'Retroviral'
    retro_true = y_true[retro_mask]
    retro_pred = y_pred[retro_mask]
    retro_score = y_score[retro_mask]
    retro_tp = int(((retro_pred == 1) & (retro_true == 1)).sum())
    retro_fp = int(((retro_pred == 1) & (retro_true == 0)).sum())
    retro_fn = int(((retro_pred == 0) & (retro_true == 1)).sum())

    print(f"\n--- RETROVIRAL FOLD (THE CRITICAL TEST) ---")
    print(f"  TP: {retro_tp}/12 active Retroviral RTs correctly identified")
    print(f"  FP: {retro_fp}/6 inactive Retroviral RTs incorrectly called active")
    print(f"  FN: {retro_fn}/12 active Retroviral RTs missed")
    try:
        retro_auc = roc_auc_score(retro_true, retro_score)
        print(f"  AUC: {retro_auc:.3f}")
    except:
        print(f"  AUC: N/A")

    # --- 4. Ranking quality (active RTs only) ---
    active_mask = merged['active'] == 1
    if active_mask.sum() > 2:
        active_pe = merged.loc[active_mask, 'pe_efficiency_pct'].values
        active_scores = merged.loc[active_mask, 'predicted_score'].values

        rho, rho_p = spearmanr(active_pe, active_scores)
        tau, tau_p = kendalltau(active_pe, active_scores)

        print(f"\n--- RANKING QUALITY (21 active RTs by PE efficiency) ---")
        print(f"  Spearman rho: {rho:.4f} (p={rho_p:.4f})")
        print(f"  Kendall tau:  {tau:.4f} (p={tau_p:.4f})")

        # Top-5 accuracy: are the 5 highest-scored RTs among the 5 highest PE?
        top5_by_score = merged.loc[active_mask].nlargest(5, 'predicted_score')['rt_name'].values
        top5_by_pe = merged.loc[active_mask].nlargest(5, 'pe_efficiency_pct')['rt_name'].values
        overlap = len(set(top5_by_score) & set(top5_by_pe))
        print(f"  Top-5 overlap: {overlap}/5 of your top-5 are in the true top-5")

    # --- Summary score ---
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Primary metric (inter-family F1):  {f1:.4f}")
    print(f"  Retroviral TP:                     {retro_tp}/12")
    print(f"  Overall AUC:                       {auc:.4f}")
    if active_mask.sum() > 2:
        print(f"  Ranking Spearman rho:              {rho:.4f}")

    return {
        'f1': f1, 'auc': auc, 'acc': acc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'retroviral_tp': retro_tp, 'retroviral_fp': retro_fp,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RT-Predict predictions')
    parser.add_argument('--predictions', required=True, help='Path to predictions CSV')
    args = parser.parse_args()
    evaluate(args.predictions)
