#!/usr/bin/env python3
"""
Baseline: Random Forest on hand-crafted biophysical features.
Leave-one-family-out evaluation.

Expected result: F1 ~ 0.533, AUC ~ 0.611, TP=8/21, Retroviral TP=2/12

This is the strongest simple baseline we found. Beat it.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

DATA_DIR = Path(__file__).parent.parent / "data"

# ============================================================
# Load data
# ============================================================
sequences = pd.read_csv(DATA_DIR / "rt_sequences.csv")
features = pd.read_csv(DATA_DIR / "handcrafted_features.csv")

# Merge on rt_name
df = sequences.merge(features, on='rt_name')

# Feature columns (everything except metadata)
meta = ['rt_name', 'sequence', 'active', 'pe_efficiency_pct', 'rt_family', 'protein_length_aa']
feat_cols = [c for c in df.columns if c not in meta]

X = df[feat_cols].fillna(0).values  # Fill missing structural features with 0
y = df['active'].values
families = df['rt_family'].values
unique_fams = sorted(set(families))

print(f"Data: {len(df)} RTs, {y.sum()} active, {len(feat_cols)} features, {len(unique_fams)} families")
print(f"\nFamily breakdown:")
for fam in unique_fams:
    mask = families == fam
    print(f"  {fam:25s}  n={mask.sum():2d}  active={y[mask].sum():2d}  inactive={mask.sum()-y[mask].sum():2d}")

# ============================================================
# Leave-one-family-out evaluation
# ============================================================
print(f"\n{'='*70}")
print("LEAVE-ONE-FAMILY-OUT: Random Forest (d=10) on HandCrafted Features")
print(f"{'='*70}\n")

all_predictions = []

for held_out in unique_fams:
    test_mask = families == held_out
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    names_test = df.loc[test_mask, 'rt_name'].values

    # Check both classes in training
    if len(set(y_train)) < 2:
        print(f"  {held_out}: training set has only 1 class — predicting all inactive")
        for i, name in enumerate(names_test):
            all_predictions.append({
                'rt_name': name,
                'predicted_active': 0,
                'predicted_score': 0.0,
            })
        continue

    # Scale features (fit on training only!)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Train
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_train)

    # Predict
    y_pred = clf.predict(X_te)
    y_score = clf.predict_proba(X_te)[:, 1]

    # Record
    for i, name in enumerate(names_test):
        all_predictions.append({
            'rt_name': name,
            'predicted_active': int(y_pred[i]),
            'predicted_score': float(y_score[i]),
        })

    # Per-fold metrics
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    n_act = int(y_test.sum())
    n_inact = len(y_test) - n_act
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if n_act > 0 and n_inact > 0:
        auc = roc_auc_score(y_test, y_score)
        auc_s = f"{auc:.3f}"
    else:
        auc_s = "  N/A"

    print(f"  {held_out:25s}  n={len(y_test):2d} ({n_act}A:{n_inact}I)  "
          f"F1={f1:.3f}  AUC={auc_s}  TP={tp} FP={fp} FN={fn} TN={tn}")

# Save predictions
pred_df = pd.DataFrame(all_predictions)
output_path = Path(__file__).parent / "baseline_predictions.csv"
pred_df.to_csv(output_path, index=False)
print(f"\nSaved predictions to {output_path}")

# ============================================================
# Aggregate
# ============================================================
merged = sequences[['rt_name', 'active']].merge(pred_df, on='rt_name')
y_true = merged['active'].values
y_pred = merged['predicted_active'].values
y_score = merged['predicted_score'].values

tp = int(((y_pred == 1) & (y_true == 1)).sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum())
fn = int(((y_pred == 0) & (y_true == 1)).sum())
tn = int(((y_pred == 0) & (y_true == 0)).sum())
f1 = f1_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, y_score)

print(f"\nAGGREGATE:")
print(f"  F1={f1:.3f}  AUC={auc:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")

# Retroviral fold
retro = merged[sequences['rt_family'] == 'Retroviral']
retro_tp = int(((retro['predicted_active'] == 1) & (retro['active'] == 1)).sum())
print(f"  Retroviral TP: {retro_tp}/12")

print(f"\nTo evaluate: python ../evaluation/evaluate.py --predictions {output_path}")
