#!/usr/bin/env python3
"""
classify_features.py

Loads:
  --features-csv : CSV of region features (one row per region)
  --model        : trained LightGBM Booster (.pkl)
Produces:
  --out-csv      : original CSV + pred_label_id, pred_score
"""

import argparse
import pandas as pd
import numpy as np
import joblib

# Features used during classification (must match training order)
FEATURES_TO_USE = [
    'y_center_norm',
    'std_intensity',
    'text_density',
    'mean_intensity',
    'aspect_ratio',
    'edge_density',
    'area_norm',
    'x_center_norm'
]

def main():
    p = argparse.ArgumentParser(
        description="Classify extracted region features with a LightGBM model"
    )
    p.add_argument("--features-csv", required=True,
                   help="Input CSV of features (one row per region)")
    p.add_argument("--model",        required=True,
                   help="Path to trained LightGBM Booster (.pkl)")
    p.add_argument("--out-csv",      default="classified_features.csv",
                   help="Where to save the predictions CSV")
    args = p.parse_args()

    # 1) Load features
    df = pd.read_csv(args.features_csv)

    # 1a) Select only the required features
    missing = [f for f in FEATURES_TO_USE if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}")
    X = df[FEATURES_TO_USE].values.astype(float)

    # 2) Load model
    clf = joblib.load(args.model)

    # 3) Predict
    raw = clf.predict(X)
    arr = np.asarray(raw)
    if arr.ndim == 2:
        # probability outputs
        probas = arr
        preds = np.argmax(probas, axis=1)
        scores = probas[np.arange(len(preds)), preds]
    elif arr.ndim == 1:
        # direct class IDs
        preds = arr.astype(int)
        scores = np.ones(len(preds), dtype=float)
    else:
        raise ValueError(f"Unexpected prediction shape: {arr.shape}")

    # 4) Attach predictions
    df["pred_label_id"] = preds
    df["pred_score"]    = scores

    # 5) Save
    df.to_csv(args.out_csv, index=False)
    print(f"✅ Wrote {len(df)} predictions to {args.out_csv}")

if __name__ == "__main__":
    main()
