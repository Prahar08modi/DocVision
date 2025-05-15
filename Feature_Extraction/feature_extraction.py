#!/usr/bin/env python3
"""
Unified feature extraction script that computes geometry, intensity, edge, and text density features
for COCO-format annotations (e.g. from DocStrum, XYCut, or hybrid pipelines).

Now also captures each annotation's original `id` as `annotation_id`.

Usage:
    python feature_extraction.py \
        --coco-json path/to/annotations.json \
        --image-dir path/to/images/ \
        --output-dir path/to/save/
"""
import os
import argparse
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from load_coco_annotations import load_coco_annotations


def compute_text_density(ann_df: pd.DataFrame, image_dir: str) -> np.ndarray:
    densities = np.full(len(ann_df), np.nan, dtype=float)
    grouped = ann_df.groupby('file_name')

    for file_name, group in tqdm(grouped, desc="Computing text densities", total=len(grouped)):
        img_path = os.path.join(image_dir, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        H, W = bw.shape

        for idx, row in group.iterrows():
            x0, y0, w0, h0 = row['bbox']
            x = max(0, int(np.floor(x0)))
            y = max(0, int(np.floor(y0)))
            w = max(0, int(np.ceil(w0)))
            h = max(0, int(np.ceil(h0)))
            if x + w > W:
                w = W - x
            if y + h > H:
                h = H - y
            if w <= 0 or h <= 0:
                continue

            roi = bw[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            ink_count = int((roi == 0).sum())
            densities[idx] = ink_count / float(w * h)

    return densities


def extract_features(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        img_path = os.path.join(image_dir, row['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        H, W = img.shape
        x0, y0, w0, h0 = row['bbox']
        x = int(max(0, np.floor(x0)))
        y = int(max(0, np.floor(y0)))
        w = int(max(0, np.ceil(w0)))
        h = int(max(0, np.ceil(h0)))
        if x + w > W:
            w = W - x
        if y + h > H:
            h = H - y
        if w <= 0 or h <= 0:
            continue

        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        # geometry
        area = w * h
        aspect_ratio = w / h
        width_norm    = w / W
        height_norm   = h / H
        x_center_norm = (x + w/2) / W
        y_center_norm = (y + h/2) / H
        area_norm     = area / (W * H)

        # intensity
        mean_intensity   = float(np.mean(roi))
        std_intensity    = float(np.std(roi))
        median_intensity = float(np.median(roi))

        # edge density
        edges = cv2.Canny(roi, 100, 200)
        edge_density = float((edges > 0).sum()) / area

        # text density
        text_density = float(row.get('text_density', np.nan))

        records.append({
            'annotation_id':   int(row['annotation_id']),
            'file_name':       row['file_name'],
            'category':        row.get('category', None),
            'width_norm':      width_norm,
            'height_norm':     height_norm,
            'x_center_norm':   x_center_norm,
            'y_center_norm':   y_center_norm,
            'area_norm':       area_norm,
            'aspect_ratio':    aspect_ratio,
            'area':            area,
            'mean_intensity':  mean_intensity,
            'std_intensity':   std_intensity,
            'median_intensity':median_intensity,
            'edge_density':    edge_density,
            'text_density':    text_density
        })

    return pd.DataFrame(records)


def save_features(coco_json: str, image_dir: str, output_dir: str = '.') -> pd.DataFrame:
    # load annotations DataFrame
    ann_df = load_coco_annotations(coco_json)

    # also pull original annotation IDs in order
    with open(coco_json, 'r') as f:
        raw = json.load(f)
    ann_ids = [ann['id'] for ann in raw['annotations']]
    ann_df['annotation_id'] = ann_ids

    # compute and attach text_density
    ann_df['text_density'] = compute_text_density(ann_df, image_dir)

    # extract full feature set
    feat_df = extract_features(ann_df, image_dir)

    # save to CSV
    split = os.path.splitext(os.path.basename(coco_json))[0]
    out_name = f"features_{split}.csv"
    out_path = os.path.join(output_dir, out_name)
    feat_df.to_csv(out_path, index=False)
    print(f"[✓] Extracted {len(feat_df)} regions → {out_path}")

    return feat_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract geometry, intensity, edge, and text-density features (with annotation_id)"
    )
    parser.add_argument('--coco-json', required=True,
                        help='Path to COCO-format annotations JSON')
    parser.add_argument('--image-dir', required=True,
                        help='Directory containing source images')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to write the features CSV')
    args = parser.parse_args()
    save_features(args.coco_json, args.image_dir, args.output_dir)


if __name__ == '__main__':
    main()
