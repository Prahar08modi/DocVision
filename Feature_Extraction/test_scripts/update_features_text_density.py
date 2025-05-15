import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from load_coco_annotations import load_coco_annotations

def compute_all_densities(ann_df, image_dir):
    """
    Compute text_density for every ROI in ann_df by:
      - grouping by file_name
      - thresholding each page once
      - slicing and counting ink pixels per bbox
    Returns a NumPy array of densities in the same order as ann_df.
    """
    densities = np.empty(len(ann_df), dtype=float)
    densities.fill(np.nan)

    # Group annotations by image file
    grouped = ann_df.groupby('file_name')

    for file_name, group in tqdm(grouped, total=len(grouped),
                                 desc="Images"):
        img_path = os.path.join(image_dir, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Binarize once per page
        _, bw = cv2.threshold(
            img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        H, W = bw.shape

        # Process all ROIs for this image
        for idx, row in zip(group.index, group.itertuples(False)):
            x0, y0, w0, h0 = row.bbox
            # clamp & int‐cast
            x = max(0, int(np.floor(x0)))
            y = max(0, int(np.floor(y0)))
            w = max(0, int(np.ceil(w0)))
            h = max(0, int(np.ceil(h0)))
            if x + w > W: w = W - x
            if y + h > H: h = H - y
            if w <= 0 or h <= 0:
                densities[idx] = np.nan
                continue

            roi_bw = bw[y:y+h, x:x+w]
            if roi_bw.size == 0:
                densities[idx] = np.nan
                continue

            ink = int((roi_bw == 0).sum())
            densities[idx] = ink / float(w*h)

    return densities


def update_split_fast(split,
                      base_dir='Data/DocLayNet_core',
                      feature_dir='.'):
    coco_dir  = os.path.join(base_dir, 'COCO')
    png_dir   = os.path.join(base_dir, 'PNG')
    ann_path  = os.path.join(coco_dir, f"{split}.json")
    feat_path = os.path.join(feature_dir, f"features_{split}.csv")
    out_path  = os.path.join(feature_dir, f"updated_features_{split}.csv")

    print(f"\n▶️  Processing split: {split}")

    # 1. Load annotations and existing features
    print("   • Loading annotations…")
    ann_df = load_coco_annotations(ann_path)
    print(f"   • Loading features from {feat_path}…")
    feat_df = pd.read_csv(feat_path)

    # 2. Compute densities in grouped fashion
    print("   • Computing text_density (threshold each image once)…")
    densities = compute_all_densities(ann_df, png_dir)

    # 3. Append and save
    feat_df['text_density'] = densities
    feat_df.to_csv(out_path, index=False)
    print(f"✅  Saved updated features to {out_path}")


def main():
    for split in ['train', 'val', 'test']:
        update_split_fast(split)

if __name__ == "__main__":
    main()
