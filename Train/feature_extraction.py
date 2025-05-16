import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from load_coco_annotations import load_coco_annotations

def extract_features(df, image_dir):
    """
    Given a DataFrame of COCO annotations, crop each bbox from the corresponding
    image and compute a richer set of features:
      - normalized width, height, center position
      - normalized area
      - aspect_ratio, absolute area
      - mean, std, median intensity
      - edge density (Canny)
    """
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        img_path = os.path.join(image_dir, row['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        H, W = img.shape
        # original bbox coords (may be floats)
        x0, y0, w0, h0 = row['bbox']
        # clamp/round to ints
        x = int(max(0, np.floor(x0)))
        y = int(max(0, np.floor(y0)))
        w = int(max(0, np.ceil(w0)))
        h = int(max(0, np.ceil(h0)))
        # adjust width/height if it spills outside
        if x + w > W:
            w = W - x
        if y + h > H:
            h = H - y
        # skip empty or invalid
        if w <= 0 or h <= 0:
            continue
        
        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        
        # geometric
        aspect_ratio = w / h
        area = w * h
        
        # normalized geometry
        width_norm    = w / W
        height_norm   = h / H
        x_center_norm = (x + w/2) / W
        y_center_norm = (y + h/2) / H
        area_norm     = area / (W * H)
        
        # intensity stats (will not warn since roi.size > 0)
        mean_intensity   = float(np.mean(roi))
        std_intensity    = float(np.std(roi))
        median_intensity = float(np.median(roi))
        
        # edge density (safe since roi is non-empty)
        edges = cv2.Canny(roi, 100, 200)
        edge_density = float((edges > 0).sum()) / area
        
        records.append({
            'file_name':        row['file_name'],
            'category':         row['category'],
            'width_norm':       width_norm,
            'height_norm':      height_norm,
            'x_center_norm':    x_center_norm,
            'y_center_norm':    y_center_norm,
            'area_norm':        area_norm,
            'aspect_ratio':     aspect_ratio,
            'area':             area,
            'mean_intensity':   mean_intensity,
            'std_intensity':    std_intensity,
            'median_intensity': median_intensity,
            'edge_density':     edge_density
        })
    
    return pd.DataFrame(records)

def save_features(json_path, image_dir, output_dir='.'):
    ann_df = load_coco_annotations(json_path)
    feat_df = extract_features(ann_df, image_dir)
    
    split_name = os.path.splitext(os.path.basename(json_path))[0]
    out_name   = f"features_{split_name}.csv"
    out_path   = os.path.join(output_dir, out_name)
    
    feat_df.to_csv(out_path, index=False)
    print(f"[✓] Extracted {len(feat_df)} regions → {out_path}")
    return feat_df

if __name__ == "__main__":
    base = 'Data/DocLayNet_core'
    coco = os.path.join(base, 'COCO')
    png  = os.path.join(base, 'PNG')
    out  = '.'
    
    for split in ['train', 'val', 'test']:
        json_file = os.path.join(coco, f"{split}.json")
        save_features(json_file, png, out)
