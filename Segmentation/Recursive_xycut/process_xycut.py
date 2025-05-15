import os
import cv2
import numpy as np
import json
from Segmentation.Recursive_xycut.xy_cut import find_cuts

# ─────────── CONFIG ────────────
INPUT_DIR  = "../../Dataset/DocLayNet/DocLayNet_core/PNG"
OUTPUT_DIR = "./block_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────── Binarization ───────────
def binarize_image_path(image_path: str) -> np.ndarray:
    """Load image, convert to grayscale, then Otsu‐binarize (inverted)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bin_

# ─────── Recursive XY‑Cut ─────────
def smooth(proj: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(proj, kernel, mode='same')

def recursive_xy_cut_debug(binary: np.ndarray,
                           x0=0, y0=0,
                           min_block=70,
                           thr_h=0.2, thr_v=0.05,
                           gap_h=18, gap_v=15,
                           k_h=25, k_v=15) -> list[tuple[int,int,int,int]]:
    H, W = binary.shape
    if W < min_block or H < min_block:
        return [(x0, y0, W, H)]

    h_proj = smooth(np.sum(binary, axis=1), k_h)
    v_proj = smooth(np.sum(binary, axis=0), k_v)
    raw_h  = find_cuts(h_proj, threshold_ratio=thr_h, min_gap=gap_h)
    raw_v  = find_cuts(v_proj, threshold_ratio=thr_v, min_gap=gap_v)
    h_cuts = sorted(c for c in raw_h if 0 < c < H)
    v_cuts = sorted(c for c in raw_v if 0 < c < W)

    if not h_cuts and not v_cuts:
        return [(x0, y0, W, H)]

    # vertical first
    if v_cuts:
        regions, prev = [], 0
        for cut in v_cuts + [W]:
            w = cut - prev
            if w >= min_block:
                sub = binary[:, prev:cut]
                regions += recursive_xy_cut_debug(
                    sub, x0+prev, y0,
                    min_block, thr_h, thr_v,
                    gap_h, gap_v, k_h, k_v
                )
            prev = cut
        return regions

    # else horizontal
    regions, prev = [], 0
    for cut in h_cuts + [H]:
        h = cut - prev
        if h >= min_block:
            sub = binary[prev:cut, :]
            regions += recursive_xy_cut_debug(
                sub, x0, y0+prev,
                min_block, thr_h, thr_v,
                gap_h, gap_v, k_h, k_v
            )
        prev = cut
    return regions

# ───────── Exports ────────────
def save_as_coco(blocks, binary, path, image_filename):
    h, w = binary.shape
    coco = {
        "info":       {"description":"XY-Cut Blocks","version":"1.0"},
        "licenses":   [],
        "images":     [{"id":1,"width":w,"height":h,"file_name":image_filename}],
        "annotations":[],"categories":[{"id":1,"name":"block","supercategory":"layout"}]
    }
    ann_id = 1
    for x,y,bw,bh in blocks:
        coco["annotations"].append({
            "id":       ann_id,
            "image_id": 1,
            "category_id":1,
            "bbox":     [x,y,bw,bh],
            "area":     bw*bh,
            "iscrowd":  0,
            "segmentation": [[x,y, x+bw,y, x+bw,y+bh, x,y+bh]]
        })
        ann_id += 1
    with open(path, "w") as f:
        json.dump(coco, f, indent=2)

# ─────────── Pipeline ────────────
def process_image(image_path: str):
    base = os.path.splitext(os.path.basename(image_path))[0]
    binary = binarize_image_path(image_path)
    blocks = recursive_xy_cut_debug(binary)

    # prepare output paths
    coco_path = os.path.join(OUTPUT_DIR, f"{base}_blocks_coco.json")

    # save all formats
    save_as_coco(blocks, binary, coco_path, image_filename=os.path.basename(image_path))

    print(f"[+] {base}: {len(blocks)} blocks → COCO")

if __name__ == "__main__":
    # process every image in INPUT_DIR
    for fname in sorted(os.listdir(INPUT_DIR)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            process_image(os.path.join(INPUT_DIR, fname))
    print("✅ All images processed. Outputs in:", OUTPUT_DIR)
