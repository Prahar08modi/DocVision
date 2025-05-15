import cv2
import numpy as np
import json
import csv
from test_scripts.binarize import binarize_image
from xy_cut import find_cuts

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

    # smooth + find valley cuts
    h_proj = smooth(np.sum(binary, axis=1), k_h)
    v_proj = smooth(np.sum(binary, axis=0), k_v)
    raw_h = find_cuts(h_proj, threshold_ratio=thr_h, min_gap=gap_h)
    raw_v = find_cuts(v_proj, threshold_ratio=thr_v, min_gap=gap_v)
    h_cuts = sorted(c for c in raw_h if 0 < c < H)
    v_cuts = sorted(c for c in raw_v if 0 < c < W)

    if not h_cuts and not v_cuts:
        return [(x0, y0, W, H)]

    # vertical splits first
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

    # horizontal splits
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

def extract_blocks():
    """Runs binarization and XY‑Cut; returns list of (x,y,w,h) and the binary image."""
    binary = binarize_image()
    blocks = recursive_xy_cut_debug(
        binary,
        min_block=70,
        thr_h=0.2, thr_v=0.05,
        gap_h=18, gap_v=15,
        k_h=25, k_v=15
    )
    return blocks, binary

def save_as_json(blocks, filename="blocks.json"):
    data = [{"x":x, "y":y, "w":w, "h":h} for (x,y,w,h) in blocks]
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written {len(blocks)} blocks to {filename}")

def save_as_csv(blocks, filename="blocks.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x","y","w","h"])
        writer.writerows(blocks)
    print(f"Written {len(blocks)} blocks to {filename}")

def save_as_coco(blocks, binary, filename="blocks_coco.json", image_filename="image.png"):
    h, w = binary.shape
    coco = {
        "info": {"description": "XY-Cut Blocks", "version": "1.0"},
        "licenses": [],
        "images": [{"id": 1, "width": w, "height": h, "file_name": image_filename}],
        "annotations": [],
        "categories": [{"id": 1, "name": "block", "supercategory": "layout"}]
    }
    ann_id = 1
    for (x, y, bw, bh) in blocks:
        coco["annotations"].append({
            "id": ann_id,
            "image_id": 1,
            "category_id": 1,
            "bbox": [x, y, bw, bh],
            "area": bw * bh,
            "iscrowd": 0,
            "segmentation": [[x, y, x+bw, y, x+bw, y+bh, x, y+bh]]
        })
        ann_id += 1

    with open(filename, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Written {len(blocks)} COCO annotations to {filename}")

if __name__ == "__main__":
    # 1) Extract
    blocks, binary = extract_blocks()

    # 2) Save to JSON & CSV
    save_as_json(blocks, "blocks.json")
    save_as_csv(blocks,  "blocks.csv")

    # 3) Save to COCO
    #    Update image_filename if you want the actual input file name recorded
    save_as_coco(blocks, binary, filename="blocks_coco.json", image_filename="input.png")