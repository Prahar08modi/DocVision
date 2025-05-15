import cv2
import numpy as np
import matplotlib.pyplot as plt
from Segmentation.Recursive_xycut.test_scripts.binarize import binarize_image
from xy_cut import find_cuts

def smooth(proj: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(proj, kernel, mode='same')

def recursive_xy_cut_debug(
    binary, x0=0, y0=0,
    min_block=70,
    thr_h=0.2, thr_v=0.05,
    gap_h=18, gap_v=15,
    k_h=25, k_v=15
):
    H, W = binary.shape
    if W < min_block or H < min_block:
        return [(x0, y0, W, H)]

    # smooth + find true‑valley cuts
    h_proj = smooth(np.sum(binary, axis=1), k_h)
    v_proj = smooth(np.sum(binary, axis=0), k_v)
    raw_h = find_cuts(h_proj, threshold_ratio=thr_h, min_gap=gap_h)
    raw_v = find_cuts(v_proj, threshold_ratio=thr_v, min_gap=gap_v)
    h_cuts = sorted([c for c in raw_h if 0 < c < H])
    v_cuts = sorted([c for c in raw_v if 0 < c < W])

    # if no cuts remain, this is a leaf
    if not h_cuts and not v_cuts:
        return [(x0, y0, W, H)]

    # vertical splits first
    if v_cuts:
        regions, prev = [], 0
        for cut in v_cuts + [W]:
            if (cut - prev) >= min_block:
                sub = binary[:, prev:cut]
                regions += recursive_xy_cut_debug(
                    sub, x0+prev, y0,
                    min_block, thr_h, thr_v,
                    gap_h, gap_v, k_h, k_v
                )
            prev = cut
        return regions

    # else horizontal splits
    regions, prev = [], 0
    for cut in h_cuts + [H]:
        if (cut - prev) >= min_block:
            sub = binary[prev:cut, :]
            regions += recursive_xy_cut_debug(
                sub, x0, y0+prev,
                min_block, thr_h, thr_v,
                gap_h, gap_v, k_h, k_v
            )
        prev = cut
    return regions

def merge_paragraphs(boxes, x_overlap_thresh=0.7, y_gap_thresh=20):
    """
    Merge line‑level boxes into paragraph boxes.
    - Horizontal overlap ≥ x_overlap_thresh
    - Vertical gap between them ≤ y_gap_thresh
    """
    if not boxes:
        return []

    # sort by top‑y
    boxes = sorted(boxes, key=lambda b: b[1])
    merged = []
    used = [False]*len(boxes)

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if used[i]:
            continue
        # start a new cluster
        cur_x0, cur_y0 = x1, y1
        cur_x1, cur_y1 = x1+w1, y1+h1
        used[i] = True
        changed = True

        # keep aggregating any box that should merge
        while changed:
            changed = False
            for j, (x2, y2, w2, h2) in enumerate(boxes):
                if used[j]:
                    continue
                xx0, yy0 = x2, y2
                xx1, yy1 = x2+w2, y2+h2

                # horizontal overlap fraction
                overlap = min(cur_x1, xx1) - max(cur_x0, xx0)
                minw = min(cur_x1-cur_x0, xx1-xx0)
                if minw <= 0:
                    continue
                if overlap/minw < x_overlap_thresh:
                    continue

                # vertical gap
                if yy0 > cur_y1:
                    gap = yy0 - cur_y1
                else:
                    gap = cur_y0 - yy1
                if gap > y_gap_thresh:
                    continue

                # merge
                cur_x0 = min(cur_x0, xx0)
                cur_y0 = min(cur_y0, yy0)
                cur_x1 = max(cur_x1, xx1)
                cur_y1 = max(cur_y1, yy1)
                used[j] = True
                changed = True

        merged.append((cur_x0, cur_y0, cur_x1-cur_x0, cur_y1-cur_y0))

    return merged

def visualize_and_merge_blocks():
    # 1. Binarize
    binary = binarize_image()

    # 2. Extract all leaf boxes
    leaves = recursive_xy_cut_debug(binary)

    # 3. Merge into paragraphs
    paras = merge_paragraphs(leaves,
                             x_overlap_thresh=0.7,
                             y_gap_thresh=20)

    # 4. Draw
    img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in paras:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title("Paragraph‑Level Blocks after Merge")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_and_merge_blocks()
