# docstrum_blocks.py

import cv2
import numpy as np
from typing import List, Tuple
import argparse

# your existing modules
from docstrum_components import get_connected_components, Component
from docstrum_geometry   import (
    build_nn_graph,
    estimate_page_orientation,
    estimate_stroke_height,
    estimate_char_spacing
)
from docstrum_lines      import (
    build_docstrum_lines,
    clusters_to_boxes,
    draw_line_boxes
)

def overlap_frac(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Horizontal overlap fraction between two boxes."""
    x0, y0, w0, h0 = a
    x1, y1, w1, h1 = b
    inter_left  = max(x0, x1)
    inter_right = min(x0 + w0, x1 + w1)
    if inter_right <= inter_left:
        return 0.0
    return (inter_right - inter_left) / min(w0, w1)

def merge_line_boxes(
    boxes: List[Tuple[int,int,int,int]],
    stroke_height: float,
    gap_mul: float = 1.5,
    overlap_thresh: float = 0.3
) -> List[Tuple[int,int,int,int]]:
    """
    Merge sorted line‑boxes into paragraph blocks.
    """
    if not boxes:
        return []

    # 1) sort by top‑y
    boxes = sorted(boxes, key=lambda b: b[1])
    blocks: List[List[Tuple[int,int,int,int]]] = []
    current = [boxes[0]]

    for b in boxes[1:]:
        prev = current[-1]
        prev_y2 = prev[1] + prev[3]
        gap     = b[1] - prev_y2

        if gap <= stroke_height * gap_mul and overlap_frac(prev, b) >= overlap_thresh:
            current.append(b)
        else:
            blocks.append(current)
            current = [b]
    blocks.append(current)

    # 2) collapse each block into one bbox
    merged: List[Tuple[int,int,int,int]] = []
    for blk in blocks:
        xs = [x for x,y,w,h in blk]
        ys = [y for x,y,w,h in blk]
        xe = [x+w for x,y,w,h in blk]
        ye = [y+h for x,y,w,h in blk]
        merged.append((
            min(xs), min(ys),
            max(xe) - min(xs),
            max(ye) - min(ys)
        ))
    return merged

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Merge DocStrum line‑boxes into paragraph blocks"
    )
    parser.add_argument("image", help="Path to input page image")
    parser.add_argument("--min-area", type=int,   default=10)
    parser.add_argument("--max-area", type=int,   default=5000)
    parser.add_argument("--k",        type=int,   default=5)
    parser.add_argument("--angle-tol",     type=float, default=30.0)
    parser.add_argument("--min-dist-factor", type=float, default=0.5)
    parser.add_argument("--max-dist-factor", type=float, default=6.0)
    parser.add_argument("--gap-mul",        type=float, default=1.5,
                        help="max vertical gap = gap_mul * stroke_height")
    parser.add_argument("--overlap-thresh", type=float, default=0.3,
                        help="min horizontal overlap fraction to merge lines")
    parser.add_argument("--out", help="Where to save block overlay PNG")
    args = parser.parse_args()

    # 1) load & binarize
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot load: {args.image}")
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2) CC → components
    comps = get_connected_components(
        bin_img,
        min_area=args.min_area,
        max_area=args.max_area
    )
    print(f"[Blocks] → {len(comps)} CCs")

    # 3) geometry estimates
    nn_graph = build_nn_graph(comps, k=args.k)
    ori     = estimate_page_orientation(nn_graph)
    stroke  = estimate_stroke_height(comps)
    spacing = estimate_char_spacing(nn_graph, ori)

    print(f"[Blocks] ori={ori:.1f}°, stroke={stroke:.1f}px, spacing={spacing:.1f}px")

    # 4) line clusters → line boxes
    clusters  = build_docstrum_lines(
        comps, ori, spacing, stroke,
        min_dist_factor=args.min_dist_factor,
        max_dist_factor=args.max_dist_factor,
        angle_tol=args.angle_tol
    )
    line_boxes = clusters_to_boxes(clusters, comps)
    print(f"[Blocks] → {len(line_boxes)} line boxes")

    # 5) merge into paragraph blocks
    para_boxes = merge_line_boxes(
        line_boxes,
        stroke_height=stroke,
        gap_mul=args.gap_mul,
        overlap_thresh=args.overlap_thresh
    )
    print(f"[Blocks] → {len(para_boxes)} paragraph blocks")

    # 6) draw & save
    vis = draw_line_boxes(gray, para_boxes)
    if args.out:
        cv2.imwrite(args.out, vis)
        print(f"Saved block overlay to {args.out}")
    else:
        cv2.imshow("Blocks", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
