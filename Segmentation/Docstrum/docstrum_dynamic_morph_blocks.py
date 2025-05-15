#!/usr/bin/env python3
"""
docstrum_dynamic_morph_blocks.py

Dynamic morphological merging of DocStrum line boxes into paragraph blocks
using adaptive closing height and opening.
"""
import cv2
import numpy as np
import argparse
from typing import List, Tuple

# reuse your existing modules
from Segmentation.Docstrum.docstrum_components import get_connected_components, Component
from Segmentation.Docstrum.docstrum_geometry   import (
    build_nn_graph,
    estimate_page_orientation,
    estimate_stroke_height,
    estimate_char_spacing
)
from Segmentation.Docstrum.docstrum_lines      import (
    build_docstrum_lines,
    clusters_to_boxes
)

def dynamic_morphological_merge(
    image_shape: Tuple[int,int],
    line_boxes: List[Tuple[int,int,int,int]],
    stroke_height: float,
    dynamic_pct: float = 80.0,
    open_factor: float = 0.2
) -> List[Tuple[int,int,int,int]]:
    """
    1) Rasterize each line box into a mask.
    2) Compute vertical gaps between successive lines and select the
       'dynamic_pct'-th percentile as closing height.
    3) Perform closing (dilate->erode) with a (1 x se_h) element.
    4) Perform opening (erode->dilate) with a (se_w x 1) element to break
       across-column bridges.
    5) Connected-components on result -> paragraph blocks.
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x, y, w, h) in line_boxes:
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

    # Compute vertical gaps between sorted lines
    boxes = sorted(line_boxes, key=lambda b: b[1])
    gaps = []
    for (x0, y0, w0, h0), (x1, y1, w1, h1) in zip(boxes, boxes[1:]):
        gaps.append(y1 - (y0 + h0))
    if not gaps:
        return []

    # Dynamic closing height = dynamic_pct-th percentile of gaps
    se_h = max(1, int(np.percentile(gaps, dynamic_pct)))
    # Opening width = stroke_height * open_factor
    se_w = max(1, int(stroke_height * open_factor))

    # print(f"[DynamicMorph] closing SE height: {se_h}px, opening SE width: {se_w}px (pct={dynamic_pct})")

    # Structuring elements
    se_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, se_h))
    se_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (se_w, 1))

    # Closing then opening
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se_close)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, se_open)

    # Final CC on merged mask
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(
        merged, connectivity=8
    )
    blocks = []
    for lbl in range(1, n_lbl):
        x, y, w, h, area = stats[lbl]
        blocks.append((x, y, w, h))
    return blocks


def draw_boxes(
    img: np.ndarray,
    boxes: List[Tuple[int,int,int,int]],
    color: Tuple[int,int,int],
    thickness: int = 2
) -> np.ndarray:
    """
    Draw rectangles for each box in `boxes` on `img`.
    """
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, thickness)
    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic morphological merging of DocStrum lines into paragraph blocks"
    )
    parser.add_argument("image", help="Path to input page image")
    parser.add_argument("--min-area",    type=int,   default=10)
    parser.add_argument("--max-area",    type=int,   default=5000)
    parser.add_argument("--k",           type=int,   default=5)
    parser.add_argument("--angle-tol",   type=float, default=30.0)
    parser.add_argument("--min-dist-factor", type=float, default=0.5)
    parser.add_argument("--max-dist-factor", type=float, default=6.0)
    parser.add_argument("--dynamic-pct",     type=float, default=80.0,
                        help="percentile of inter-line gaps for closing SE")
    parser.add_argument("--open-factor",     type=float, default=0.2,
                        help="opening SE width = open_factor * stroke_height")
    parser.add_argument("--out",             help="Output visualization path")
    args = parser.parse_args()

    # 1) load & binarize
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot load: {args.image}")
    H, W = gray.shape
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2) connected components -> components
    comps = get_connected_components(
        bin_img,
        min_area=args.min_area,
        max_area=args.max_area
    )

    # 3) geometry estimates
    nn_graph = build_nn_graph(comps, k=args.k)
    ori     = estimate_page_orientation(nn_graph)
    stroke  = estimate_stroke_height(comps)
    spacing = estimate_char_spacing(nn_graph, ori)

    print(f"[DynamicMorph] ori={ori:.1f}°, stroke={stroke:.1f}px, spacing={spacing:.1f}px")

    # 4) build line boxes
    clusters   = build_docstrum_lines(
        comps, ori, spacing, stroke,
        min_dist_factor=args.min_dist_factor,
        max_dist_factor=args.max_dist_factor,
        angle_tol=args.angle_tol
    )
    line_boxes = clusters_to_boxes(clusters, comps)
    # print(f"[DynamicMorph] detected {len(line_boxes)} lines")

    # 5) dynamic morphological merge
    para_boxes = dynamic_morphological_merge(
        (H, W),
        line_boxes,
        stroke_height=stroke,
        dynamic_pct=args.dynamic_pct,
        open_factor=args.open_factor
    )
    print(f"[DynamicMorph] merged into {len(para_boxes)} paragraph blocks")

    # 6) visualize: lines in blue, paragraphs in red
    vis = draw_boxes(gray, line_boxes, color=(255, 0,   0), thickness=1)
    vis = draw_boxes(vis,    para_boxes,  color=(0,   0, 255), thickness=2)

    if args.out:
        cv2.imwrite(args.out, vis)
        print(f"Saved visualization to {args.out}")
    else:
        cv2.imshow("DynamicMorph Blocks", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
