# docstrum_morph_blocks.py

import cv2
import numpy as np
from typing import List, Tuple
import argparse

# re‑use your existing modules
from docstrum_components import get_connected_components, Component
from docstrum_geometry   import (
    build_nn_graph,
    estimate_page_orientation,
    estimate_stroke_height,
    estimate_char_spacing
)
from docstrum_lines      import (
    build_docstrum_lines,
    clusters_to_boxes
)

def morphological_merge(
    image_shape: Tuple[int,int],
    line_boxes: List[Tuple[int,int,int,int]],
    stroke_height: float,
    vert_factor: float = 3.0,
    horz_factor: float = 0.5
) -> List[Tuple[int,int,int,int]]:
    """
    Merge line boxes via morphology:
      1) Rasterize each line box into a blank mask.
      2) Dilate vertically with a tall structuring element.
      3) Connected‑component on the dilated mask → paragraph blocks.
    
    Args:
      image_shape: (H, W) of the original page.
      line_boxes: list of (x, y, w, h) for each text line.
      stroke_height: median character height (from Stage 2).
      vert_factor: height of SE = stroke_height * vert_factor.
      horz_factor: width of SE = stroke_height * horz_factor.
    
    Returns:
      List of merged paragraph boxes (x, y, w, h).
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for x, y, w, h in line_boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # build structuring element
    se_h = max(1, int(stroke_height * vert_factor))
    se_w = max(1, int(stroke_height * horz_factor))
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_w, se_h))

    # dilate to bridge lines into paragraphs
    dilated = cv2.dilate(mask, se)

    # find paragraph blocks via CC
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    blocks = []
    for lbl in range(1, n_lbl):
        x, y, w, h, area = stats[lbl]
        # optional: filter tiny/huge blocks
        blocks.append((x, y, w, h))
    return blocks

def draw_boxes(
    img: np.ndarray,
    boxes: List[Tuple[int,int,int,int]],
    color: Tuple[int,int,int],
    thickness: int = 2
) -> np.ndarray:
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    return vis

def main():
    parser = argparse.ArgumentParser(
        description="Morphological merging of DocStrum line boxes into paragraph blocks"
    )
    parser.add_argument("image", help="Path to page image")
    parser.add_argument("--min-area", type=int,   default=10)
    parser.add_argument("--max-area", type=int,   default=5000)
    parser.add_argument("--k",        type=int,   default=5)
    parser.add_argument("--angle-tol",     type=float, default=30.0)
    parser.add_argument("--min-dist-factor", type=float, default=0.5)
    parser.add_argument("--max-dist-factor", type=float, default=6.0)
    parser.add_argument("--vert-factor",   type=float, default=3.0,
                        help="dilation height = vert_factor * stroke_height")
    parser.add_argument("--horz-factor",   type=float, default=0.5,
                        help="dilation width = horz_factor * stroke_height")
    parser.add_argument("--out", help="Where to save overlay PNG")
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

    # 2) CC → components
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

    # 4) build line boxes
    clusters  = build_docstrum_lines(
        comps, ori, spacing, stroke,
        min_dist_factor=args.min_dist_factor,
        max_dist_factor=args.max_dist_factor,
        angle_tol=args.angle_tol
    )
    line_boxes = clusters_to_boxes(clusters, comps)

    # 5) morphological merge
    para_boxes = morphological_merge(
        (H, W),
        line_boxes,
        stroke_height=stroke,
        vert_factor=args.vert_factor,
        horz_factor=args.horz_factor
    )

    print(f"[Morph] → {len(line_boxes)} lines → {len(para_boxes)} paragraph blocks")

    # 6) visualize: lines in blue, paragraphs in red
    vis = draw_boxes(gray, line_boxes, color=(255,0,0), thickness=1)
    vis = draw_boxes(vis, para_boxes, color=(0,0,255), thickness=2)

    if args.out:
        cv2.imwrite(args.out, vis)
        print(f"Saved overlay to {args.out}")
    else:
        cv2.imshow("Morphological Blocks", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()