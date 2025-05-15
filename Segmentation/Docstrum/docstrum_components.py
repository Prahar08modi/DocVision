import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Component:
    label: int
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    area: int
    centroid: Tuple[float, float]

def get_connected_components(
    binary_img: np.ndarray,
    min_area: int = 1,
    max_area: int = 10_000_000
) -> List[Component]:
    """
    Find connected components in `binary_img` and filter by area.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )

    # print(f"[debug] raw labels: {n_labels-1}")   # how many objects before filtering

    comps: List[Component] = []
    for lbl in range(1, n_labels):
        x, y, w, h, area = stats[lbl]
        cx, cy = centroids[lbl]
        if min_area <= area <= max_area:
            comps.append(Component(
                label=lbl,
                bbox=(x, y, w, h),
                area=area,
                centroid=(cx, cy)
            ))
    return comps

def draw_centroids(
    orig_img: np.ndarray,
    components: List[Component],
    radius: int = 3
) -> np.ndarray:
    """
    Overlay component centroids on the image.
    """
    vis = orig_img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for comp in components:
        cx, cy = comp.centroid
        cv2.circle(vis, (int(cx), int(cy)), radius, (0, 0, 255), -1)
    return vis

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 1: Connected Components + Centroid Overlay"
    )
    parser.add_argument("image", help="Path to any page image (will binarize)")
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=10_000_000)
    parser.add_argument("--out", help="Path to save overlay (PNG)")
    args = parser.parse_args()

    # ——— binarize (inverted) ———
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(args.image)
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    comps = get_connected_components(
        bin_img,
        min_area=args.min_area,
        max_area=args.max_area
    )
    print(f"[Stage 1] → {len(comps)} components after filtering")

    overlay = draw_centroids(bin_img, comps)
    if args.out:
        cv2.imwrite(args.out, overlay)
        print(f"Overlay saved to {args.out}")
    else:
        cv2.imshow("Centroids", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()