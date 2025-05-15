import numpy as np
from scipy.spatial import KDTree
from typing import List, Dict
from Segmentation.Docstrum.docstrum_components import Component, get_connected_components
import cv2

def build_nn_graph(
    components: List[Component],
    k: int = 5
) -> Dict[int, List[Dict[str, float]]]:
    """
    For each component i, find its k nearest neighbors.
    Returns a dict: i -> [ { idx, dist, angle } ... ].
    """
    n = len(components)
    # need at least 2 points to build a graph
    if n < 2:
        return {}

    # cap k so we never ask for more neighbors than exist
    k_eff = min(k, n - 1)
    coords = np.array([c.centroid for c in components])
    tree = KDTree(coords)
    # query for self + k_eff neighbors
    dists, idxs = tree.query(coords, k_eff + 1)
    graph: Dict[int, List[Dict[str, float]]] = {}
    for i, (di, ii) in enumerate(zip(dists, idxs)):
        neighs = []
        for dist, j in zip(di[1:], ii[1: 1 + k_eff]):
            dx, dy = coords[j] - coords[i]
            angle = np.degrees(np.arctan2(dy, dx))
            neighs.append({"idx": j, "dist": float(dist), "angle": float(angle)})
        graph[i] = neighs
    return graph

def estimate_page_orientation(
    nn_graph: Dict[int, List[Dict[str, float]]]
) -> float:
    """
    Take all neighbor‐link angles, fold into [-90,90], and return the median.
    """
    angles = []
    for neighs in nn_graph.values():
        for n in neighs:
            a = n["angle"]
            # fold to [-90,90]
            a = ((a + 90) % 180) - 90
            angles.append(a)
    return float(np.median(angles))

def estimate_stroke_height(
    components: List[Component]
) -> float:
    """
    Median component height = rough stroke (char) height.
    """
    heights = [c.bbox[3] for c in components]
    return float(np.median(heights))

def estimate_char_spacing(
    nn_graph: Dict[int, List[Dict[str, float]]],
    orientation: float,
    tol_deg: float = 30.0
) -> float:
    """
    Collect neighbor distances whose angle is within ±tol_deg of orientation,
    then return the median distance.
    """
    dists = []
    for neighs in nn_graph.values():
        for n in neighs:
            a = n["angle"]
            # normalize to [-180,180]
            a = ((a + 180) % 360) - 180
            if abs(a - orientation) <= tol_deg:
                dists.append(n["dist"])
    return float(np.median(dists)) if dists else 0.0

# ——————— CLI for quick testing ———————
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 2: DocStrum Geometry Estimation"
    )
    parser.add_argument("image", help="Path to *page* image (will binarize & CC)")
    parser.add_argument("--k", type=int, default=5, help="# of NN per CC")
    parser.add_argument("--tol", type=float, default=30.0,
                        help="angle tolerance (deg) for spacing")
    args = parser.parse_args()

    # 1. load & binarize
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2. CC → components
    comps = get_connected_components(bin_img, min_area=10, max_area=5000)
    print(f"[Stage 2] → {len(comps)} components")

    # 3. NN graph
    nn = build_nn_graph(comps, k=args.k)

    # 4. estimates
    ori = estimate_page_orientation(nn)
    stroke = estimate_stroke_height(comps)
    spacing = estimate_char_spacing(nn, ori, tol_deg=args.tol)

    print(f"Estimated orientation: {ori:.2f}°")
    print(f"Estimated stroke height: {stroke:.1f}px")
    print(f"Estimated char spacing:  {spacing:.1f}px")
