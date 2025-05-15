import cv2
import numpy as np
from scipy.spatial import KDTree
import random

from Segmentation.Docstrum.docstrum_components import Component, get_connected_components
from Segmentation.Docstrum.docstrum_geometry   import (
    build_nn_graph,
    estimate_page_orientation,
    estimate_stroke_height,
    estimate_char_spacing
)

def build_docstrum_lines(
    components: list[Component],
    orientation: float,
    char_spacing: float,
    stroke_height: float,
    min_dist_factor: float = 0.5,
    max_dist_factor: float = 6.0,
    angle_tol: float = 30.0
) -> list[list[int]]:
    """
    Region‑growing clustering of CCs into horizontal text lines.
    Returns a list of clusters, each as a list of component indices.
    """
    coords = np.array([c.centroid for c in components])
    tree   = KDTree(coords)

    min_d = char_spacing * min_dist_factor
    max_d = char_spacing * max_dist_factor
    rad   = np.deg2rad(orientation)

    assigned = set()
    clusters = []

    for i in range(len(components)):
        if i in assigned:
            continue

        # start new cluster
        cluster = {i}
        queue   = [i]

        while queue:
            k = queue.pop(0)
            # find all neighbors within max_d
            neighs = tree.query_ball_point(coords[k], r=max_d)
            for j in neighs:
                if j in cluster:
                    continue

                dx, dy = coords[j] - coords[k]
                # project onto text direction / perpendicular
                along = dx * np.cos(rad) + dy * np.sin(rad)
                perp  = -dx * np.sin(rad) + dy * np.cos(rad)
                # absolute angle between cc‑pair
                angle = (np.degrees(np.arctan2(dy, dx)) + 180) % 360 - 180

                if not (min_d <= along <= max_d):
                    continue
                if abs(perp) > stroke_height * 0.5:
                    continue
                if abs(angle - orientation) > angle_tol:
                    continue

                cluster.add(j)
                queue.append(j)

        assigned.update(cluster)
        clusters.append(sorted(cluster))

    return clusters

def clusters_to_boxes(
    clusters: list[list[int]],
    components: list[Component]
) -> list[tuple[int,int,int,int]]:
    """
    Convert each cluster of indices into an axis‑aligned bbox.
    """
    boxes = []
    for cl in clusters:
        xs = [components[i].bbox[0] for i in cl]
        ys = [components[i].bbox[1] for i in cl]
        ws = [components[i].bbox[2] for i in cl]
        hs = [components[i].bbox[3] for i in cl]

        x0 = min(xs)
        y0 = min(ys)
        x1 = max(x + w for x, w in zip(xs, ws))
        y1 = max(y + h for y, h in zip(ys, hs))

        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes

def draw_line_boxes(
    img: np.ndarray,
    boxes: list[tuple[int,int,int,int]]
) -> np.ndarray:
    """
    Overlay each line‑box in a random color.
    """
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for (x, y, w, h) in boxes:
        color = tuple(random.randint(0, 255) for _ in range(3))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

    return vis

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 3: DocStrum Line Building"
    )
    parser.add_argument("image", help="Path to page image (will binarize)")
    parser.add_argument("--min-dist-factor", type=float, default=0.5,
                        help="min factor of char_spacing for clustering")
    parser.add_argument("--max-dist-factor", type=float, default=6.0,
                        help="max factor of char_spacing for clustering")
    parser.add_argument("--min-area", type=int,   default=10)
    parser.add_argument("--max-area", type=int,   default=5000)
    parser.add_argument("--k",        type=int,   default=5,
                        help="# of NN for geometry stage")
    parser.add_argument("--angle-tol", type=float, default=30.0,
                        help="° tolerance for line‐orientation")
    parser.add_argument("--out",      help="Path to save overlay PNG")
    args = parser.parse_args()

    # 1) load & binarize
    gray   = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
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

    # 3) reuse Stage 2 to get orientation, spacing, stroke
    nn_graph = build_nn_graph(comps, k=args.k)
    ori       = estimate_page_orientation(nn_graph)
    stroke    = estimate_stroke_height(comps)
    spacing   = estimate_char_spacing(nn_graph, ori)

    print(f"[Stage 3] lines on {len(comps)} CCs → ori={ori:.1f}°, "
          f"stroke={stroke:.1f}px, spacing={spacing:.1f}px")

    # 4) build clusters & boxes
    clusters = build_docstrum_lines(
        comps, ori, spacing, stroke,
        min_dist_factor=args.min_dist_factor,
        max_dist_factor=args.max_dist_factor,
        angle_tol=args.angle_tol
    )
    boxes = clusters_to_boxes(clusters, comps)
    print(f"[Stage 3] → {len(boxes)} line boxes detected")

    # 5) draw & save/show
    vis = draw_line_boxes(gray, boxes)
    if args.out:
        cv2.imwrite(args.out, vis)
        print("Overlay saved to", args.out)
    else:
        cv2.imshow("DocStrum Lines", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
