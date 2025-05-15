import numpy as np
import matplotlib.pyplot as plt

def recursive_xy_cut(binary_image, x_start=0, y_start=0, min_block_size=70, depth=0, max_depth=100):
    """
    Recursively applies the XY Cut algorithm to segment the binary image into rectangular blocks.

    Parameters:
    - binary_image: 2D numpy array of the binarized image.
    - x_start, y_start: Top-left coordinates of the current block within the original image.
    - min_block_size: Minimum size (in pixels) of a block to consider for further splitting.
    - depth: Current depth of recursion.
    - max_depth: Maximum allowed depth of recursion.

    Returns:
    - List of bounding boxes for the segmented blocks. Each bounding box is represented as (x, y, width, height).
    """
    height, width = binary_image.shape

    # Base case: stop recursion if block is too small or maximum depth reached
    if width < min_block_size or height < min_block_size or depth >= max_depth:
        return [(x_start, y_start, width, height)]

    # Compute projection profiles
    horizontal_proj = np.sum(binary_image, axis=1)
    vertical_proj = np.sum(binary_image, axis=0)

    # Identify potential split points
    horizontal_cuts = find_cuts(horizontal_proj)
    vertical_cuts = find_cuts(vertical_proj)

    # Determine the best split direction
    if not horizontal_cuts and not vertical_cuts:
        return [(x_start, y_start, width, height)]

    if vertical_cuts and (not horizontal_cuts or len(vertical_cuts) >= len(horizontal_cuts)):
        # Perform vertical split
        cut = select_best_cut(vertical_proj, vertical_cuts)
        left = binary_image[:, :cut]
        right = binary_image[:, cut:]
        boxes_left = recursive_xy_cut(left, x_start, y_start, min_block_size, depth + 1, max_depth)
        boxes_right = recursive_xy_cut(right, x_start + cut, y_start, min_block_size, depth + 1, max_depth)
        return boxes_left + boxes_right
    else:
        # Perform horizontal split
        cut = select_best_cut(horizontal_proj, horizontal_cuts)
        top = binary_image[:cut, :]
        bottom = binary_image[cut:, :]
        boxes_top = recursive_xy_cut(top, x_start, y_start, min_block_size, depth + 1, max_depth)
        boxes_bottom = recursive_xy_cut(bottom, x_start, y_start + cut, min_block_size, depth + 1, max_depth)
        return boxes_top + boxes_bottom

def find_cuts(projection: np.ndarray,
              threshold_ratio: float = 0.1,
              min_gap: int = 50) -> list[int]:
    """
    Finds cut points at the *lowest* projection within each sufficiently wide gap.

    Args:
      projection: 1D array (sum along rows or columns).
      threshold_ratio: gaps where proj < max(proj)*threshold_ratio
      min_gap: ignore gaps narrower than this many pixels

    Returns:
      List of cut indices (the true minima inside each gap).
    """
    thresh = projection.max() * threshold_ratio
    cuts = []
    in_gap = False
    start = 0

    for i, v in enumerate(projection):
        if v < thresh and not in_gap:
            in_gap, start = True, i
        elif v >= thresh and in_gap:
            in_gap = False
            end = i
            width = end - start
            if width >= min_gap:
                # locate the deepest valley in this gap
                local = projection[start:end]
                valley = int(start + np.argmin(local))
                cuts.append(valley)
    return cuts

def select_best_cut(projection, cuts):
    """
    Selects the best cut point among potential cuts based on the minimum projection value.

    Parameters:
    - projection: 1D numpy array representing the projection profile.
    - cuts: List of indices representing potential cut points.

    Returns:
    - Index of the best cut point.
    """
    min_value = float('inf')
    best_cut = cuts[0]
    for cut in cuts:
        if projection[cut] < min_value:
            min_value = projection[cut]
            best_cut = cut
    return best_cut
