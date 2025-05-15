import numpy as np
import cv2
import matplotlib.pyplot as plt
from test_scripts.binarize import binarize_image
from xy_cut import find_cuts

def smooth(proj: np.ndarray, k: int) -> np.ndarray:
    """Simple moving‐average smoother."""
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(proj, kernel, mode='same')

def draw_cuts_on_image(binary: np.ndarray,
                       h_cuts: list[int],
                       v_cuts: list[int]) -> np.ndarray:
    img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for y in h_cuts:
        cv2.line(img, (0, y), (img.shape[1], y), (0, 0, 255), 1)   # red
    for x in v_cuts:
        cv2.line(img, (x, 0), (x, img.shape[0]), (255, 0, 0), 1)   # blue
    return img

def visualize_cut_overlay():
    # 1. Binarize
    binary = binarize_image()

    # 2. Raw projections
    h_proj = np.sum(binary, axis=1)
    v_proj = np.sum(binary, axis=0)

    # 3. Smooth
    h_proj_s = smooth(h_proj, k=25)
    v_proj_s = smooth(v_proj, k=15)

    # 4. Find cuts with min_gap
    h_cuts = find_cuts(h_proj_s, threshold_ratio=0.2, min_gap=18)
    v_cuts = find_cuts(v_proj_s, threshold_ratio=0.05, min_gap=15)

    # 5. Overlay & display
    overlay = draw_cuts_on_image(binary, h_cuts, v_cuts)
    plt.figure(figsize=(12,10))
    plt.imshow(overlay)
    plt.title("Cut Lines with Smoothing + Min‑Gap")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_cut_overlay()
