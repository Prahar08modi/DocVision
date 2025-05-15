import numpy as np
import cv2
import matplotlib.pyplot as plt
from test_scripts.binarize import binarize_image
from xy_cut import find_cuts

def draw_cuts_on_image(binary_image, horizontal_cuts, vertical_cuts):
    # Convert binary to BGR for color drawing
    image_bgr = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Draw horizontal cuts (red lines)
    for y in horizontal_cuts:
        cv2.line(image_bgr, (0, y), (image_bgr.shape[1], y), (0, 0, 255), 1)

    # Draw vertical cuts (blue lines)
    for x in vertical_cuts:
        cv2.line(image_bgr, (x, 0), (x, image_bgr.shape[0]), (255, 0, 0), 1)

    return image_bgr

def visualize_cut_overlay():
    binary = binarize_image()

    # Compute projection profiles
    horizontal_proj = np.sum(binary, axis=1)
    vertical_proj = np.sum(binary, axis=0)

    # Find cuts
    horizontal_cuts = find_cuts(horizontal_proj, threshold_ratio=0.2)
    vertical_cuts = find_cuts(vertical_proj, threshold_ratio=0.05)

    # Overlay cuts on image
    overlay_img = draw_cuts_on_image(binary, horizontal_cuts, vertical_cuts)

    # Show the result
    plt.figure(figsize=(12, 10))
    plt.imshow(overlay_img)
    plt.title('Cut Lines Overlayed on Binarized Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_cut_overlay()
