import numpy as np
import matplotlib.pyplot as plt
from binarize import binarize_image
from xy_cut import find_cuts

def visualize_projection_with_cuts(binary_image):
    # Compute projection profiles
    horizontal_proj = np.sum(binary_image, axis=1)
    vertical_proj = np.sum(binary_image, axis=0)

    # Normalize for plotting
    horizontal_proj_norm = horizontal_proj / np.max(horizontal_proj)
    vertical_proj_norm = vertical_proj / np.max(vertical_proj)

    # Detect cut positions
    horizontal_cuts = find_cuts(horizontal_proj, threshold_ratio=0.2)
    vertical_cuts = find_cuts(vertical_proj, threshold_ratio=0.05)

    # Plot Horizontal Projection
    plt.figure(figsize=(12, 4))
    plt.plot(horizontal_proj_norm, label='Horizontal Projection')
    for cut in horizontal_cuts:
        plt.axvline(x=cut, color='red', linestyle='--', linewidth=0.8)
    plt.title('Horizontal Projection Profile with Cut Lines')
    plt.xlabel('Row Index')
    plt.ylabel('Normalized Sum')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Vertical Projection
    plt.figure(figsize=(12, 4))
    plt.plot(vertical_proj_norm, label='Vertical Projection')
    for cut in vertical_cuts:
        plt.axvline(x=cut, color='red', linestyle='--', linewidth=0.8)
    plt.title('Vertical Projection Profile with Cut Lines')
    plt.xlabel('Column Index')
    plt.ylabel('Normalized Sum')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    binary = binarize_image()
    visualize_projection_with_cuts(binary)
