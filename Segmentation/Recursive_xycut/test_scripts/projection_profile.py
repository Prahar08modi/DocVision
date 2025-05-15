import numpy as np
import matplotlib.pyplot as plt
from test_scripts.binarize import binarize_image

# Binarize the image
binary = binarize_image()

def projection_profile(binary):
    # Compute horizontal and vertical projection profiles
    horizontal_proj = np.sum(binary, axis=1)
    vertical_proj = np.sum(binary, axis=0)

    # Normalize the profiles for visualization
    horizontal_proj_norm = horizontal_proj / np.max(horizontal_proj)
    vertical_proj_norm = vertical_proj / np.max(vertical_proj)

    # Plot the projection profiles
    # Horizontal Projection
    plt.figure(figsize=(10, 4))
    plt.plot(horizontal_proj_norm)
    plt.title('Normalized Horizontal Projection Profile')
    plt.xlabel('Row Index')
    plt.ylabel('Normalized Sum')
    plt.grid(True)
    plt.show()

    # Vertical Projection
    plt.figure(figsize=(10, 4))
    plt.plot(vertical_proj_norm)
    plt.title('Normalized Vertical Projection Profile')
    plt.xlabel('Column Index')
    plt.ylabel('Normalized Sum')
    plt.grid(True)
    plt.show()

    return horizontal_proj, vertical_proj