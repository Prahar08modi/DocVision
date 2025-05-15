import matplotlib.pyplot as plt
import matplotlib.patches as patches
from test_scripts.binarize import binarize_image
from xy_cut import recursive_xy_cut

def test_recursive_xy_cut():
    binary_image = binarize_image()
    boxes = recursive_xy_cut(binary_image)

    # Display the binary image with bounding boxes
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(binary_image, cmap='gray')
    for (x, y, w, h) in boxes:
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    plt.title('Recursive XY Cut Segmentation')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_recursive_xy_cut()
