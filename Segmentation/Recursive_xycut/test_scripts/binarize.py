import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarize_image():
    # Load the image
    image_path = '../../../test_images/0a2c475664e7e18068424c8d29e5d819158494fa9b20bf3375cf31b3f17d93cd.png'  
    # image_path = "../../../test_images/0a3bc6f54adeedfb7b60678a83a89bb0f4d0135dc26a7a8d89a3ae2e3ccbf98d.png"
    # image_path = "../../../test_images/fff0c2168b8f7612fd591c56990205b50d2f2fd13022f1217571a87f19031f68.png"
    # image_path = "../../../test_images/ffa3355ed2e37823c5c2d861156f758f64308edc253476af9281454f930d02f6.png"
    # image_path = "../../../test_images/0a4ca7c1c65a7f02689468ee1cad03bbd93bfc8ad454456a52352cb449a5b1f6.png"
    # image_path = "../../../test_images/ffefa122538e3c98ac564fcf501b27531e5e43d9e884c8ebbc7fba0025f9edfc.png"
    # image_path = "../../../test_images/ffd7dc800865513dbed6e4aaf17bd96275c89149171b4026c3c7085b407e0424.png"
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image using Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # # Display the binary image
    # plt.figure(figsize=(10, 8))
    # plt.imshow(binary, cmap='gray')
    # plt.title('Binarized Image')
    # plt.axis('off')
    # plt.show()

    return binary

binarize_image()