import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

CELL_SIZE = 8
CELL_PER_BLOCK = 2
N_BINS = 9

image = cv2.imread("../../../assets/human.jpg", cv2.IMREAD_GRAYSCALE)

features, hog_image = hog(
    image,
    orientations=N_BINS,
    pixels_per_cell=(CELL_SIZE, CELL_SIZE),
    cells_per_block=(CELL_PER_BLOCK, CELL_PER_BLOCK),
    visualize=True,
)

# 시각화를 위한 Normalize
hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Visualize the images
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Image")

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap="gray")
plt.title("HOG features")
plt.show()