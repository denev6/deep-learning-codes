import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../../assets/object4.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError()

hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))
total_pixels = image.size

sum_all = np.dot(np.arange(256), hist)
# sum_bg: Cumulative sum for background
# w_bg: Weight of background
# w_fg: Weight of foreground
sum_bg, w_bg, w_fg = 0, 0, 0
max_variance, threshold = 0, 0

for bin in range(256):
    w_bg += hist[bin]
    if w_bg == 0:
        continue

    w_fg = total_pixels - w_bg
    if w_fg == 0:
        break

    sum_bg += bin * hist[bin]
    mean_bg = sum_bg / w_bg
    mean_fg = (sum_all - sum_bg) / w_fg

    variance = w_bg * w_fg * (mean_bg - mean_fg) ** 2

    # Find the maximum variance
    if variance > max_variance:
        max_variance = variance
        threshold = bin

_, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# Visualization
plt.figure(figsize=(12, 6))

# Plot the histogram
plt.subplot(1, 3, 1)
plt.plot(hist, color="black")
plt.axvline(x=threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
plt.title("Max Variance")
plt.legend()

plt.subplot(1, 3, 2)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary_image, cmap="gray")
plt.title("Thresholded")
plt.axis("off")

plt.tight_layout()
plt.show()
