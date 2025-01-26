import cv2
import numpy as np
import matplotlib.pyplot as plt


def sqrt_gamma_compression(image, gamma=0.5):
    normalized_image = image / 255.0
    compressed_image = np.power(normalized_image, gamma)
    compressed_image = (compressed_image * 255).astype(np.uint8)

    return compressed_image


def gradients(image):
    # [-1 0 1] 마스크 적용
    mask_x = np.array([[-1, 0, 1]])
    mask_y = np.array([[-1], [0], [1]])
    gx = cv2.filter2D(image, cv2.CV_64F, mask_x)
    gy = cv2.filter2D(image, cv2.CV_64F, mask_y)

    # Gradients 계산
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (np.arctan2(gy, gx) * (180 / np.pi)) % 180
    return magnitude, orientation


def gaussian_filter(magnitude, cell_size, block_size, sigma):
    h, w = magnitude.shape
    block_stride = cell_size * block_size  # block 단위로 계산
    filtered_magnitude = np.zeros_like(magnitude)

    for i in range(0, h - block_stride + 1, cell_size):
        for j in range(0, w - block_stride + 1, cell_size):
            # 가우시안 필터 적용
            block = magnitude[i : i + block_stride, j : j + block_stride]
            gaussian_block = cv2.GaussianBlur(block, (0, 0), sigma)
            filtered_magnitude[i : i + block_stride, j : j + block_stride] = (
                gaussian_block
            )

    return filtered_magnitude


def vote_histogram(magnitude, orientation, cell_size, n_bins, degree):
    h, w = magnitude.shape
    n_cell_rows, n_cell_cols = h // cell_size, w // cell_size
    hist = np.zeros((n_cell_rows, n_cell_cols, n_bins))
    bin_width = degree // n_bins

    for row in range(n_cell_rows):
        for col in range(n_cell_cols):
            # Cell 범위
            row_start = row * cell_size
            row_end = (row + 1) * cell_size
            col_start = col * cell_size
            col_end = (col + 1) * cell_size

            cell_magnitude = magnitude[row_start:row_end, col_start:col_end].ravel()
            cell_orientation = orientation[row_start:row_end, col_start:col_end].ravel()

            for mag, ori in zip(cell_magnitude, cell_orientation):
                # Bilinear interpolation
                prev_bin = int(ori // bin_width)
                next_bin = (prev_bin + 1) % n_bins
                bin_fraction = (ori % bin_width) / bin_width
                # Vote
                hist[row, col, prev_bin] += mag * (1 - bin_fraction)
                hist[row, col, next_bin] += mag * bin_fraction

    return hist


def normalize(hist, block_size, stride):
    rows, cols, _ = hist.shape
    norm_hist = np.zeros_like(hist)

    for row in range(0, rows - block_size + 1, stride):
        for col in range(0, cols - block_size + 1, stride):
            block_magnitude = hist[row : row + block_size, col : col + block_size, :]
            norm = np.sqrt(np.sum(block_magnitude.ravel() ** 2) + 1e-6)
            norm_block = block_magnitude / norm
            norm_hist[row : row + block_size, col : col + block_size, :] = norm_block

    return norm_hist


def visualize(image):
    height, width = image.shape
    plt.figure(figsize=(width, height))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


CELL_SIZE = 8  # Cell: 8 x 8 pixel
BLOCK_SIZE = 2  # Block: 16 x 16 pixel
BLOCK_STRIDE = 1  # 4-fold coverage
STD = 8  # Block_width * 0.5
N_BINS = 9
UNSIGNED = 180

# 이미지 준비
image = cv2.imread("../../../assets/human.jpg", cv2.IMREAD_GRAYSCALE)
visualize(image)

# Gradient 크기 및 방향
magnitude, orientation = gradients(image)
visualize(magnitude)

# 가우시안 필터 적용
filtered_magnitude = gaussian_filter(magnitude, CELL_SIZE, BLOCK_SIZE, STD)
visualize(filtered_magnitude)

# Histogram 생성
hist = vote_histogram(filtered_magnitude, orientation, CELL_SIZE, N_BINS, UNSIGNED)

# Block 정규화
norm_hist = normalize(hist, BLOCK_SIZE, BLOCK_STRIDE)
