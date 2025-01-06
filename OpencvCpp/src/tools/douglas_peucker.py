import numpy as np
import matplotlib.pyplot as plt
import cv2


def perpendicular_distance(point, line_start, line_end):
    """
    Calculate the perpendicular distance of a point from a line segment.
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Area of triangle formula and line length
    area = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return area / line_length if line_length != 0 else 0


def douglas_peucker(points, epsilon):
    """
    Simplify a curve using the Douglas-Peucker algorithm.

    Args:
        points (list of tuple): List of (x, y) points representing the curve.
        epsilon (float): Distance threshold for retaining points.

    Returns:
        list of tuple: Simplified curve as a list of points.
    """
    if len(points) < 2:
        return points

    start_point = points[0]
    end_point = points[-1]

    max_distance = 0
    index_of_farthest = 0

    for i in range(1, len(points) - 1):
        distance = perpendicular_distance(points[i], start_point, end_point)
        if distance > max_distance:
            max_distance = distance
            index_of_farthest = i

    # If the maximum distance exceeds the threshold, keep the farthest point
    if max_distance > epsilon:
        # Recursively simplify the two sub-curves
        left_segment = douglas_peucker(points[: index_of_farthest + 1], epsilon)
        right_segment = douglas_peucker(points[index_of_farthest:], epsilon)

        return left_segment + right_segment
    else:
        return [start_point, end_point]


image_size = 200
binary_image = np.zeros((image_size, image_size), dtype=np.uint8)

# Draw random shapes
np.random.seed(0)
for _ in range(5):
    center = tuple(np.random.randint(20, image_size - 20, 2))
    axes = tuple(np.random.randint(10, 50, 2))
    angle = np.random.randint(0, 180)
    cv2.ellipse(binary_image, center, axes, angle, 0, 360, 255, -1)


contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
epsilon = 5.0
simplified_contours = [
    douglas_peucker(contour.squeeze(1).tolist(), epsilon) for contour in contours
]
output_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)  # Original contours

for contour in simplified_contours:
    for i in range(len(contour) - 1):
        cv2.line(
            output_image, tuple(contour[i]), tuple(contour[i + 1]), (0, 0, 255), 2
        )  # Simplified contours


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Binary Image")
plt.imshow(binary_image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Contours")
plt.imshow(output_image[..., ::-1])  # BGR to RGB
plt.show()
