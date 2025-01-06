import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm


def plot_1d_gaussian(ax, mean, std_dev):
    """
    Plot a 1D Gaussian distribution and highlight the region within ±4.
    """
    x = np.linspace(-5 * std_dev, 5 * std_dev, 500)
    y = norm.pdf(x, mean, std_dev)

    lower_bound = mean - 4 * std_dev
    upper_bound = mean + 4 * std_dev
    percentage_within_4sigma = norm.cdf(upper_bound, mean, std_dev) - norm.cdf(
        lower_bound, mean, std_dev
    )
    percentage_within_4sigma *= 100

    ax.plot(x, y, label="Gaussian Curve", color="blue")
    ax.axvline(-4 * std_dev, color="red", linestyle="--", label="-4")
    ax.axvline(4 * std_dev, color="green", linestyle="--", label="4")

    x_fill = np.linspace(lower_bound, upper_bound, 300)
    y_fill = norm.pdf(x_fill, mean, std_dev)
    ax.fill_between(
        x_fill,
        y_fill,
        color="lightblue",
        alpha=0.5,
        label=f"{percentage_within_4sigma:.2f}%",
    )

    ax.set_title("1D Gaussian Distribution")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid()


def plot_2d_gaussian(ax, mean, variance, covariance):
    """
    Plot a 2D Gaussian distribution as a 3D surface plot.
    """
    cov_matrix = [[variance, covariance], [covariance, variance]]

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    pos = np.dstack((X, Y))
    factor = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov_matrix)))
    inv_cov = np.linalg.inv(cov_matrix)
    Z = factor * np.exp(
        -0.5 * np.einsum("...i,ij,...j", pos - mean, inv_cov, pos - mean)
    )

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)

    ax.set_title("2D Gaussian Distribution")
    ax.set_zlabel("Probability Density")


fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121)
plot_1d_gaussian(ax1, mean=0, std_dev=1)

ax2 = fig.add_subplot(122, projection="3d")
plot_2d_gaussian(ax2, mean=[0, 0], variance=1, covariance=0)

plt.tight_layout()
plt.show()
