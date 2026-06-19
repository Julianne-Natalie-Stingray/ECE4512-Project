import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r"D:\pythonProject\Private\CUHKSZ Schoolwork\Y1T2S\ECE4512\Project\MVP ver.1\test_cases\hatsune_miku.jpg"

def sobel_edge_map(image, ksize=3):
    """
    Returns gradient magnitude and direction from a grayscale image.
    """
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mag_norm = magnitude / np.max(magnitude)
    direction = np.arctan2(grad_y, grad_x)   # radians, [-π, π]
    return mag_norm, direction


def main():
    # Load grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image.")
        return

    magnitude, direction = sobel_edge_map(img)

    # Normalize magnitude to [0, 1] for display
    

    # ---- Visualization ----
    plt.figure(figsize=(18, 6))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # Gradient Magnitude
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude, cmap="gray")
    plt.title("Gradient Magnitude (Sobel)")
    plt.axis("off")
    # Colorbar to show relative strength
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Magnitude")

    # Gradient Direction
    # Map direction from [-π, π] to [0, 1] for HSV display
    dir_norm = (direction + np.pi) / (2 * np.pi)
    plt.subplot(1, 3, 3)
    plt.imshow(dir_norm, cmap="hsv")
    plt.title("Gradient Direction")
    plt.axis("off")
    cbar2 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar2.set_label("Angle (radians)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()