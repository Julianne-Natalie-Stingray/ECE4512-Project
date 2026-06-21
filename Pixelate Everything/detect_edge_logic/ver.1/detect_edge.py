import cv2
import numpy as np
import matplotlib.pyplot as plt


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