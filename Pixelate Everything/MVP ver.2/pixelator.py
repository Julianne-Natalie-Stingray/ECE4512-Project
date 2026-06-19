import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r"D:\pythonProject\Private\CUHKSZ Schoolwork\Y1T2S\ECE4512\Project\MVP ver.1\test_cases\albert.jpg"
outpath = r"D:\pythonProject\Private\CUHKSZ Schoolwork\Y1T2S\ECE4512\Project\MVP ver.2\output.jpg"

BLOCK_SIZE = 10
EDGE_THRESH = 0.3
EDGE_COLOR = [0, 0, 0] # default black


def derive_block_color(block):
    avg = np.average(block, axis=(0, 1))
    return np.clip(avg, 0, 255).astype(np.uint8)


def pixelate_image(img, block_size=5):
    h, w = img.shape[:2]
    new_w = w // block_size
    new_h = h // block_size
    resized_w = new_w * block_size
    resized_h = new_h * block_size

    resized = cv2.resize(img, (resized_w, resized_h), interpolation = cv2.INTER_NEAREST)
    
    pixelated = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    for row in range(new_h):
        for col in range(new_w):
            block = resized[row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size]
            pixelated[row][col] = derive_block_color(block)

    return pixelated


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


def pixelate_edge_aware(img):
    # pixelate by simple resizing
    # block_size_w, block_size_h = adaptive_block_size(img)
    pixelated = pixelate_image(img, BLOCK_SIZE)

    # derive edge map
    grey = cv2.cvtColor(pixelated, cv2.COLOR_BGR2GRAY)
    magnitude, _ = sobel_edge_map(grey)
    print(magnitude)

    # edge_thresh = adaptive_edge_thresh(magnitude)
    
    edge_mask = magnitude > EDGE_THRESH

    # sketch edges
    # edge_color = edge_interpolate_color()
    
    sketched = np.copy(pixelated)
    sketched[edge_mask] = EDGE_COLOR

    return sketched


if __name__ == "__main__":
    

    img = cv2.imread(path)
    if img is None:
        print("Error loading image")
    else:
        result = pixelate_edge_aware(img)

        cv2.imshow("Original", img)
        cv2.imwrite(outpath, result)
        cv2.imshow("Edge-aware pixelated", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()