import cv2
import numpy as np
import matplotlib.pyplot as plt


def derive_block_color(block, colors):
    avg = np.average(block, axis=(0, 1))
    avg = np.clip(avg, 0, 255).astype(np.float32)
    diff = colors.astype(np.float32) - avg
    dist = np.linalg.norm(diff, axis=1)
    color_index = np.argmin(dist)
    return colors[color_index]


def pixelate_image(img, block_size, color_map, colors):
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
            pixelated[row][col] = derive_block_color(block, colors)

    return pixelated