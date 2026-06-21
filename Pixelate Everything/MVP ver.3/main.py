import cv2
import numpy as np
import matplotlib.pyplot as plt
from smart_grid import estimate_block_size_fast, estimate_block_size
from palette import derive_palette
from derive_grid import pixelate_image, derive_block_color
from sklearn.cluster import KMeans


PATH = r"D:\pythonProject\Private\CUHKSZ Schoolwork\Y1T2S\ECE4512\Pixelate Everything\test_cases"
OUT_PATH = r"D:\pythonProject\Private\CUHKSZ Schoolwork\Y1T2S\ECE4512\Pixelate Everything\MVP ver.3\output"
    

def _batch_process(input_dir: str, output_dir: str):
    import os
    from pathlib import Path

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image formats
    image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    for img_file in input_path.iterdir():
        if not img_file.is_file():
            continue
        if img_file.suffix.lower() not in image_exts:
            continue

        print(f"Processing {img_file.name} …")

        pixelated, pixelated_fast = _process_image(str(img_file))

        # Save results
        stem = img_file.stem
        cv2.imwrite(str(output_path / f"{stem}_normal.png"), pixelated)
        cv2.imwrite(str(output_path / f"{stem}_fast.png"),   pixelated_fast)

    print("Batch processing finished.")


def _process_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR_BGR)
    if img is None:
        print(f"Error: Could not read image from {path}")
        exit(1)
    recommended_size = estimate_block_size(img)
    recommended_size_fast = estimate_block_size_fast(img)
    color_map, colors = derive_palette(img, palette_size = 64)
    pixelated = pixelate_image(img, recommended_size, color_map, colors)
    pixelated_fast = pixelate_image(img, recommended_size_fast, color_map, colors)
    return pixelated, pixelated_fast


if __name__ == "__main__":
    _batch_process(PATH, OUT_PATH)