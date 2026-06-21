import cv2
import numpy as np

def estimate_block_size_from_edges(gray, min_block=4, max_block=64):
    edges = cv2.Canny(gray, 50, 150)
    # Horizontal distances
    h_dists = []
    for row in edges:
        cols = np.where(row)[0]
        if len(cols) >= 2:
            h_dists.extend(np.diff(cols))
    # Vertical distances
    v_dists = []
    for col in edges.T:
        rows = np.where(col)[0]
        if len(rows) >= 2:
            v_dists.extend(np.diff(rows))
    median_h = np.median(h_dists) if h_dists else max_block
    median_v = np.median(v_dists) if v_dists else max_block
    bw = int(round(median_h / 2))
    bh = int(round(median_v / 2))
    bw = max(min_block, min(bw, max_block))
    bh = max(min_block, min(bh, max_block))
    return bw, bh

def pixelate(image, block_w, block_h):
    out = np.zeros_like(image)
    h, w = image.shape[:2]
    for y in range(0, h, block_h):
        for x in range(0, w, block_w):
            y_end = min(y+block_h, h)
            x_end = min(x+block_w, w)
            block = image[y:y_end, x:x_end]
            mean_color = block.mean(axis=(0,1)).astype(np.uint8)
            out[y:y_end, x:x_end] = mean_color
    return out

def main():
    path = r"D:\pythonProject\Private\CUHKSZ Schoolwork\Y1T2S\ECE4512\Pixelate Everything\test_cases\why_are_you_sitting.jpg"  # change to your image
    out_path = r"D:\pythonProject\Private\CUHKSZ Schoolwork\Y1T2S\ECE4512\Pixelate Everything\detect_edge_logic\ver.1\output.jpg"
    img = cv2.imread(path)
    if img is None:
        print("Image not found")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw, bh = estimate_block_size_from_edges(gray)
    print(f"Estimated block size: {bw}x{bh}")
    pixelated = pixelate(img, bw, bh)
    cv2.imshow("Original", img)
    cv2.imshow("Pixelated (adaptive block size)", pixelated)
    cv2.imwrite(out_path, pixelated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()