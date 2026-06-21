import cv2
import numpy as np

# -------------------------------------------------------------------
def estimate_block_size(gray, detail="medium", min_block=4, max_block=64):
    """
    Returns (bw, bh) based on median edge distance, scaled by detail.
    detail: 'low' -> 2× larger blocks, 'medium' -> 1×, 'high' -> 0.5×
    """
    edges = cv2.Canny(gray, 50, 150)
    h_dists, v_dists = [], []
    for row in edges:
        cols = np.where(row)[0]
        if len(cols) >= 2: h_dists.extend(np.diff(cols))
    for col in edges.T:
        rows = np.where(col)[0]
        if len(rows) >= 2: v_dists.extend(np.diff(rows))

    med_h = np.median(h_dists) if h_dists else max_block
    med_v = np.median(v_dists) if v_dists else max_block

    # Nyquist base (half the median gap)
    bw = max(min_block, min(int(round(med_h / 2)), max_block))
    bh = max(min_block, min(int(round(med_v / 2)), max_block))

    # Detail scaling
    factor = {"low": 2.0, "medium": 1.0, "high": 0.5}[detail]
    bw = max(min_block, min(int(round(bw * factor)), max_block))
    bh = max(min_block, min(int(round(bh * factor)), max_block))
    return bw, bh

# -------------------------------------------------------------------
def pixelate(image, bw, bh, detail="medium"):
    """
    Replace each block with a single color:
      - low : pixel with smallest gradient (smooth interiors)
      - medium : mean color
      - high : pixel with largest gradient (keep edges)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)

    out = np.zeros_like(image)
    h, w = image.shape[:2]
    for y in range(0, h, bh):
        for x in range(0, w, bw):
            y2 = min(y + bh, h)
            x2 = min(x + bw, w)
            block = image[y:y2, x:x2]
            mblock = mag[y:y2, x:x2]

            if detail == "low":
                idx = np.unravel_index(np.argmin(mblock), mblock.shape)
                color = block[idx[0], idx[1]]
            elif detail == "high":
                idx = np.unravel_index(np.argmax(mblock), mblock.shape)
                color = block[idx[0], idx[1]]
            else:  # medium
                color = block.mean(axis=(0, 1)).astype(np.uint8)

            out[y:y2, x:x2] = color
    return out

# -------------------------------------------------------------------
def main():
    path = r"D:\pythonProject\Private\CUHKSZ Schoolwork\Y1T2S\ECE4512\Pixelate Everything\test_cases\van_gogh_starsky.jpg"
    img = cv2.imread(path)
    if img is None:
        print("Image not found")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    images = {}
    for detail in ("low", "medium", "high"):
        bw, bh = estimate_block_size(gray, detail=detail)
        print(f"{detail:6s}: block {bw:2d}x{bh:2d}")
        pix = pixelate(img, bw, bh, detail=detail)
        cv2.imwrite(f"output_{detail}.jpg", pix)
        images[detail] = pix

    # Quick side‑by‑side comparison
    comp = np.hstack((img, images["low"], images["medium"], images["high"]))
    cv2.imshow("Original | Low | Medium | High", comp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()