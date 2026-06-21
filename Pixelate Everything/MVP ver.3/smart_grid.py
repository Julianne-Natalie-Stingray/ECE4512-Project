"""
Smart Grid Estimation for Natural Images
=========================================
Automatically determines the optimal pixel block size for converting
a natural image into pixel art.

FFT-based grid detection (e.g. perfectPixel) only works on images that
are already approximately pixelated. For natural photos, we need a
content-aware approach.

Methods implemented:
1. Edge density analysis — measures image detail level
2. Multi-scale structural similarity (SSIM) — evaluates quality at each block size
3. Gradient-based elbow detection — finds optimal trade-off point

Usage:
    from smart_grid import estimate_block_size
    block_size = estimate_block_size(image_bgr)
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _compute_edge_density(gray: np.ndarray) -> float:
    """
    Fraction of pixels that are 'edge' pixels (Canny).
    Higher density => more detail => smaller block size needed.
    """
    edges = cv2.Canny(gray, 50, 150)
    return np.count_nonzero(edges) / edges.size


def _compute_local_variance(gray: np.ndarray, ksize: int = 5) -> float:
    """
    Mean local variance — measures texture complexity.
    """
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (ksize, ksize))
    sq_mean = cv2.blur(gray_f ** 2, (ksize, ksize))
    variance = sq_mean - mean ** 2
    return float(np.mean(variance))


def _pixelate(img: np.ndarray, block_size: int) -> np.ndarray:
    """
    Simple block-average pixelation -> resize back to original dimensions.
    """
    h, w = img.shape[:2]
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _ssim_gray(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Simplified SSIM between two grayscale images (same size).
    Reference: Wang et al., "Image Quality Assessment: From Error Visibility
    to Structural Similarity", IEEE TIP, 2004.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


# ---------------------------------------------------------------------------
# Multi-scale SSIM evaluation
# ---------------------------------------------------------------------------

def _evaluate_block_sizes(img: np.ndarray, candidates: list) -> list:
    """
    For each candidate block size, pixelate the image and compute SSIM
    against the original.  Returns list of (block_size, ssim) tuples.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    results = []
    for bs in candidates:
        pix = _pixelate(img, bs)
        pix_gray = cv2.cvtColor(pix, cv2.COLOR_BGR2GRAY) if pix.ndim == 3 else pix
        score = _ssim_gray(gray, pix_gray)
        results.append((bs, score))
    return results


def _find_elbow(scores: list) -> int:
    """
    Given (block_size, ssim) pairs sorted by block_size, find the 'elbow'
    where SSIM drop rate accelerates.  Returns the block size just before
    quality degrades significantly.

    Uses the maximum-distance-to-line method (Kneedle-like).
    """
    if len(scores) <= 2:
        return scores[0][0]

    xs = np.array([s[0] for s in scores], dtype=np.float64)
    ys = np.array([s[1] for s in scores], dtype=np.float64)

    # Normalise to [0, 1]
    xs_n = (xs - xs[0]) / (xs[-1] - xs[0] + 1e-12)
    ys_n = (ys - ys[-1]) / (ys[0] - ys[-1] + 1e-12)

    # Line from first point to last point
    # Distance of each point from this line
    # Line: from (0, 1) to (1, 0) -> x + y - 1 = 0
    # dist = |x_i + y_i - 1| / sqrt(2)
    distances = np.abs(xs_n + ys_n - 1.0)

    elbow_idx = int(np.argmax(distances))
    return int(xs[elbow_idx])


# ---------------------------------------------------------------------------
# Heuristic bounds
# ---------------------------------------------------------------------------

def _heuristic_range(img: np.ndarray, edge_density: float,
                     local_var: float) -> tuple:
    """
    Determine a reasonable search range for block sizes based on
    image dimensions and content complexity.

    Returns (min_block, max_block).
    """
    h, w = img.shape[:2]
    short_side = min(h, w)

    # Target output roughly 16-256 pixels on shortest side
    abs_min = max(2, short_side // 256)
    abs_max = max(abs_min + 1, short_side // 16)

    # Adjust based on edge density (more edges -> tighter upper bound)
    if edge_density > 0.15:          # very detailed
        abs_max = max(abs_min + 1, int(abs_max * 0.5))
    elif edge_density > 0.08:        # moderately detailed
        abs_max = max(abs_min + 1, int(abs_max * 0.7))

    # Clamp
    abs_min = max(2, abs_min)
    abs_max = min(abs_max, short_side // 4)
    abs_max = max(abs_min + 1, abs_max)

    return abs_min, abs_max


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_block_size(img: np.ndarray, *,
                        min_block=None,
                        max_block=None,
                        num_candidates: int = 20,
                        debug: bool = False) -> int:
    """
    Automatically estimate the optimal pixel block size for a natural image.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format (as read by cv2.imread).
    min_block : int, optional
        Minimum block size to consider.  Auto-detected if None.
    max_block : int, optional
        Maximum block size to consider.  Auto-detected if None.
    num_candidates : int
        Number of candidate block sizes to evaluate.
    debug : bool
        If True, print intermediate results and show plots.

    Returns
    -------
    int
        Recommended block size.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # Step 1: content analysis
    edge_density = _compute_edge_density(gray)
    local_var = _compute_local_variance(gray)

    # Step 2: determine search range
    lo, hi = _heuristic_range(img, edge_density, local_var)
    if min_block is not None:
        lo = max(2, min_block)
    if max_block is not None:
        hi = max(lo + 1, max_block)

    # Build candidate list (roughly log-spaced for efficiency)
    candidates = sorted(set(
        np.unique(np.geomspace(lo, hi, num=num_candidates).astype(int)).tolist()
    ))
    if len(candidates) < 3:
        candidates = list(range(lo, hi + 1))

    # Step 3: multi-scale SSIM evaluation
    scores = _evaluate_block_sizes(img, candidates)

    # Step 4: find the elbow
    best_block = _find_elbow(scores)

    if debug:
        _debug_plot(img, scores, best_block, edge_density, local_var)

    return best_block


def estimate_block_size_fast(img: np.ndarray, *,
                             target_short_side: int = 64,
                             debug: bool = False) -> int:
    """
    A faster, simpler heuristic:
      block_size ~ short_side / target_short_side
    adjusted by edge density.

    Useful when speed matters more than precision.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR).
    target_short_side : int
        Desired shortest dimension of the pixelated output (default 64).
    debug : bool
        If True, print the reasoning.

    Returns
    -------
    int
        Recommended block size.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    h, w = img.shape[:2]
    short_side = min(h, w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    edge_density = _compute_edge_density(gray)

    # Base estimate
    base = short_side / target_short_side

    # Adjust: more edges -> smaller blocks to preserve detail
    if edge_density > 0.15:
        factor = 0.6
    elif edge_density > 0.08:
        factor = 0.8
    else:
        factor = 1.0

    block_size = max(2, int(round(base * factor)))

    if debug:
        print(f"[fast] short_side={short_side}, target={target_short_side}")
        print(f"[fast] edge_density={edge_density:.4f}, factor={factor}")
        print(f"[fast] block_size={block_size}")

    return block_size


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def _debug_plot(img, scores, best_block, edge_density, local_var):
    """Show SSIM curve, elbow point, and pixelated preview."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[debug] matplotlib not available; skipping plot.")
        return

    block_sizes = [s[0] for s in scores]
    ssim_vals = [s[1] for s in scores]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) SSIM curve
    axes[0].plot(block_sizes, ssim_vals, "o-", color="steelblue")
    axes[0].axvline(best_block, color="red", linestyle="--",
                    label=f"chosen = {best_block}")
    axes[0].set_xlabel("Block Size")
    axes[0].set_ylabel("SSIM")
    axes[0].set_title("Multi-scale SSIM Evaluation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2) Original
    axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Original Image")
    axes[1].axis("off")

    # 3) Pixelated preview
    pix = _pixelate(img, best_block)
    axes[2].imshow(cv2.cvtColor(pix, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Pixelated (block={best_block})")
    axes[2].axis("off")

    fig.suptitle(f"Edge density = {edge_density:.4f}  |  "
                 f"Local variance = {local_var:.1f}", fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smart_grid.py <image_path> [--debug]")
        sys.exit(1)

    path = sys.argv[1]
    debug = "--debug" in sys.argv

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: cannot load image '{path}'")
        sys.exit(1)

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    bs = estimate_block_size(img, debug=debug)
    print(f"[SSIM-elbow method] Recommended block size: {bs}")

    bs_fast = estimate_block_size_fast(img, debug=debug)
    print(f"[fast heuristic]    Recommended block size: {bs_fast}")
