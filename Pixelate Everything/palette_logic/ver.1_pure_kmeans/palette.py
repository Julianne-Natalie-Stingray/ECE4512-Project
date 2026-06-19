import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


PATH = r"Project\palette_logic\ver.1\test_cases\van_gogh_starsky.jpg"
OUT_PATH = r"Project/palette_logic/ver.1/output.jpg"

PALETTE_SIZE = 16


def palette(img, palette_size = 32):
    '''
    Accepts BGR image.
    Returns: (labels, centroids)
    Stages:
    1. Choose centroids (connect AI)
    2. K-means
    '''

    def init_centroids(img, k):
        '''
        returns centroids
        randomized picking for now
        '''
        return np.random.Generator.choice(img, k, replace = False)


    # MAIN_BLOCK_HERE
    # start_centroids = init_centroids(img)
    # consider setting n_init = 1 when AI expert system is integrated
    pixels = img.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters = palette_size, init = "k-means++")
    kmeans.fit(pixels)
    labelled, centroids = kmeans.labels_.reshape(img.shape[:2]), kmeans.cluster_centers_
    return labelled, centroids


def visualize(color_map, colors):
    print(colors)
    image = colors[color_map].astype(np.uint8)
    cv2.imwrite(OUT_PATH, image)


def main():
    img = cv2.imread(PATH, cv2.IMREAD_COLOR_BGR)
    color_map, colors = palette(img, palette_size = PALETTE_SIZE)
    visualize(color_map, colors)


if __name__ == "__main__":
    main()