import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


PATH = r"Project/test_cases/why_are_you_sitting.jpg"
OUT_PATH = r"D:\pythonProject\Private\CUHKSZ Schoolwork\Y1T2S\ECE4512\Project\palette_logic\ver.2_LAB_colors\output.jpg"

PALETTE_SIZE = 16


def derive(img, palette_size = 32):
    '''
    Accepts BGR image.
    Returns: (labels, centroids)
    Stages:
    1. Choose centroids (connect AI)
    2. CONVERT TO LAB (NEWLY ADDED)
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
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = img_lab.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters = palette_size, init = "k-means++")
    kmeans.fit(pixels)
    labelled, centroids_lab = kmeans.labels_.reshape(img.shape[:2]), kmeans.cluster_centers_.astype(np.uint8).reshape(1, -1, 3)
    centroids = cv2.cvtColor(centroids_lab,cv2.COLOR_LAB2BGR).reshape(-1, 3)
    return labelled, centroids


def visualize(color_map, colors):
    print(colors)
    image = colors[color_map].astype(np.uint8)
    cv2.imwrite(OUT_PATH, image)


def main():
    img = cv2.imread(PATH, cv2.IMREAD_COLOR_BGR)
    color_map, colors = derive(img, palette_size = PALETTE_SIZE)
    visualize(color_map, colors)


if __name__ == "__main__":
    main()