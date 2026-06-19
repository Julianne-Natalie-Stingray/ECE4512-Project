import cv2

path = r"D:\pythonProject\Private\CUHKSZ Schoolwork\Y1T2S\ECE4512\Project\MVP ver.1\test_cases\mona_lisa.jpg"

def pixelate_by_resizing(img, block_size = 5):
    ori_shape = img.shape[0], img.shape[1]
    small_size = img.shape[0] // block_size, img.shape[1] // block_size
    scaled = cv2.resize(img, small_size, interpolation = cv2.INTER_AREA)
    pixelated = cv2.resize(scaled, ori_shape, interpolation = cv2.INTER_NEAREST)
    return pixelated


def main():
    img = cv2.imread(path)
    scaled = pixelate_by_resizing(img, 10)
    cv2.imshow("Scaled", scaled)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()