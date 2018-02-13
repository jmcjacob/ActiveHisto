import cv2
from Model import Model


def heatmap(image_file):
    model = Model([15, 15, 3], 2)
    image = cv2.imread(image_file)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    border_size, count = 15 // 2, 0
    border_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_DEFAULT)
    for pi in range(0, border_image.shape[1], 5):
        for pj in range(0, border_image.shape[2], 5):
            patch = border_image[pj - border_size:pj + border_size + 1, pi - border_size:pi + border_size + 1]
            model


