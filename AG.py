import cv2
import numpy as np
import math


def avgGradient(image):
    width = image.shape[1]
    width = width - 1
    heigt = image.shape[0]
    heigt = heigt - 1
    tmp = 0.0

    for j in range(width):
        for i in range(heigt):
            dx = float(image[i, j + 1]) - float(image[i, j])
            dy = float(image[i + 1, j]) - float(image[i, j])
            ds = math.sqrt((dx * dx + dy * dy) / 2)
            tmp += ds

    imageAG = tmp / (width * heigt)
    return imageAG


if __name__ == '__main__':
    image1 = cv2.imread('./MSRS/Fusion/train/MSRS/00054N.png', 0)
    image2 = cv2.imread('./00054N.png', 0)

    print(avgGradient(image1))
    print(avgGradient(image2))

'''
2.6979879010501238
2.6890317382301037


'''