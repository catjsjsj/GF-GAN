import cv2
import numpy as np
import math

import skimage.measure


def imageEn(image):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(image)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))

    return res

if __name__ == '__main__':
    image1 = cv2.imread('./MSRS/Fusion/train/MSRS/00054N.png',0)
    image2 = cv2.imread('./00054N.png', 0)
    print(imageEn(image1))
    print(skimage.measure.shannon_entropy(image2, base=2))
