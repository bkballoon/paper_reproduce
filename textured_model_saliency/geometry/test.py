import numpy as np
import cv2

a = cv2.imread("/home/simula/Downloads/qq-files/3292492251/file_recv/AAA.png")
height, width, c = a.shape

for i in range(height):
    for j in range(width):
        pixel = a[i][j]
        if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
            a[i][j] = np.array([255, 0, 0], dtype=np.uint8)

cv2.imshow("jflksj", a)
cv2.waitKey()