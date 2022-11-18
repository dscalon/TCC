import math
import time
import numpy as np
import cv2
from Bezier import Bezier

# myColors = [[0, 179, 0, 135, 0, 45, "Black"],
#             [0, 10, 154, 194, 112, 182, "Red"],
#             [102, 117, 110, 184, 86, 160, "Blue"],
#             [18, 85, 136, 229, 124, 227, "Yellow"]]

# myColors = [[0, 179, 0, 135, 0, 45, "Black"],
#             [126, 179, 0, 255, 0, 255, "Red"],
#             [30, 130, 0, 255, 0, 255, "Blue"],
#             [20, 60, 0, 255, 0, 255, "Yellow"]]

Red = [123, 179, 220, 225, 230, 240, "Red"]
Blue = [0, 173, 0, 255, 194, 245, "Blue"]
Yellow = [12, 179, 0, 255, 255, 255, "Yellow"]

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver



while True:
   #
   # success, img = cap.read()
   # if success == False:
   #     cap = cv2.VideoCapture("Images\Input15.gif")
   #     success, img = cap.read()
   # #
   #  #img = img[0:682, 160:1119]


    img = cv2.imread("Images\Input.png")  # Le a imagem do disco

  #Imagem RGB x HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   #
    mask = cv2.inRange(imgHSV, (Red[0], Red[2], Red[4]),  (Red[1], Red[3], Red[5]))
   # mask1 = cv2.inRange(imgHSV,(Blue[0], Blue[2], Blue[4]),  (Blue[1], Blue[3], Blue[5]))
   # mask2 = cv2.inRange(imgHSV,(Yellow[0], Yellow[2], Yellow[4]),  (Yellow[1], Yellow[3], Yellow[5]))
   # imgStack = stackImages(0.6,([img,mask],
   #
#Imagem gaussian Blur
    Gaussian = cv2.GaussianBlur(mask, (11, 11), 1)
    imgStack = stackImages(1, ([img, mask, Gaussian]))

    cv2.imshow("Array", imgStack)

    time.sleep(3)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break