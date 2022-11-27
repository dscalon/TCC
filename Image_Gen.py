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

myColors = [[0, 179, 0, 135, 0, 45, "Black"],
            [0, 15, 13, 255, 253, 255, "Red"],
            [109, 125, 239, 255, 252, 255, "Blue"],
            [15, 27, 252, 255, 238, 255, "Orange"],
            [84, 103, 180, 195, 230, 240, "Turquish"]]


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


    img = cv2.imread("Images\Input18.png")  # Le a imagem do disco

  #Imagem RGB x HSV
    posX, posY, width, height = 0, 0, 0, 0
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    startEndPoint = []
    boxes = []
    for color in myColors:
        lower = np.array([color[0], color[2], color[4]])
        upper = np.array([color[1], color[3], color[5]])
        mask = cv2.inRange(imgHSV, lower, upper)

        Gaussian = cv2.GaussianBlur(mask, (11, 11), 1)

        contours, hierarchy = cv2.findContours(Gaussian, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # Algoritmo que pega os contornos externos
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                objectType = ""
              #  cv2.drawContours(img, cnt, -1, (0,0,0), 5) #-1 desenha todos os contornos

                perimeter = cv2.arcLength(cnt, True)  # pega o perímetro do contorno
                polygon = cv2.approxPolyDP(cnt, 0.025 * perimeter,
                                       True)  # Retorna os pontos que fazem parte do contorno
                corners = len(polygon)  # Pega o número de lados estimado de acordo com os pontos do contorno
            # print(corners)
                posX, posY, width, height = cv2.boundingRect(polygon)  # cria um retângulo ao redor do contorno

            # if posX > 0 and posY > 0:
             #   cv2.putText(img, color[6], ((posX - 10), (posY - 30)),
             #          cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

                if corners == 3:
                    objectType = "Triangle"
                elif corners == 4:
                    if 0.95 <= float(width) / float(height) <= 1.05:
                        objectType = "Square"
                    else:
                        objectType = "Rectangle"
                elif corners > 4:
                    objectType = "Circle"

              #  cv2.putText(img, objectType, ((posX), (posY - 5)),
               #                  cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.rectangle(img, (posX, posY), (posX + width, posY + height),
                      (0, 255, 0), 2)  # Desenha a bounding box

                if (objectType == "Square") and (color[6] == "Turquish"):
                    startEndPoint.insert(0, [(posX + posX + width) / 2, (posY + posY + width) / 2])
                if (objectType == "Square") and (color[6] == "Black"):
                    startEndPoint.append([(posX + posX + width) / 2, (posY + posY + width) / 2])

        orangeBox = startEndPoint.pop(0)
        startEndPoint.sort()
        startEndPoint.insert(0, orangeBox)
        wheigts = []
        path = []
        nodes = startEndPoint.copy()
        startPos = [nodes[0][0], nodes[0][1]]
        #nodes.pop(0)

    while len(nodes) > 0:

            for point in nodes:
                cv2.circle(img, (int(point[0]), int(point[1])), 5,
                           (0, 255, 0), -1)  # Desenha o ponto inicial na tela

                currentWheigt = math.sqrt((startPos[0] - point[0]) ** 2 + (startPos[1] - point[1]) ** 2)

                if currentWheigt > 0:
                    wheigts.append(currentWheigt)

            if len(wheigts) > 0:
                min_wheigt = min(wheigts)
                min_wheigt_index = wheigts.index(min_wheigt)
                startPos = nodes[min_wheigt_index]
                path.append(nodes[min_wheigt_index])

            wheigts.clear()
            nodes.remove(nodes[min_wheigt_index])

    startPoint = startEndPoint[0]

    for point in path:
        cv2.line(img, (int(startPoint[0]), int(startPoint[1])), (int(point[0]), int(point[1])), (0, 0, 0),
                     2)  # Cria uma linha reta entra o ponto inicial e o final
        startPoint = point.copy()

    startPosition = startEndPoint[0][0], startEndPoint[0][1]
    imgStack = stackImages(1, ([img,Gaussian]))

    cv2.imshow("Array", imgStack)

    time.sleep(3)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break