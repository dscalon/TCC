import math

from cv2 import cv2
from Bezier import Bezier
from scipy import interpolate
import matplotlib.pyplot as plt

import numpy as np
from numpy import array as a

# myColors = [[0, 179, 0, 135, 0, 45, "Black"],
#             [0, 10, 154, 194, 112, 182, "Red"],
#             [102, 117, 110, 184, 86, 160, "Blue"],
#             [18, 85, 136, 229, 124, 227, "Yellow"]]

myColors = [[0, 179, 0, 135, 0, 45, "Black"],
            [126, 179, 0, 255, 0, 255, "Red"],
            [30, 130, 0, 255, 0, 255, "Blue"],
            [20, 60, 0, 255, 0, 255, "Yellow"]]

startEndPoint = []

# Função pronta para exibir as imagens lado a lado fonte: https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/blob/master/Basics/Joining_Multiple_Images_To_Display.py
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


# Função para pegar a posição dos contornos da imagem depois do Canny
def getContours(img):
    global startEndPoint
    posX, posY, width, height = 0, 0, 0, 0
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    startEndPoint = []
    boxes = []
    for color in myColors:
        lower = np.array([color[0], color[2], color[4]])
        upper = np.array([color[1], color[3], color[5]])
        mask = cv2.inRange(imgHSV, lower, upper)

        img = cv2.GaussianBlur(mask, (11, 11), 1)

        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)  # Algoritmo que pega os contornos externos
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                objectType = ""
                # cv2.drawContours(imgResult, cnt, -1, (0,0,0), 5) #-1 desenha todos os contornos

                perimeter = cv2.arcLength(cnt, True)  # pega o perímetro do contorno
                polygon = cv2.approxPolyDP(cnt, 0.025 * perimeter,
                                           True)  # Retorna os pontos que fazem parte do contorno
                corners = len(polygon)  # Pega o número de lados estimado de acordo com os pontos do contorno
                # print(corners)
                posX, posY, width, height = cv2.boundingRect(polygon)  # cria um retângulo ao redor do contorno

                if posX > 0 and posY > 0:
                    cv2.putText(imgResult, color[6], ((posX - 10), (posY - 20)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                if corners == 3:
                    objectType = "Triangle"
                elif corners == 4:
                    if 0.95 <= float(width) / float(height) <= 1.05:
                        objectType = "Square"
                    else:
                        objectType = "Rectangle"
                elif corners > 4:
                    objectType = "Circle"

                cv2.rectangle(imgResult, (posX, posY), (posX + width, posY + height),
                              (0, 255, 0), 2)  # Desenha a bounding box

                cv2.putText(imgResult, objectType, ((posX + 10), (posY + 10)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                if (objectType == "Square") and (color[6] == "Black"):
                    startEndPoint.append([(posX + posX + width) / 2, (posY + posY + width) / 2])
                else:
                    boxes.append([posX, posY, posX + width, posY + height])

    findWay(imgResult, startEndPoint, boxes)

    return posX, posY


def findWay(img, startEndPoint, boundingBoxes):
    startEndPoint.sort()
    wheigts = []
    path = []
    nodes = startEndPoint.copy()
    startPos = [nodes[0][0], nodes[0][1]]
    nodes.pop(0)

    while len(nodes) > 0:

        for point in nodes:
            cv2.circle(imgResult, (int(point[0]), int(point[1])), 5,
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
    findColisions(img, startPosition, path, boundingBoxes)
    return


def findColisions(img, startPoint, path, boundingBoxes):
    startX = int(startPoint[0])
    startY = int(startPoint[1])
    params = []
    contoursWithColision = []
    pointsWithColisions = {}
    contours = []

    for box in boundingBoxes:
        myContour = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                             dtype=np.int32)  # cria um contour partindo da bounding box
        contour = myContour.reshape((-1, 1, 2))
        contours.append([box, contour])

    for point in path:
        # y = mx+b ou y-y1 = a(x-x1) #Criar a reta entre os pontos iniciais e finais
        if startX < point[0]:
            Range = range(startX, int(point[0]))
        else:
            Range = range(int(point[0]), startX)

        a = (point[1] - startY) / (point[0] - startX)

        for x in Range:

            y = a * (x - startX) + startY

            for contour in contours:
                cv2.polylines(img, [contour[1]], True, (255),
                              2)  # Desenha o contorno das formas detectadas além da bounding box

                colision = []
                status = cv2.pointPolygonTest(contour[1], (x, y),
                                              False)  # Testa se algum ponto da reta está dentro do contorno da bounding box

                if status >= 0:
                    if contoursWithColision.count(contour[0]) == 0:
                       # cv2.circle(imgResult, (int(x), int(y)), 5, (255, 120, 255), -1)
                        contoursWithColision.append(contour[0])
                        params.append([a, startX, startY, contour])

                    pointsWithColisions.setdefault(tuple(contour[0]), []).append([int(x), int(y)])



        startX = int(point[0])
        startY = int(point[1])
        counter = 0

    for key in pointsWithColisions.keys():
        findControlPoints(key, pointsWithColisions[key], params[counter])
        counter = counter + 1

    return


def findControlPoints(box, points, params):


    listOfPoints = []
    for point in points:
        listOfPoints.append(point)


    # Pegar primeiro e último ponto com colisão para criar control point antes da colisão // params[0] = a (inclinação), params[1] = x e params[2] = y
    x0 = listOfPoints[0][0] - 4
    y0 = params[0] * (listOfPoints[0][0] - 4 - params[1]) + params[2]

    xf = listOfPoints[-1][0] + 4
    yf = params[0] * (listOfPoints[-1][0] + 4 - params[1]) + params[2]


    listOfPoints.insert(0, [x0, int(y0)])
    listOfPoints.append([xf, int(yf)])

    cv2.circle(imgResult, (listOfPoints[0][0], listOfPoints[0][1]), 5, (0, 255, 255), -1)
    cv2.circle(imgResult, (listOfPoints[-1][0], listOfPoints[-1][1]), 5, (255, 0, 255), -1)

    #Pegar o ponto do meio da box e criar uma circunferência, deixando uma área livre
    clearance = 20
    radius = box[2] - box[0] + clearance
    center = [(box[0] + box[2])/2, (box[1]+box[3])/2]

    angleOffset = math.atan(params[0])

    theta = np.arange(math.pi/4 + angleOffset, 2*math.pi + math.pi/4 + angleOffset -math.pi/20, math.pi/2)

    circlePoints = []
    for angle in theta:
        circlePoints.append([int(radius * math.cos(angle) + center[0]), int(radius * math.sin(angle) + center[1])])
        cv2.circle(imgResult, (int(radius * math.cos(angle) + center[0]), int(radius * math.sin(angle) + center[1])), 5, (255, 0, 255), -1)

    #Pegar ponto do meio da reta que cruza a box e calcular a distância até os pontos da circunferência

    middleIndex = float(len(listOfPoints)) / 2

    if middleIndex % 2 != 0:
        middleElement = listOfPoints[int(middleIndex - .5)]
    else:
        middleElement = listOfPoints[int(middleIndex-1)]

    distanceMiddleVertices = []

    for point in circlePoints:
        distanceMiddleVertices.append([middleElement, point, math.sqrt((middleElement[0] - point[0]) ** 2
                                                                            + (middleElement[1] - point[1]) ** 2)])

    def sortKey(elem):
        return elem[2] #Dar o sort pela distância

    distanceMiddleVertices.sort(key=sortKey)
    controlPoints = [distanceMiddleVertices[0][1], distanceMiddleVertices[1][1], distanceMiddleVertices[2][1],
                     distanceMiddleVertices[3][1]]


    angularCoeffs = []
    for points in controlPoints:
        if points[0] != controlPoints[0][0]:
            angularCoeffs.append([points, (points[1] - controlPoints[0][1]) / (points[0] - controlPoints[0][0])])

    angularCoeffs = [[x[0], x[1] - params[0]] for x in angularCoeffs]

    def sortKey2(elem):
        return elem[1] #Dar o sort pela diferença

    angularCoeffs.sort(key=sortKey2)

    controlPoints.insert(1,angularCoeffs[0][0])
    print(str(angularCoeffs))
    print(str(controlPoints))


    t_points = np.arange(0, 1, 0.01)
    alternativePath = np.array([[listOfPoints[0][0], listOfPoints[0][1]], [int(controlPoints[0][0]), int(controlPoints[0][1])],
                                [int(controlPoints[1][0]), int(controlPoints[1][1])], [listOfPoints[-1][0], listOfPoints[-1][1]]])
    test_set = Bezier.Curve(t_points, alternativePath)

    for interpolated in test_set:
        cv2.circle(imgResult, (int(interpolated[0]), int(interpolated[1])), 5, (255, 0, 0), -1)
        status = cv2.pointPolygonTest(params[3][1], (int(interpolated[0]), int(interpolated[1])),
                                      False)  # Testa se algum ponto do caminho está dentro do contorno da bounding box
        if status > 0:
            cv2.circle(imgResult, (int(interpolated[0]), int(interpolated[1])), 5, (45, 45, 255), -1)

    cv2.circle(imgResult, (int(controlPoints[0][0]), int(controlPoints[0][1])), 5, (255, 0, 0), -1)
    cv2.circle(imgResult, (int(controlPoints[1][0]), int(controlPoints[1][1])), 5, (255, 0, 0), -1)
    cv2.circle(imgResult, (listOfPoints[0][0], listOfPoints[0][1]), 5, (0, 255, 255), -1)
    cv2.circle(imgResult, (listOfPoints[-1][0], listOfPoints[-1][1]), 5, (255, 0, 255), -1)


    return


"""
    # Pegar ponto do meio do vetor e calcular a distância até os vértices
    # vértices -  [box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]

    middleIndex = float(len(listOfPoints))/2

    if middleIndex % 2 != 0:
        middleElement = listOfPoints[int(middleIndex - .5)]
    else:
        middleElement = listOfPoints[int(middleIndex-1)]

    vertices = [box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]
    distanceMiddleVertices = []

    for vertex in vertices:
        distanceMiddleVertices.append([middleElement, vertex, math.sqrt((middleElement[0] - vertex[0])**2
                                                                        + (middleElement[1] - vertex[1])**2)])

    def sortKey(elem):
        return elem[2]

    distanceMiddleVertices.sort(key=sortKey)

    middlePoint1, vertexPoint1, distance1 = distanceMiddleVertices[0]

    if distance1 > 0:
        angularCoef = (vertexPoint1[1] - middlePoint1[1]) / (vertexPoint1[0] - middlePoint1[0])

        if (middlePoint1[1] - vertexPoint1[1]) >= 0:
            controlPointX1 = int(vertexPoint1[0] + 20)
        else:
            controlPointX1 = int(vertexPoint1[0] - 20)

        controlPointY1 = int(angularCoef * (controlPointX1 - middlePoint1[0]) + middlePoint1[1])

        status = cv2.pointPolygonTest(params[3][1], (controlPointX1, controlPointY1),
                                      False)

        if status >= 0:

            middlePoint1, vertexPoint1, distance1 = distanceMiddleVertices[1]
            if distance1 > 0:
                angularCoef = (vertexPoint1[1] - middlePoint1[1]) / (vertexPoint1[0] - middlePoint1[0])

                if (middlePoint1[1] - vertexPoint1[1]) >= 0:
                    controlPointX1 = int(vertexPoint1[0] + 20)
                else:
                    controlPointX1 = int(vertexPoint1[0] - 20)

                controlPointY1 = angularCoef * (controlPointX1 - middlePoint1[0]) + middlePoint1[1]

        #cv2.circle(imgResult, (int(controlPointX1), int(controlPointY1)), 5, (255, 0, 0), -1)
        #cv2.circle(imgResult, (int(controlPointX1), int(controlPointY1)), 5, (255, 0, 0), -1)
        #cv2.circle(imgResult, (listOfPoints[0][0], listOfPoints[0][1]), 5, (0, 255, 255), -1)
        #cv2.circle(imgResult, (listOfPoints[-1][0], listOfPoints[-1][1]), 5, (255, 0, 255), -1)
        t_points = np.arange(0,1, 0.01)
        alternativePath = np.array([[listOfPoints[0][0], listOfPoints[0][1]], [int(controlPointX1), int(controlPointY1)], [listOfPoints[-1][0], listOfPoints[-1][1]]])
        test_set = Bezier.Curve(t_points, alternativePath)

        for interpolated in test_set:
          cv2.circle(imgResult, (int(interpolated[0]), int(interpolated[1])), 2, (127, 128, 129), -1)
        
    middlePoint2, vertexPoint2, distance2 = distanceMiddleVertices[1]

    if distance2 > 0:
        angularCoef = (vertexPoint2[1] - middlePoint2[1]) / (vertexPoint2[0] - middlePoint2[0])

        if (middlePoint2[1] - vertexPoint2[1]) >= 0:
            controlPointX2 = int(vertexPoint2[0] + 20)
        else:
            controlPointX2 = int(vertexPoint2[0] - 20)

        controlPointY2 = int(angularCoef * (controlPointX2 - middlePoint2[0]) + middlePoint2[1])

        status = cv2.pointPolygonTest(params[3][1], (controlPointX2, controlPointY2),
                                      False)

        if status >= 0:

            middlePoint2, vertexPoint2, distance2 = distanceMiddleVertices[1]
            if distance1 > 0:
                angularCoef = (vertexPoint2[1] - middlePoint2[1]) / (vertexPoint2[0] - middlePoint2[0])

                if (middlePoint2[1] - vertexPoint2[1]) >= 0:
                    controlPointX2 = int(vertexPoint2[0] + 20)
                else:
                    controlPointX2 = int(vertexPoint2[0] - 20)

                controlPointY2 = angularCoef * (controlPointX2 - middlePoint2[0]) + middlePoint2[1]
        """

        #270 163 351 206 361 159



# Função para pegar as cores dos objetos
def getColors(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = []
    for color in myColors:
        lower = np.array([color[0], color[2], color[4]])
        upper = np.array([color[1], color[3], color[5]])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        # cv2.imshow(color[6], mask)
        if x > 0 and y > 0:
            cv2.putText(imgResult, color[6], ((x - 10), (y - 20)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return mask


# cap = cv2.VideoCapture(0)
# cap.set(3, 720)
# cap.set(4, 1280)

while True:

    # success, img = cap.read()
    # img = img[0:682, 160:1119]

    img = cv2.imread("Images\Input3.png")  # Le a imagem do disco
    imgResult = img.copy()

    mask = getContours(img)

    # imgStack = stackImages(0.5,([img,imgGray,imgBlur],
    #                          [imgCanny, mask, imgResult]))

    cv2.imshow("Array", imgResult)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
