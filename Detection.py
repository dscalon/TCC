import math
import time
import numpy as np
import cv2
from Bezier import Bezier
from imutils.video import FileVideoStream

# myColors = [[0, 179, 0, 135, 0, 45, "Black"],
#             [0, 10, 154, 194, 112, 182, "Red"],
#             [102, 117, 110, 184, 86, 160, "Blue"],
#             [18, 85, 136, 229, 124, 227, "Yellow"],
#             [10, 22, 236, 255, 238, 255, "Orange"]]

#myColors = [[0, 179, 0, 135, 0, 45, "Black"],
#            [126, 179, 0, 255, 0, 255, "Red"],
#            [30, 130, 0, 255, 0, 255, "Blue"],
#            [20, 60, 0, 255, 0, 255, "Yellow"],
#            [10, 22, 236, 255, 238, 255, "Orange"]]

myColors = [[0, 179, 0, 135, 0, 45, "Black"],
            [0, 15, 13, 255, 253, 255, "Red"],
            [109, 125, 239, 255, 252, 255, "Blue"],
            [15, 27, 252, 255, 238, 255, "Yellow"],
            [84, 103, 180, 195, 230, 240, "Turquish"]]


global startEndPoint
startEndPoint = []
global finalWay
finalWay = []



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

                # if posX > 0 and posY > 0:
                #     cv2.putText(imgResult, color[6], ((posX - 10), (posY - 20)),
                #                 cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

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

                # cv2.putText(imgResult, objectType, ((posX + 10), (posY + 10)),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                if (objectType == "Square") and (color[6] == "Turquish"):
                    startEndPoint.insert(0, [(posX + posX + width) / 2, (posY + posY + width) / 2])

                elif (objectType == "Square") and (color[6] == "Black"):
                    startEndPoint.append([(posX + posX + width) / 2, (posY + posY + width) / 2])
                else:
                    boxes.append([posX - 10, posY - 10, posX + width + 10, posY + height + 10])

    findWay(imgResult, startEndPoint, boxes)

    return posX, posY


def findWay(img, startEndPoint, boundingBoxes):
    orangeBox = []
    orangeBox = startEndPoint.pop(0)
    startEndPoint.sort()
    startEndPoint.insert(0, orangeBox)
    wheigts = []
    path = []
    nodes = startEndPoint.copy()
    startPos = [nodes[0][0], nodes[0][1]]


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
        cv2.line(img, (int(startPoint[0]), int(startPoint[1])), (int(point[0]), int(point[1])), (151, 85, 47),
                 4)  # Cria uma linha reta entra o ponto inicial e o final
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

        a = (point[1] - startY) / (point[0] + 1 - startX)

        if a > 90 or a < -90 :
            if startY < point[1]:
                Range = range(startY, int(point[1]))
            else:
                Range = range(int(point[1]), startY)

            for y in Range:

                x = startX
                finalWay.append([int(x), int(y)])
                for contour in contours:
                    cv2.polylines(img, [contour[1]], True, (255),
                                  2)  # Desenha o contorno das formas detectadas além da bounding box

                    colision = []
                    status = cv2.pointPolygonTest(contour[1], (x, y),
                                                  False)  # Testa se algum ponto da reta está dentro do contorno da bounding box

                    if status >= 0:
                        finalWay.pop(-1)
                        cv2.circle(imgResult, (int(x), int(y)), 5, (255, 120, 255), -1)
                        if contoursWithColision.count(contour[0]) == 0:
                            #cv2.circle(imgResult, (int(x), int(y)), 5, (255, 120, 255), -1)
                            contoursWithColision.append(contour[0])
                            params.append([a, startX, startY, contour])

                        pointsWithColisions.setdefault(tuple(contour[0]), []).append([int(x), int(y)])

        else:
            if startX < point[0]:
                Range = range(startX, int(point[0]))
            else:
                Range = range(int(point[0]), startX)

            for x in Range:
                status = 0
                y = a * (x - startX) + startY
                finalWay.append([int(x), int(y)])
                for contour in contours:
                    cv2.polylines(img, [contour[1]], True, (255),
                              2)  # Desenha o contorno das formas detectadas além da bounding box

                    colision = []
                    status = cv2.pointPolygonTest(contour[1], (x, y),
                                              False)  # Testa se algum ponto da reta está dentro do contorno da bounding box

                    if status >= 0:
                        finalWay.pop(-1)
                        cv2.circle(imgResult, (int(x), int(y)), 5, (255, 120, 255), -1)
                        if contoursWithColision.count(contour[0]) == 0:
                            #cv2.circle(imgResult, (int(x), int(y)), 5, (255, 120, 255), -1)
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



    if params[0] > -90 and params[0] < 90:

        # Pegar primeiro e último ponto com colisão para criar control point antes da colisão // params[0] = a (inclinação), params[1] = x e params[2] = y
        x0 = listOfPoints[0][0] - 1
        y0 = params[0] * (listOfPoints[0][0] - 1 - params[1]) + params[2]

        xf = listOfPoints[-1][0] + 1
        yf = params[0] * (listOfPoints[-1][0] + 1 - params[1]) + params[2]

        listOfPoints.insert(0, [x0, int(y0)])
        listOfPoints.append([xf, int(yf)])

    else:
        x0 = listOfPoints[0][0]
        y0 = listOfPoints[0][1]
        xf = listOfPoints[-1][0]
        yf = listOfPoints[-1][1]


    #cv2.circle(imgResult, (listOfPoints[0][0], listOfPoints[0][1]), 5, (45, 50, 160), -1)
    #cv2.circle(imgResult, (listOfPoints[-1][0], listOfPoints[-1][1]), 5, (45, 50, 160), -1)

    #Pegar o ponto do meio da box e criar uma circunferência, deixando uma área livre
    clearance = 15
    #clearance = (math.sqrt((xf-x0)**2 + (yf-y0)**2)/2) + 10
    #radius = box[2] - box[0]# + clearance
    #center = [(box[0] + box[2])/2, (box[1]+box[3])/2]
    radius = math.sqrt((xf - x0) ** 2 + (yf - y0) ** 2) + clearance
    center = [(x0 + xf)/2, (yf+y0)/2]
    #cv2.circle(imgResult, [int(center[0]), int(center[1])], int(radius), (0, 0, 0), 4, cv2.LINE_AA)
    #cv2.circle(imgResult, [int(center[0]), int(center[1])], 8, (0, 0, 0), -1, cv2.LINE_AA)
    angleOffset = math.atan(params[0])

    controlPoints = []

    #controlPoints.append([int(radius * math.cos(math.pi/4 + angleOffset) + center[0]), int(radius * math.sin(math.pi/4 + angleOffset) + center[1])])
    #controlPoints.append([int(radius * math.cos(3*math.pi/4 + angleOffset) + center[0]), int(radius * math.sin(3*math.pi/4 + angleOffset) + center[1])])

    controlPoints.append([int(radius * math.cos(math.pi / 4 + angleOffset) + center[0]),
                          int(radius * math.sin(math.pi / 4 + angleOffset) + center[1])])

    controlPoints.append([int(radius * math.cos(3 * math.pi / 4 + angleOffset) + center[0]),
                          int(radius * math.sin(3 * math.pi / 4 + angleOffset) + center[1])])

    #Pegar ponto do meio da reta que cruza a box e calcular a distância até os pontos da circunferência

    # cv2.circle(imgResult, (controlPoints[0]), 5,
    #            (255, 0, 255), -1)
    #
    # cv2.circle(imgResult, (controlPoints[1]), 5,
    #            (255, 0, 255), -1)

    distanceStartControlPoints = []

    for point in controlPoints:
        distanceStartControlPoints.append([point, math.sqrt((point[1] - y0)**2 + (point[0] - x0)**2)])

    def SortKey(obj):
        return obj[1]

    distanceStartControlPoints.sort(key=SortKey)

    controlPoints[0] = distanceStartControlPoints[0][0]
    controlPoints[1] = distanceStartControlPoints[1][0]

    t_points = np.arange(0, 1, 0.01)
    alternativePath = np.array([[listOfPoints[0][0], listOfPoints[0][1]], [int(controlPoints[0][0]), int(controlPoints[0][1])],
                                [int(controlPoints[1][0]), int(controlPoints[1][1])], [listOfPoints[-1][0], listOfPoints[-1][1]]])
    test_set = Bezier.Curve(t_points, alternativePath)

    for interpolated in test_set:

        status = cv2.pointPolygonTest(params[3][1], (int(interpolated[0]), int(interpolated[1])),
                                      False)  # Testa se algum ponto do caminho está dentro do contorno da bounding box
        if status > 0:
            controlPoints = []

            controlPoints.append([int(radius * math.cos(5 * math.pi / 4 + angleOffset) + center[0]),
                                  int(radius * math.sin(5 * math.pi / 4 + angleOffset) + center[1])])
            controlPoints.append([int(radius * math.cos(7 * math.pi / 4 + angleOffset) + center[0]),
                                  int(radius * math.sin(7 * math.pi / 4 + angleOffset) + center[1])])

            # cv2.circle(imgResult, (controlPoints[0]), 5,
            #            (123, 0, 12), -1)
            #
            # cv2.circle(imgResult, (controlPoints[1]), 5,
            #            (123, 0, 12), -1)
            distanceStartControlPoints = []

            for point in controlPoints:
                distanceStartControlPoints.append([point, math.sqrt((point[1] - y0) ** 2 + (point[0] - x0) ** 2)])

            def SortKey(obj):
                return obj[1]

            distanceStartControlPoints.sort(key=SortKey)

            controlPoints[0] = distanceStartControlPoints[0][0]
            controlPoints[1] = distanceStartControlPoints[1][0]

            t_points = np.arange(0, 1, 0.01)
            alternativePath = np.array(
                [[listOfPoints[0][0], listOfPoints[0][1]], [int(controlPoints[0][0]), int(controlPoints[0][1])],
                 [int(controlPoints[1][0]), int(controlPoints[1][1])], [listOfPoints[-1][0], listOfPoints[-1][1]]])
            test_set = Bezier.Curve(t_points, alternativePath)


    for interpolated in test_set:
        #cv2.circle(imgResult, (int(interpolated[0]), int(interpolated[1])), 3, (151, 85, 47), -1,  cv2.LINE_AA)
        finalWay.append([int(interpolated[0]), int(interpolated[1])])

    for way in finalWay:
        cv2.circle(imgResult, (way[0], way[1]), 3, (151, 85, 47), -1,  cv2.LINE_AA)


    cv2.circle(imgResult, (int(controlPoints[0][0]), int(controlPoints[0][1])), 8, (0, 140, 255), -1, cv2.LINE_AA)
    cv2.circle(imgResult, (int(controlPoints[1][0]), int(controlPoints[1][1])), 8, (0, 140, 255), -1, cv2.LINE_AA)
    cv2.circle(imgResult, (listOfPoints[0][0], listOfPoints[0][1]), 8, (128, 128, 128), -1,  cv2.LINE_AA)
    cv2.circle(imgResult, (listOfPoints[-1][0], listOfPoints[-1][1]), 8, (128, 128, 128), -1,  cv2.LINE_AA)

    return

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
        # if x > 0 and y > 0:
        #     cv2.putText(imgResult, color[6], ((x - 10), (y - 20)),
        #                 cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return mask

#fvs = FileVideoStream("Images\Input19.mp4").start()

#time.sleep(1.0)
#cap = cv2.VideoCapture("Images\Input18.gif", cv2.CAP_FFMPEG)
#cap.set(3, 720)
#cap.set(4, 1280)
#fps = cap.get(cv2.CAP_PROP_FPS)

while True:

    #success, img = cap.read()
    #img = fvs.read()

    #if not fvs.more():
    #    fvs = FileVideoStream("Images\Input19.mp4").start()
    #    time.sleep(1.0)
    #    img = fvs.read()
    #    cv2.resize(img, (1280, 720))
#    if success == False:
#        cap = cv2.VideoCapture("Images\Input18.gif", cv2.CAP_FFMPEG)
#        success, img = cap.read()
#
 #   img = img[0:682, 160:1119]


    img = cv2.imread("Images\Input6.1.png")  # Le a imagem do disco
    imgResult = img.copy()

    mask = getContours(img)

   # imgStack = stackImages(0.5,([img,imgGray,imgBlur],
   #                          [imgCanny, mask, imgResult]))

    cv2.imshow("Array", imgResult)
    finalWay.clear()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
