import math

from cv2 import cv2
import numpy as np

# myColors = [[0, 179, 0, 135, 0, 45, "Black"],
#             [0, 10, 154, 194, 112, 182, "Red"],
#             [102, 117, 110, 184, 86, 160, "Blue"],
#             [18, 85, 136, 229, 124, 227, "Yellow"]]

myColors = [[0, 179, 0, 135, 0, 45, "Black"],
            [126, 179, 0, 255, 0, 255, "Red"],
            [30, 130, 0, 255, 0, 255, "Blue"],
            [20, 60, 0, 255, 0, 255, "Yellow"]]


#Função pronta para exibir as imagens lado a lado fonte: https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/blob/master/Basics/Joining_Multiple_Images_To_Display.py
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
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

#Função para pegar a posição dos contornos da imagem depois do Canny
def getContours(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = []
    startEndPoint = []
    for color in myColors:
        lower = np.array([color[0], color[2], color[4]])
        upper = np.array([color[1], color[3], color[5]])
        mask = cv2.inRange(imgHSV, lower, upper)

        img = cv2.GaussianBlur(mask, (11, 11), 1)
        posX, posY, width, height = 0, 0, 0, 0


        contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE ) #Algoritmo que pega os contornos externos
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                objectType = ""
                #cv2.drawContours(imgResult, cnt, -1, (0,0,0), 5) #-1 desenha todos os contornos

                perimeter = cv2.arcLength(cnt, True) #pega o perímetro do contorno
                polygon = cv2.approxPolyDP(cnt, 0.025*perimeter, True) #Retorna os pontos que fazem parte do contorno
                corners = len(polygon) # Pega o número de lados estimado de acordo com os pontos do contorno
                #print(corners)
                posX, posY, width, height = cv2.boundingRect(polygon) #cria um retângulo ao redor do contorno

                if posX > 0 and posY > 0:
                    cv2.putText(imgResult, color[6], ((posX - 10), (posY - 20)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                if corners == 3:
                    objectType = "Triangle"
                elif corners == 4:
                    if 0.95 <= float(width)/float(height) <= 1.05:
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
                    startEndPoint.append([(posX + posX + width)/2,(posY + posY + width)/2])

    findWay(imgResult, posX, posY, width, height, startEndPoint)

    return posX, posY


def findWay(img, posX, posY, width, height, startEndPoint):


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

            currentWheigt = math.sqrt((startPos[0] - point[0])**2 + (startPos[1] - point[1])**2)

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

        cv2.line(img, (int(startPoint[0]), int(startPoint[1])), (int(point[0]), int(point[1])) , (0, 0, 0), 2)  # Cria uma linha reta entra o ponto inicial e o final
        startPoint = point.copy()

    return

def findColisions(path):

    #for point in path:
    # y = mx+b ou y-y1 = a(x-x1) #Criar a reta entre os pontos iniciais e finais
    #     a = (point[1] - startPoint[1]) / (point[0] - startPoint[0])
    #     y1 = int(startPoint[1])
    #     x1 = int(startPoint[0])

    # myContour = np.array([[posX, posY], [posX + width, posY], [posX + width, posY + height], [posX, posY + height]],
    #                  dtype=np.int32)  # cria um contour partindo da bounding box
    # contour = myContour.reshape((-1, 1, 2))

    #cv2.polylines(imgResult, [contour], True, (255), 2) #Desenha o contorno das formas detectadas além da bounding box


    # colision = []
    #
    # for x in range(x1, point):
    #     y = a * (x - x1) + y1
    #
    #     status = cv2.pointPolygonTest(contour, (x, y), False) #Testa se algum ponto da reta está dentro do contorno da bounding box
    #
    #     if status >= 0:
    #         cv2.circle(imgResult, (x, int(y)), 5, (0, 0, 255), -1)
    #         colision.append([x, int(y)]) #Se encontrou algum ponto da reta dentro do contorno adiciona no vetor
    #
    # midElement = int((len(colision)/2))
    # if midElement > 0:
    #     middlePoint = colision[midElement]
    #     x1 = middlePoint[0]
    #     y1 = middlePoint[1]
    #     #a2 = -1/a
    #     #x2 = x1
    #     #status = 1
    #
    #     # while status >= 0:
    #     #     y2 = a2 * (x2 - x1) + y1 #Traça uma reta perpendicular a reta que liga o início e o fim
    #     #     cv2.circle(imgResult, (x2, int(y2)), 9, (0, 255, 0), -1)
    #     #     status = cv2.pointPolygonTest(contour, (x2, int(y2)), False)
    #     #     x2 += 1
    #     #
    #     #
    #     # x2 += 10
    #     # y2 = a2 * (x2 - x1) + y1
    #
    #     firstColisionValue = colision[0]
    #     lastColisionValue = colision[len(colision)-1]
    #
    #     if a < 0: #se a reta é crescente
    #         clearance = np.sqrt((posX + width - x1)**2 + (posY + height - y1)**2)
    #
    #         x = (firstColisionValue[1] - y1)/a + x1 #isolar o x na eq da reta para ver qual valor a partir da pos atual
    #         x = x - clearance/2
    #
    #         y = a * (x - x1) + y1
    #
    #         firstColisionValue[0] = int(x)
    #         firstColisionValue[1] = int(y)
    #
    #         x = (lastColisionValue[1] - y1) / a + x1  # isolar o x na eq da reta para ver qual valor a partir da pos atual
    #         x = x + clearance/2
    #         y = a * (x - x1) + y1
    #
    #         lastColisionValue[0] = int(x)
    #         lastColisionValue[1] = int(y)
    #
    #     elif a > 0: #reta decrescente
    #         firstColisionValue[0] = firstColisionValue[0] - 10
    #         firstColisionValue[1] = firstColisionValue[1] - 10
    #
    #         lastColisionValue[0] = lastColisionValue[0] + 10
    #         lastColisionValue[1] = lastColisionValue[1] + 10
    #     else: #reta horizontal
    #         firstColisionValue[0] = firstColisionValue[0] - 10
    #
    #     #imageResult = cv2.line(img, (firstColisionValue[0], firstColisionValue[1]), (x2, int(y2)), (0, 0, 0), 2)
    #     #imageResult = cv2.line(img, (x2, int(y2)), (lastColisionValue[0], lastColisionValue[1]), (0, 0, 0), 2)
    #     cv2.line(img, (firstColisionValue[0], firstColisionValue[1]), (posX + width + 10, posY + height + 10), (0, 0, 0), 2)
    #     cv2.line(img, (posX + width + 10, posY + height + 10), (lastColisionValue[0], lastColisionValue[1]), (0, 0, 0), 2)
    #     #cv2.circle(imgResult, (x1, int(y1)), int(clearance), (0, 0, 255), 5)

    return


#Função para pegar as cores dos objetos
def getColors(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = []
    for color in myColors:
        lower = np.array([color[0],color[2],color[4]])
        upper = np.array([color[1],color[3],color[5]])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        #cv2.imshow(color[6], mask)
        if x > 0 and y > 0:
            cv2.putText(imgResult, color[6], ((x-10), (y - 20)),
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


    return mask


#cap = cv2.VideoCapture(0)
#cap.set(3, 720)
#cap.set(4, 1280)

while True:

   # success, img = cap.read()
   # img = img[0:682, 160:1119]


    img = cv2.imread("Images\Input3.png") #Le a imagem do disco
    imgResult = img.copy()
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converte para escala de cinza
    # imgBlur = cv2.GaussianBlur(imgGray,(11,11),1) #Suaviza as bordas (reduz ruido nas fotos). Também chamado de filtro gaussiano
    # imgCanny = cv2.Canny(imgBlur, 50, 50) #Algoritmo de detecção de borda chamado Canny Edge Detection
    # #getContours(imgCanny) #Função para pegar a posição dos pontos do cntorno da imagem depois do Canny

    mask = getContours(img)

    #imgStack = stackImages(0.5,([img,imgGray,imgBlur],
    #                          [imgCanny, mask, imgResult]))

    cv2.imshow("Array",imgResult)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break