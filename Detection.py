from cv2 import cv2
import numpy as np

myColors = [[0, 10, 184, 255, 77, 253, "Red"],
            [100, 135, 119, 237, 41, 134, "Blue"],
            [9, 34, 184, 255, 135, 253, "Yellow"]]


#Função pronta para exibir as imagens lado a lado fonte: https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/blob/master/Basics/Joining_Multiple_Images_To_Display.py
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
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
    img = cv2.GaussianBlur(img, (11, 11), 1)
    posX, posY, width, height = 0, 0, 0, 0

    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE ) #Algoritmo que pega os contornos externos
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 11000:
            objectType = ""
            #cv2.drawContours(imgResult, cnt, -1, (0,0,0), 5) #-1 desenha todos os contornos

            perimeter = cv2.arcLength(cnt, True) #pega o perímetro do contorno
            poligon = cv2.approxPolyDP(cnt, 0.02*perimeter, True) #Retorna os pontos que fazem parte do contorno
            corners = len(poligon) # Pega o número de lados estimado de acordo com os pontos do contorno
            print(corners)
            posX, posY, width, height = cv2.boundingRect(poligon) #cria um retângulo ao redor do contorno

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
                          (0, 255, 0), 2)  # Desenha um retângulo na tela

            cv2.putText(imgResult, objectType, ((posX-41), (posY - 10)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
    return posX, posY
#Função para pegar as cores dos objetos
def getColors(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array([color[0],color[2],color[4]])
        upper = np.array([color[1],color[3],color[5]])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        #cv2.imshow(color[6], mask)
        if x!= 0 and y!= 0:
            cv2.putText(imgResult, color[6], ((x - 60), (y - 40)),
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
    return mask

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)

while True:
    success, img = cap.read()


    #img = cv2.imread("Images\Input.png") #Le a imagem do disco
    imgResult = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converte para escala de cinza
    imgBlur = cv2.GaussianBlur(imgGray,(11,11),1) #Suaviza as bordas (reduz ruido nas fotos). Também chamado de filtro gaussiano
    imgCanny = cv2.Canny(imgBlur, 50, 50) #Algoritmo de detecção de borda chamado Canny Edge Detection
    #getContours(imgCanny) #Função para pegar a posição dos pontos do cntorno da imagem depois do Canny

    mask = getColors(img)

    imgStack = stackImages(0.5,([img,imgGray,imgBlur],
                              [imgCanny, mask, imgResult]))

    cv2.imshow("Array",imgStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break