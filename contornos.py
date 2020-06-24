from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import os

from difflib import SequenceMatcher as SM

def imprimir(prediccion):
    for i in range(len(prediccion)):
        if prediccion[i]==1.0:
            return categorias[i]

    return categorias[i]

def reconocer(direct):

    # direct = Contornos(direct)
    direct = cv2.cvtColor(direct, cv2.COLOR_BGR2GRAY)

    x_offset=0
    y_offset=0
    new_array=0
    IMG_SIZE=32

    background = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    background[:] = (255,255,255)

    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    if direct.shape[1]>direct.shape[0]:
        new_array = cv2.resize(direct, ((int)((direct.shape[0]/((direct.shape[1])/20))), 20))
    else:
        new_array = cv2.resize(direct, ((int)((direct.shape[1])/(direct.shape[0]/20)), 20))
    x_offset=(int)(16-(new_array.shape[1]/2))
    y_offset=(int)(16-(new_array.shape[0]/2))

    background[y_offset:y_offset+new_array.shape[0], x_offset:x_offset+new_array.shape[1]] = new_array
    threshold=190
    max_value=255
    threshold_stype=cv2.THRESH_BINARY
    ret, background = cv2.threshold(background, threshold, max_value, threshold_stype)

    #Engrosamiento de bordes
    # kernel = np.ones((2,2),np.uint8)
    # background = cv2.dilate(255-background,kernel,iterations = 1)
    # background = 255-background

    return background.reshape(-1,IMG_SIZE, IMG_SIZE, 1)

def Contornos(direct):
    z = cv2.cvtColor(direct.copy(), cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Img', direct.copy())
    # cv2.waitKey(0)
    ret, th = cv2.threshold(z, 127, 255, 0)
    cont, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cont)<=3 and (z.shape[0]/z.shape[1]*100)>35:
        [x,y,w,h] = cv2.boundingRect(cont[1])
        z = z[y:y+h, x:x+w]
        # cv2.imshow('Img', z)
        # cv2.waitKey(0)
    return z

def AVGSizeLetras():
    cantidad=0
    suma=0
    maximo=0
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if h>maximo:
            maximo=h
        suma+=h
        cantidad+=1
    return (suma/cantidad)/2 #Eliminar ruido

def RellenarContornosLetrasNum():
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if h>promedio and h<im.shape[0]:
            out = round(h/4,0)
            for i in range(2):
                if (w/h*100)<35:
                    cv2.rectangle(im,(x+1,(int)(y-5)),(x+w-2,(int)(y+h+out/2)),(0,0,255),cv2.FILLED)
                else:
                    if h>promedio*2:
                        cv2.rectangle(im,(x+1,y),(x+w-2,(int)(y+h+out/3)),(0,0,255),cv2.FILLED)
                    else:
                        if i==0:
                            cv2.rectangle(im,(x+1,y),(x+w-2,(int)(y+h+out/3)),(255,255,255),cv2.FILLED)
                            cv2.fillPoly(im,cnt,(255,255,255))

def RellenarContornos():
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if h>promedio and h<im.shape[0]:
            out = round(h/4,0)
            for i in range(2):
                if (w/h*100)<35:
                    cv2.rectangle(im,(x+1,(int)(y-5)),(x+w-2,(int)(y+h+out/2)),(0,0,255),cv2.FILLED)
                else:
                    if h>promedio*2:
                        cv2.rectangle(im,(x+1,y),(x+w-2,(int)(y+h+out/3)),(0,0,255),cv2.FILLED)

def EncontrarContornos(img):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def OrdenarCaracteres():
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])#Ordenar por lineas [y]
    caracteres=[]
    lineas=[]
    anterior=0
    for cnt in sorted_ctrs:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if h>promedio and h<im.shape[0]:
            if abs(anterior-y)>45:
                anterior =y
                caracteres=sorted(caracteres, key=lambda a: a[0]) #Ordenar por [x]
                lineas.append(caracteres.copy())
                caracteres=[]
            else:
                caracteres.append([x,y,w,h])
    caracteres=sorted(caracteres, key=lambda a: a[0])
    lineas.append(caracteres.copy())
    return lineas

def ReconocerPorModelo():
    cadena=""
    ant=[0,0,0,0]
    for l in lineas:
        for char in l:
            [x,y,w,h] = char
            if h>promedio and h<im.shape[0]:
                if (x-(ant[0]+ant[2]))>promedio/2 and ant[0]>0:
                    cadena+=" "
                cv2.rectangle(im2,(x,y),(x+w,y+h),(255,0,0),1)
                recorte = im3[y:y+h, x:x+w]
                prediccion = new_model.predict([reconocer(recorte)])
                cadena+=imprimir(prediccion[0])
                ant = [x,y,w,h]
            else:
                cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)
        cadena+="\n"
    return cadena


im = cv2.imread("fuentes/3.jpg")# Filtro de contornos
im2 = cv2.imread("fuentes/3.jpg")# Letras enmarcadas
im3 = cv2.imread("fuentes/3.jpg")# Original y de procesamiento

new_model = keras.models.load_model("caracteres32")
categorias = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

contours, hierarchy = EncontrarContornos(im)

# imx = im3.copy()
# cv2.drawContours(imx, contours, -1, (255,0,0), 1)
# cv2.imshow('Img', imx)
# cv2.waitKey(0)

promedio = AVGSizeLetras()
RellenarContornosLetrasNum()

# cv2.imshow('Img', im)
# cv2.waitKey(0)

contours, hierarchy = EncontrarContornos(im)
RellenarContornos()

# cv2.imshow('Img', im)
# cv2.waitKey(0)

contours, hierarchy = EncontrarContornos(im)
lineas = OrdenarCaracteres()
cadena = ReconocerPorModelo()

print("Texto Reconocido de imagen: ")
print(cadena)
texto_Escrito="Cuando sacas el Motorola Edge de la caja te das cuenta de que no es un Motorola mas Si bien el dispositivo bebe del diseno de sus hermanos de la gama One poco o nada tiene que ver con ellos cuando observamos su muy curvada pantalla OLED y su tasa de refresco Es un gama media premium bastante interesante y por que no decirlo llamativo Ya tuvimos ocasion de probarlo para contaros nuestras primeras impresiones pero ahora es el momento de hacer un analisis con mayor profundidad asi que sin mas dilacion vamos a ello"
print("Precisión: "+str(round(SM(None, cadena, texto_Escrito).ratio()*100, 2))+"%")
cv2.imshow('Img', im2)
cv2.waitKey(0)
cv2.destroyAllWindows()