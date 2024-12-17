import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('./cascade/haarcascade_frontalface_alt.xml')
rostro = cv.CascadeClassifier('./cascade/haarcascade_frontalface_alt.xml')


def detectar_rostro_y_procesar(imagen):
    
    gray = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
    
    rostros = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(rostros) > 0:
        x, y, w, h = rostros[0]
        rostro = gray[y:y+h, x:x+w]  
        rostro = cv.resize(rostro, (100, 100), interpolation=cv.INTER_AREA)  
        
        _, rostro = cv.threshold(rostro, 127, 255, cv.THRESH_BINARY)
        return rostro
    else:
        return None

def comparar_imagenes_binarias(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Las imágenes deben tener el mismo tamaño para compararse.")

    total_pixels = img1.size
    matching_pixels = np.sum(img1 == img2)
    similitud = (matching_pixels / total_pixels) * 100
    return similitud

rostro1 = detectar_rostro_y_procesar(cv.imread('./photos/Rodri2/rodrifake.jpg'))
rostro2 = detectar_rostro_y_procesar(cv.imread('./photos/Rodri2/rodrireal.jpg'))

if rostro1 is not None and rostro2 is not None:
    similitud = comparar_imagenes_binarias(rostro1, rostro2)
    print(f"Las imágenes comparten un {similitud:.2f}% de similitud.")
else:
    print("No se encontraron rostros en una o ambas imágenes.")

cv.imshow("Rostro 1", rostro1)
cv.imshow("Rostro 2", rostro2)
cv.waitKey(0)
cv.destroyAllWindows()
