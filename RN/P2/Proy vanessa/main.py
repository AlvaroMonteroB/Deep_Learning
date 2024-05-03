import cv2
import numpy as np
import tensorflow as tf

# Cargar la imagen utilizando OpenCV
imagen = cv2.imread("/mnt/d/Repositorios/Deep_Learning/RN/P2/Proy vanessa/deca.jpg")

# Redimensionar la imagen al tamaño deseado (28x28 en este ejemplo)
nuevo_ancho = 28
nuevo_alto = 28
imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))

# Convertir la imagen a escala de grises si es necesario
imagen_gris = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2GRAY)

# Normalizar los valores de píxeles al rango [0, 1]
imagen_normalizada = imagen_gris / 255.0

# Expandir las dimensiones de la imagen para que coincida con la forma de entrada del modelo (añadir una dimensión para el lote)
imagen_para_prediccion = np.expand_dims(imagen_normalizada, axis=0)

# Cargar el modelo entrenado desde el archivo
modelo_cargado = model=tf.saved_model.load("/mnt/d/Repositorios/Deep_Learning/RN/P2/Proy vanessa/modelo_guardado")

imagen_para_prediccion = tf.cast(imagen_para_prediccion, tf.float32)
imagen_para_prediccion = tf.reshape(imagen_para_prediccion, [-1, 28, 28])


# Realizar la predicción sobre la imagen preprocesada
prediccion = modelo_cargado(imagen_para_prediccion)

# Convertir la salida de la predicción en una etiqueta de clase
# Por ejemplo, si la probabilidad es mayor que 0.5, podemos considerarla como clase 1, de lo contrario, como clase 0
clase_predicha = 1 if prediccion >0.8 else 0

if clase_predicha == 1: 
    print("La figura introducida es un cuadrilatero")
else: 
    print("La figura introducida es un polígono de más de 4 lados")