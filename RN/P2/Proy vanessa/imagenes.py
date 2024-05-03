import cv2
import numpy as np
import os
import tensorflow as tf
from  tensorflow.keras import layers, models

# Ruta de la carpeta que contiene las imágenes
carpeta_imagenes = '/mnt/d/Repositorios/Deep_Learning/RN/P2/Proy vanessa/squares/squares'

# Lista para almacenar las imágenes procesadas
imagenes_procesadas = []

# Tamaño deseado para redimensionar las imágenes
nuevo_ancho = 28
nuevo_alto = 28

# Iterar sobre los archivos en la carpeta
for archivo in os.listdir(carpeta_imagenes):
    # Comprobar si el archivo es una imagen
    if archivo.endswith('.jpg') or archivo.endswith('.png'):
        # Ruta completa de la imagen
        ruta_imagen = os.path.join(carpeta_imagenes, archivo)
        
        # Cargar la imagen utilizando OpenCV
        imagen = cv2.imread(ruta_imagen)
        
        # Redimensionar la imagen al tamaño deseado
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
        
        # Convertir la imagen a escala de grises
        imagen_gris = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2GRAY)
        
        # Normalizar los valores de píxeles al rango [0, 1]
        imagen_normalizada = imagen_gris / 255.0
        
        # Agregar la imagen procesada a la lista
        imagenes_procesadas.append(imagen_normalizada)

# Convertir la lista de imágenes procesadas en una matriz numpy
matriz_imagenes = np.array(imagenes_procesadas)
# Supongamos que tienes una lista de 100 imágenes donde las primeras 50 son imágenes de cuadrados y las siguientes 50 no lo son
num_imagenes = 100
num_imagenes_cuadrados = 50

# Crear etiquetas para las imágenes
etiquetas_clase_de_interes = [0] * num_imagenes_cuadrados + [1] * (num_imagenes - num_imagenes_cuadrados)

# Convertir la lista de etiquetas en una matriz NumPy
etiquetas_clase_de_interes = np.array(etiquetas_clase_de_interes)


# Crear un modelo de red neuronal
modelo = models.Sequential([
    layers.Flatten(input_shape=(nuevo_ancho, nuevo_alto)),  # Capa de entrada (aplanar la imagen)
    layers.Dense(256, activation='relu'),  # Capa oculta con 128 neuronas y función de activación ReLU
    layers.Dense(256, activation='relu'),  # Capa oculta con 128 neuronas y función de activación ReLU
    layers.Dense(1, activation='sigmoid')  # Capa de salida con 1 neurona y función de activación sigmoide
])

# Compilar el modelo
modelo.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(matriz_imagenes, etiquetas_clase_de_interes, epochs=100)


# Ruta de la carpeta que contiene las imágenes
carpeta_imagenes = '/mnt/d/Repositorios/Deep_Learning/RN/P2/Proy vanessa/prueba'

# Lista para almacenar las imágenes procesadas
imagenes_procesadas = []

# Tamaño deseado para redimensionar las imágenes
nuevo_ancho = 28
nuevo_alto = 28

# Iterar sobre los archivos en la carpeta
for archivo in os.listdir(carpeta_imagenes):
    # Comprobar si el archivo es una imagen
    if archivo.endswith('.jpg') or archivo.endswith('.png'):
        # Ruta completa de la imagen
        ruta_imagen = os.path.join(carpeta_imagenes, archivo)
        
        # Cargar la imagen utilizando OpenCV
        imagen = cv2.imread(ruta_imagen)
        
        # Redimensionar la imagen al tamaño deseado
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
        
        # Convertir la imagen a escala de grises
        imagen_gris = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2GRAY)
        
        # Normalizar los valores de píxeles al rango [0, 1]
        imagen_normalizada = imagen_gris / 255.0
        
        # Agregar la imagen procesada a la lista
        imagenes_procesadas.append(imagen_normalizada)

# Convertir la lista de imágenes procesadas en una matriz numpy
matriz_imagenes_prueba = np.array(imagenes_procesadas)

# Supongamos que tienes una lista de 100 imágenes donde las primeras 50 son imágenes de cuadrados y las siguientes 50 no lo son
num_imagenes = 6
num_imagenes_cuadrados = 3

# Crear etiquetas para las imágenes
etiquetas_clase_de_interes_prueba = [0] * num_imagenes_cuadrados + [1] * (num_imagenes - num_imagenes_cuadrados)

# Convertir la lista de etiquetas en una matriz NumPy
etiquetas_clase_de_interes_prueba = np.array(etiquetas_clase_de_interes_prueba)

# Evaluar el modelo en datos de prueba
loss, accuracy = modelo.evaluate(matriz_imagenes_prueba, etiquetas_clase_de_interes_prueba)
print('Precisión del modelo en datos de prueba:', accuracy)


# Guardar el modelo entrenado
tf.saved_model.save(modelo,"/mnt/d/Repositorios/Deep_Learning/RN/P2/Proy vanessa/modelo_guardado")

