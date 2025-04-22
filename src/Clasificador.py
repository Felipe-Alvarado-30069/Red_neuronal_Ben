# src/classifier.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def clasificar_imagen(ruta_modelo, ruta_imagen, clases, tamano=(64, 64)):
    # Cargar modelo
    modelo = tf.keras.models.load_model(ruta_modelo)

    # Cargar y preprocesar imagen
    img = image.load_img(ruta_imagen, target_size=tamano)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predecir
    prediccion = modelo.predict(img_array)
    indice = np.argmax(prediccion)
    clase = clases[indice]
    confianza = prediccion[0][indice] * 100

    # Mostrar resultado en consola
    print(f"Predicción: {clase}")
    print(f"Confianza: {confianza:.2f}%")

    # Mostrar imagen con matplotlib
    plt.imshow(image.load_img(ruta_imagen))
    plt.axis('off')
    plt.title(f"{clase} ({confianza:.2f}%)", fontsize=14)
    plt.show()

    return clase, confianza
