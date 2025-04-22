# Red_neuronal_Ben 🔥💎

Este proyecto consiste en una red neuronal convolucional (CNN) entrenada para **distinguir entre dos personajes de la serie Ben 10: Fuego y Diamante**, funcionando como un sistema de clasificación de imágenes de dos clases.

## 📁 Estructura del repositorio

```
Red_neuronal_Ben/
├── src/
│   ├── Captura.py            # Entrenamiento del modelo CNN
│   ├── Clasificador.py       # Clasificación de imágenes individuales
│   └── Conversion.py         # (Opcional)
├── trained_model_parameters/
│   ├── fuego_vs_diamante.h5  # Modelo final entrenado
│   └── fuego_vs_diamante_1.h5
├── main.py                   # Script principal de inferencia
├── requirements.txt          # Dependencias del proyecto
├── .gitignore
└── README.md
```

## 🧠 Descripción

La red neuronal fue entrenada usando `TensorFlow` y `Keras`, con técnicas de **data augmentation** para mejorar su generalización. El modelo tiene una arquitectura CNN con normalización por lotes, capas de convolución, max pooling y dropout para evitar el overfitting.

- Clases: `['diamante', 'fuego']`
- Tamaño de entrada: `64x64 píxeles`
- Número de épocas: `50`
- Datos divididos: `80% entrenamiento / 20% validación`

## 🛠️ Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tu_usuario/Red_neuronal_Ben.git
cd Red_neuronal_Ben
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 🚀 Entrenamiento

Para entrenar el modelo desde cero usando tus propias imágenes organizadas por carpetas:

```bash
python src/Captura.py
```

El modelo se guarda automáticamente como `fuego_vs_diamante.h5`.

## 🧪 Clasificación

Para clasificar una imagen individual:

1. Asegúrate de que `main.py` tenga la ruta correcta a tu imagen:

```python
ruta_imagen = r"ruta/a/tu/imagen.png"
```

2. Ejecuta:

```bash
python main.py
```

Se mostrará en consola la clase predicha y el nivel de confianza, además de una ventana con la imagen etiquetada.

## 📊 Resultados

Durante el entrenamiento se generan gráficas de precisión y pérdida para evaluar el rendimiento del modelo.

## 📌 Requisitos

- Python >= 3.7
- TensorFlow >= 2.0
- matplotlib
- numpy

(Ver `requirements.txt` para la lista completa.)

---

## 📷 Ejemplo de uso

```python
from src.Clasificador import clasificar_imagen

ruta_modelo = 'trained_model_parameters/fuego_vs_diamante.h5'
ruta_imagen = 'test/prueba_10.png'
clases = ['diamante', 'fuego']

clasificar_imagen(ruta_modelo, ruta_imagen, clases)
```

---

Dataset: https://drive.google.com/drive/folders/1Vik_XsPgdXAJJzT9Ib1CL-QtKJiDY-GX?usp=drive_link
