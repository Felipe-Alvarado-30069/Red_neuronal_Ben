from pathlib import Path
from PIL import Image
import os

# Ruta a la carpeta con imágenes
input_folder = Path(r"C:\Users\luis_\Repositorios\Red_neuronal_Ben\test")

# Carpeta de salida para guardar las imágenes en formato PNG
output_folder = input_folder / "png_convertidas"
output_folder.mkdir(exist_ok=True)

# Extensiones válidas de imagen (puedes añadir más si necesitas)
formatos_validos = ['.jpg', '.jpeg', '.bmp', '.tiff', '.webp']

# Conversión
for img_path in input_folder.iterdir():
    if img_path.suffix.lower() in formatos_validos:
        try:
            with Image.open(img_path) as img:
                nombre_salida = output_folder / f"{img_path.stem}.png"
                img.convert("RGB").save(nombre_salida, "PNG")
                print(f"Convertido: {img_path.name} -> {nombre_salida.name}")
        except Exception as e:
            print(f"Error al convertir {img_path.name}: {e}")
