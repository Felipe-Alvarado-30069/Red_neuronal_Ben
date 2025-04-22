from src.Clasificador import clasificar_imagen

if __name__ == "__main__":
    ruta_modelo = 'trained_model_parameters/fuego_vs_diamante.h5'
    ruta_imagen = r"C:\Users\luis_\Repositorios\Red_neuronal_Ben\test\prueba.png"
    clases = ['diamante', 'fuego']

    clasificar_imagen(ruta_modelo, ruta_imagen, clases)
