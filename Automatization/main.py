import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
import pickle
import os.path
from os import path

from funciones import feature_eng, entrenamiento, prediccion


def main():

    path_train = "path-a-los-datos"
    path_predict = "path-datos-validacion"
    path_results = "path-salida-modelo"

    print("Presione 1 para entrenar el modelo")
    print("Presione 2 para validar el modelo con los datos nuevos")
    print("Presione 3 para hacer ambas cosas\n")

    # validacion del input
    while True:
        option = int(input("Elija:"))
        if 0 < option < 4:
            break
        else:
            print("Opcion invalida, vuelva a elegir")

    # bloque de operaciones
    if option == 1:
        dataframe = feature_eng(path_train)
        entrenamiento(dataframe)

    elif option == 2:
        if path.exists("modelo_entrenado.pkl"):
            df_result = prediccion(path_predict)
            df_result.to_csv(path_results, sep='|', index=False)
            print("Archivo resultado 'output_modelo.txt' guardado con exito")
        else:
            print(
                "El archivo modelo_entrenado.pkl no existe o no esta en el mismo directorio del main.py")

    else:
        dataframe = feature_eng(path_train)
        entrenamiento(dataframe)

        df_result = prediccion(path_predict)
        df_result.to_csv(path_results, sep='|', index=False)
        print("Archivo resultado 'output_modelo.txt' guardado con exito")

    print("\n\n\nFIN DEL PROGRAMA")


if __name__ == '__main__':
    main()
