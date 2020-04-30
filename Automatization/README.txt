El script main.py es el que entrena y ejecuta el modelo, el script funciones.py solo tiene definidas las funciones a utilizar.

Al correr el main uno tiene tres opciones:

1- Entrena el modelo con los datos de entrenamiento y hace una evaluacion mostrando las metricas, a dicho modelo entrenado se lo guarda en un archivo binario .pkl en el mismo directorio del main.py.

2- Hace una prediccion de valores nunca vistos por el modelo (ni para entrenar ni para evaluar), es necesario que exista el archivo .pkl para que se puedan hacer las predicciones.
A los resultados de las predicciones se los guarda en la ruta definida en el path_results.

3- Hace las opciones 1 y 2 de manera consecutiva.

