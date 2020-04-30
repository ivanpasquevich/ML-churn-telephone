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


def feature_eng(path_datos, sepa="|", del_outliers=True):
    """Funcion para hacer la ingenieria de features del dataset de datos y el merge con el target que esta
    en el dataset target

    Parameters
    -----------

    path_datos : string
        Ruta al archivo que contiene los datos para entrenar el modelo

    sepa : string
        Caracter separador del archivo datos

    del_ouliers : bool
        Indica a la funcion si hacer el tratamiento de outliers

    Return
    -----------
    dataframe listo para el entrenamiento del modelo


    """

    print("-------------------------")
    print("Comienza la funcion de feature engineering")

    df = pd.read_csv(path_datos, sep=sepa)
    print("Dataframe de datos cargado")
    print(f"El dataframe tiene {df.shape[1]} columnas\n")

    # columnas a eliminar por defecto
    col_defecto = [col for col in df.columns.tolist() if
                   ("EXPDTR_DATA_ARPU_AMT" in col or "SMS_OFFNET_EXP_ARPU_AMT" in col
                    or "SPNDG_VOI_ONNET_ARPU" in col or "USE_LCL_VOI_AMT" in col
                    or "SPNDG_VOI_INTRNTL_ARPU" in col or "SPNDG_VOI_OFFNET_ARPU" in col
                    or "SMS_ONNET_EXP_ARPU_AMT" in col)]

    # columnas a eliminar que contienen nulos
    cant_nulos = df.isnull().sum()[df.isnull().sum() > 0]
    col_nulos = cant_nulos.index.to_list()

    # columnas extras a eliminar por analisis y no brindan info para el target
    col_extras = ["SEGMENTATION", "MICROSEGMENTATION", "SOURCE", "TENURE_CUSTOMER_BL", "PREP_RECH_CHNNL_MODE12W",
                  "MSISDN", 'PREP_RECH_LAST_DAY_12W']

    # columnas de pack de voz (tanto mensual como semanal)
    col_voz = re.findall("\S*VOICE\S*", " ".join(df.columns.to_list()))

    # columnas de pack de sms (tanto mensual como semanal) y trafico de sms
    col_sms = re.findall("\S*SMS\S*", " ".join(df.columns.to_list()))

    # columnas de las recargas acumuladas por dia
    col_dias = re.findall("\S*Q\S*[0-9]+W", " ".join(df.columns.to_list()))

    # columnas del trafico de voz
    col_trv = re.findall("TRV\S*", " ".join(df.columns.to_list()))

    # columnas del df target
    col_tg = re.findall("\S*W_[0-9]", " ".join(df.columns.to_list()))
    col_tg.append('FLAG_MIG')

    # ponemos todas las columnas en la misma lista
    lista_elim = col_defecto + col_nulos + col_extras + \
        col_voz + col_sms + col_dias + col_trv + col_tg
    lista_elim = list(set(lista_elim))

    # eliminamos las columnas de nuestro df
    df = df.drop(lista_elim, axis=1).copy()

    print(f"Se eliminaron {len(lista_elim)} columnas")
    print(f"Ahora el dataframe tiene {df.shape[1]} columnas\n")

    # Vamos a tratar los outliers de los campos de y de datos
    # en base a los graficos de bigotes realizados durante la exploracion de datos, vemos que hay outliers muy
    # lejanos a la mediana

    # Como solucion general se propone hacer lo siguiente para cada columna con outliers
    # Obtenemos el intervalo en el que se mueven los outliers de la columna [minimo,maximo]
    # De este intervalo encontramos el valor que se encuentra a una distancia de 2/3 del intervalo con respecto
    # al minimo del mismo, a este valor lo llamamos limite.
    # A todos los outliers con un valor superior al limite vamos a reemplazarlos por la media de los outliers

    if del_outliers:
        print("Se procede al tratamiento de outliers")

        lista_col_outliers = re.findall("P\S*", " ".join(df.columns.to_list()))

        lista_col_outliers.remove("PREP_RECH_NDAYS_LASTRECH_12W")

        for col_name in lista_col_outliers:

            q1 = df[col_name].quantile(0.25)
            q3 = df[col_name].quantile(0.75)
            iqr = q3 - q1
            minimo = q3 + 1.5 * iqr
            maximo = df[col_name].max()
            limite = ((maximo - minimo) * 2 / 3) + minimo

            mask = df[col_name] > minimo
            media = round(df[col_name][mask].values.mean(), 2)

            df[col_name] = df[col_name].map(
                lambda x: x if x <= limite else media)

        print("Tratamiento de outliers finalizado\n")

    # FEATURE ENGINEERING

    print("Se procede al bloque de ingenieria de features")

    # creamos la nueva feature que relaciona las expiraciones de los datos y luego eliminamos esas features
    col_exp = re.findall("\S*DATA_EXP\S*", " ".join(df.columns.to_list()))
    df["DATA_EXP_MEAN"] = (df[col_exp].sum(axis=1)) / len(col_exp)
    df = df.drop(col_exp, axis=1).copy()

    # Tenemos variables mensuales y semanales, hay que quedarnos con alguna de ellas y hacer una transformacion de
    # las otras.
    # Como solucion vamos a quedarnos con las variables mensuales de cantidad de eventos, a las variables mensuales
    # de monto de eventos vamos a transformarlas para volverlas adimensionales.

    # Entonces:

    # Campos a mantener intactos:
    #'PREP_RECH_Q_EVT_X1','PREP_RECH_Q_EVT_X2','PREP_RECH_Q_EVT_X3','PACK_DATA_Q_X1',
    #'PACK_DATA_Q_X2','PACK_DATA_Q_X3'

    # COLUMNA DE MONTO ANUAL RECARGAS
    df['PREP_RECH_AMT_ANU'] = df['PREP_RECH_AMT_X1'] + \
        df['PREP_RECH_AMT_X2'] + df['PREP_RECH_AMT_X3']
    # Montos relativos por mes RECARGAS
    df['PREP_RECH_AMT_REL_X1'] = (
        df['PREP_RECH_AMT_X1'] / df['PREP_RECH_AMT_ANU']).fillna(0)
    df['PREP_RECH_AMT_REL_X2'] = (
        df['PREP_RECH_AMT_X2'] / df['PREP_RECH_AMT_ANU']).fillna(0)
    df['PREP_RECH_AMT_REL_X3'] = (
        df['PREP_RECH_AMT_X3'] / df['PREP_RECH_AMT_ANU']).fillna(0)

    # COLUMNA DE MONTO ANUAL DATOS
    df['PACK_DATA_AMT_ANU'] = df['PACK_DATA_AMT_X1'] + \
        df['PACK_DATA_AMT_X2'] + df['PACK_DATA_AMT_X3']
    # Montos relativos por mes DATOS
    df['PACK_DATA_AMT_REL_X1'] = (
        df['PACK_DATA_AMT_X1'] / df['PACK_DATA_AMT_ANU']).fillna(0)
    df['PACK_DATA_AMT_REL_X2'] = (
        df['PACK_DATA_AMT_X2'] / df['PACK_DATA_AMT_ANU']).fillna(0)
    df['PACK_DATA_AMT_REL_X3'] = (
        df['PACK_DATA_AMT_X3'] / df['PACK_DATA_AMT_ANU']).fillna(0)

    # hay casos en los que el divisor es cero, a estas cuentas el resultado va a ser un NaN, por lo que los
    # reemplazamos por cero

    # Eliminamos las columnas de los montos totales
    df = df.drop(['PREP_RECH_AMT_ANU', 'PACK_DATA_AMT_ANU'], axis=1).copy()

    # En cuanto a las variables semanales vamos a descartarlas por el momento
    col_semanas = re.findall("\S*W[0-9]+", " ".join(df.columns.to_list()))
    df = df.drop(col_semanas, axis=1).copy()

    # Agregamos columnas relativas de cantidad de meses

    # COLUMNA mes dos con respecto al mes 1 y 2 recargas
    df['PREP_RECH_Q_REL_X2'] = df['PREP_RECH_Q_EVT_X2'] / \
        (df['PREP_RECH_Q_EVT_X1'] + df['PREP_RECH_Q_EVT_X2'])
    df['PREP_RECH_Q_REL_X2'] = df['PREP_RECH_Q_REL_X2'].fillna(0)
    df['PREP_RECH_Q_REL_X2'] = df['PREP_RECH_Q_REL_X2'].map(
        lambda x: x if x != np.inf else 0)

    # COLUMNA mes tres con respecto al total recargas
    df['PREP_RECH_Q_REL_X3'] = df['PREP_RECH_Q_EVT_X3'] / \
        (df['PREP_RECH_Q_EVT_X1'] +
         df['PREP_RECH_Q_EVT_X2'] + df['PREP_RECH_Q_EVT_X3'])
    df['PREP_RECH_Q_REL_X3'] = df['PREP_RECH_Q_REL_X3'].fillna(0)
    df['PREP_RECH_Q_REL_X3'] = df['PREP_RECH_Q_REL_X3'].map(
        lambda x: x if x != np.inf else 0)

    # COLUMNA mes dos con respecto al mes 1 y 2 datos
    df['PACK_DATA_Q_REL_X2'] = df['PACK_DATA_Q_X2'] / \
        (df['PACK_DATA_Q_X1'] + df['PACK_DATA_Q_X2'])
    df['PACK_DATA_Q_REL_X2'] = df['PACK_DATA_Q_REL_X2'].fillna(0)
    df['PACK_DATA_Q_REL_X2'] = df['PACK_DATA_Q_REL_X2'].map(
        lambda x: x if x != np.inf else 0)

    # COLUMNA mes tres con respecto al total DATOS
    df['PACK_DATA_Q_REL_X3'] = df['PACK_DATA_Q_X3'] / \
        (df['PACK_DATA_Q_X1'] + df['PACK_DATA_Q_X2'] + df['PACK_DATA_Q_X3'])
    df['PACK_DATA_Q_REL_X3'] = df['PACK_DATA_Q_REL_X3'].fillna(0)
    df['PACK_DATA_Q_REL_X3'] = df['PACK_DATA_Q_REL_X3'].map(
        lambda x: x if x != np.inf else 0)

    # Otras features interesantes se pueden obtener de la relacion que hay entre el monto y la cantidad de eventos
    # de las recargas y los datos
    # Hay casos en los que el divisor es cero, a estas cuentas el resultado va a ser un NaN, por lo que los
    # reemplazamos por cero

    df['PREP_RECH_REL_X1'] = (
        df['PREP_RECH_AMT_X1'] / df['PREP_RECH_Q_EVT_X1']).fillna(0)
    df['PREP_RECH_REL_X2'] = (
        df['PREP_RECH_AMT_X2'] / df['PREP_RECH_Q_EVT_X2']).fillna(0)
    df['PREP_RECH_REL_X3'] = (
        df['PREP_RECH_AMT_X3'] / df['PREP_RECH_Q_EVT_X3']).fillna(0)

    df['PACK_DATA_REL_X1'] = (
        df['PACK_DATA_AMT_X1'] / df['PACK_DATA_Q_X1']).fillna(0)
    df['PACK_DATA_REL_X2'] = (
        df['PACK_DATA_AMT_X2'] / df['PACK_DATA_Q_X2']).fillna(0)
    df['PACK_DATA_REL_X3'] = (
        df['PACK_DATA_AMT_X3'] / df['PACK_DATA_Q_X3']).fillna(0)

    # Eliminacion de columnas de montos
    col_montos = re.findall("\S*AMT_X[0-9]+", " ".join(df.columns.to_list()))
    df = df.drop(col_montos, axis=1).copy()

    # Variables de tiempo, obtener la diferencia en dias entre la fecha de alta y la de cierre
    # luego obtener un porcentaje en relacion al tiempo total
    # Porcentaje del tiempo entre el primer evento y el último que se tiene registro

    # Creo columna nueva con la diferencia en dias entre las fechas de activacion y de corte
    df["DAYS_CUSTOMER"] = (pd.to_datetime(df["FECHA_CORTE"])
                           - pd.to_datetime(df['COMMERCIAL_ACTIVATION_DATE'])).dt.days

    # PREP_RECH_NDAYS_LASTRECH_12W dias desde la ultima recarga
    df["PERCENT_DAYS"] = (
        df["DAYS_CUSTOMER"] - df["PREP_RECH_NDAYS_LASTRECH_12W"]) / df["DAYS_CUSTOMER"]

    # Con estas 2 nuevas columnas ya no necesito las otras columnas relacionadas con los tiempos
    # las elimino del datafrane
    df = df.drop(["PREP_RECH_NDAYS_LASTRECH_12W", "FECHA_CORTE",
                  'COMMERCIAL_ACTIVATION_DATE', 'TENURE_CUSTOMER'], axis=1).copy()

    # En cuanto al trafico de datos vamo a tomar solamente las features mensuales, al resto las tiramos
    col_trd = ["TRD_STR_M1", "TRD_STR_M2", "TRD_STR_M3", "TRD_SN_M1", "TRD_SN_M2",
               "TRD_SN_M3", "TRD_IM_M1", "TRD_IM_M2", "TRD_IM_M3", "TRD_OTH_M1", "TRD_OTH_M2", "TRD_OTH_M3"]
    df = df.drop(col_trd, axis=1).copy()

    print("Fin del bloque de ingenieria de features\n")

    print("Tratamiento variables categoricas")

    le = preprocessing.LabelEncoder()
    le.fit(df["VALUE_SEGMENT"])
    df["VALUE_SEGMENT"] = le.transform(df["VALUE_SEGMENT"])

    print("Fin proceso variables categoricas\n")

    print("Reordenamiento de columnas")

    index_target = np.where(df.columns == "target")[0][0]
    lista_orden = (df.columns.tolist()[:index_target] + df.columns.tolist()[index_target + 1:]
                   + [df.columns.tolist()[index_target]])
    df = df[lista_orden].copy()

    print("Columnas ordenadas\n")

    print("Return del dataframe en proceso\n")

    print("Fin de la funcion de feature engineering\n ")
    print("-------------------------\n\n")

    return df


def entrenamiento(df):
    """Funcion para hacer el entrenamiento del modelo y guardarlo en un archivo .pkl

    Parameters
    -----------

    df : pandas.DataFrame
        Dataframe con las features listas para el entrenamiento del modelo


    Return
    -----------
    None


    """

    print("-------------------------")
    print("\n\nIngreso a la funcion de entrenamiento del modelo\n")

    # train test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df.drop(['SUSCRIBER_KEY', 'target'], axis=1),
                                                        df['target'], test_size=0.30,
                                                        random_state=101)

    xgb_model = xgb.XGBClassifier()

    print("Comienza entrenamiento del modelo XGBoost")

    xgb_model.fit(X_train, y_train)

    print("Entrenamiento finalizado\n")

    print("Generando archivo modelo_entrenado.pkl")

    with open("modelo_entrenado.pkl", 'wb') as file:
        pickle.dump(xgb_model, file)

    print("Archivo generado\n")

    print("Haciendo las predicciones\n\n")

    XGB_preds = xgb_model.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(y_test, XGB_preds))
    print(confusion_matrix(y_test, XGB_preds))

    print("\n\nFin funcion entrenamiento")
    print("-------------------------\n\n")

    return None


def prediccion(path_predict):
    """Funcion para hacer el entrenamiento del modelo y guardarlo en un archivo .pkl

    Parameters
    -----------

    path_predict : string
        Ruta al archivo con los datos para hacer las predicciones


    Return
    -----------
    Dataframe de 3 columnas, la identificacion del cliente, la prediccion de la clase y la probabilidad de que el cliente
    permanezca en el servicio.


    """
    print("\n\n-------------------------")
    print("Ingreso a la funcion de prediccion\n")

    print("CARGANDO EL DATASET DE VALIDACION")

    df = feature_eng(path_predict, del_outliers=False)

    print("Cargando el modelo_entrenado.pkl ")

    with open("modelo_entrenado.pkl", 'rb') as file:
        loaded_model = pickle.load(file)

    print(f"Modelo cargado .. tipo: {type(loaded_model)}")

    X, y = df.iloc[:, 1:-1], df.iloc[:, -1]

    XGB_preds = loaded_model.predict(X)

    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(y, XGB_preds))
    print(confusion_matrix(y, XGB_preds))

    df["target_2"] = XGB_preds

    XGB_proba_total = loaded_model.predict_proba(X)

    df["probabilidad"] = XGB_proba_total[:, 0]

    print("Fin de la funcion de prediccion ")
    print("-------------------------\n\n")

    return df[["SUSCRIBER_KEY", "probabilidad"]]
