"""
Predicciones de los modelos sobre el conjunto de test.

Servicio compartido por todos los cuadernos de entrenamiento de `models`. Las
predicciones no son un checkpoint del entrenamiento —de eso se ocupa
`checkpoints.py`—, sino el dato que consume la comparativa entre arquitecturas,
que vive en su propio cuaderno. Por eso tienen módulo propio y se guardan en
`models/data/predictions/<modelo>_predictions.npz`, fuera de la carpeta del
cuaderno que entrena el modelo.

`predict_test_set` las genera a partir de los ficheros de checkpoint —el
`.keras` del modelo y el `.npz` del conjunto de test—, de modo que la celda que
la llama se ejecuta suelta: no reentrena, no vuelve a cargar el `.npz` de datos
completo y no depende de las variables que dejen en memoria las demás celdas
del cuaderno.

Su `.npz` va comprimido, al contrario que el del conjunto de test: se escribe
una vez y se lee muchas veces desde el cuaderno de comparativa, y son espectros
suaves que el compresor reduce a una fracción de su tamaño en memoria.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

from services.checkpoints import load_test_data


def predict_test_set(
    model_path,
    test_data_path,
    build_input,
    denormalize,
    custom_objects=None
):
    """
    Genera las predicciones de un modelo entrenado sobre el conjunto de test.

    Lo único que cambia de un modelo a otro es su normalización, así que se
    recibe en dos funciones: la que arma la entrada a partir de los arrays de
    test y la que devuelve la salida a las unidades del flujo de SDSS.

    Parameters:
        model_path (str | Path): Ruta del fichero `.keras` del modelo.
        test_data_path (str | Path): Ruta del `.npz` del conjunto de test.
        build_input (Callable): Arrays de test (`dict[str, numpy.ndarray]`) ->
            la entrada que espera el modelo, normalizada y con su misma forma.
        denormalize (Callable): Salida normalizada del modelo
            (`numpy.ndarray`) y arrays de test (`dict[str, numpy.ndarray]`) ->
            la predicción en las unidades del flujo de SDSS.
        custom_objects (dict[str, type]): Nombre de clase -> la clase, para las
            capas propias del modelo. El `.keras` guarda de ellas el nombre,
            pero no el código, así que Keras necesita recibirlas para
            reconstruir un modelo que las use.

    Returns:
        numpy.ndarray: Predicción sobre el conjunto de test, sin normalizar y
            con una fila por espectro.
    """
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    test_data = load_test_data(test_data_path)

    return denormalize(model.predict(build_input(test_data)), test_data)


def save_predictions(predictions_path, **predictions):
    """
    Guarda las predicciones de un modelo sobre el conjunto de test.

    Parameters:
        predictions_path (str | Path): Ruta del fichero `.npz`. Su carpeta se
            crea si no existe.
        **predictions (numpy.ndarray): Predicciones a guardar, cada una con el
            nombre con el que se leerán ('y_pred').

    Returns:
        Path: Ruta del fichero escrito.
    """
    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(predictions_path, **predictions)

    return predictions_path


def load_predictions(predictions_path):
    """
    Lee las predicciones guardadas de un modelo sobre el conjunto de test.

    Parameters:
        predictions_path (str | Path): Ruta del fichero `.npz`.

    Returns:
        dict[str, numpy.ndarray]: Nombre de la predicción -> su array, con la
            forma y el `dtype` originales.
    """
    with np.load(predictions_path) as prediction_arrays:
        return {name: prediction_arrays[name] for name in prediction_arrays.files}
