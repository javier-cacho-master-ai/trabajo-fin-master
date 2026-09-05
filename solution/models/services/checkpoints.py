"""
Checkpoints del entrenamiento de los modelos.

Servicio compartido por todos los cuadernos de entrenamiento de `models`. Cada
modelo entrenado deja tres ficheros en la carpeta de su cuaderno, uno por cada
forma de dato, y el cuaderno deja además el conjunto de test en uno o varios
`.npz` en `models/data/test`, de modo que las celdas de evaluación y análisis
puedan ejecutarse en una sesión nueva sin repetir el entrenamiento:

- `<modelo>.keras`: los pesos de la mejor época de validación.
- `<modelo>_training_history.csv`: una fila por época con todas las métricas.
  Es una tabla, así que la escribe directamente el callback `CSVLogger` de
  Keras, sin código propio, y queda legible y versionable en Git. Un CSV además
  admite los `NaN` de una época divergente, que en JSON no serían válidos.
- `<modelo>_training_summary.json`: el resumen del entrenamiento (mejor época,
  tamaño de lote y métricas de test). Son datos sueltos y heterogéneos
  —cadenas, enteros y un diccionario de métricas— que no caben en una tabla ni
  en un contenedor de arrays, y en JSON siguen siendo legibles.
- `<cuaderno>_test_data.npz`: las variables del conjunto de test que consumen
  las celdas posteriores. Son arrays homogéneos de hasta cientos de megabytes,
  para los que `.npz` es el único formato razonable de los tres: conserva la
  forma y el `dtype` sin código de conversión, escribe y lee en menos de un
  segundo y ocupa lo mismo que en memoria, mientras que en JSON o CSV los
  mismos datos pasarían a varias veces su tamaño en texto. Sobre todo, `.npz`
  conserva los identificadores de Gaia como `int64`: son de hasta 19 dígitos y
  más de la mitad no se representan de forma exacta en el `float64` al que los
  llevaría un CSV o un JSON leído como decimal.

Un cuaderno que entrena varios modelos escribe un trío de ficheros por modelo, y
un `.npz` por cada preparación distinta del conjunto de test: uno solo cuando
todos sus modelos comparten la misma normalización y uno por modelo cuando cada
uno usa la suya.

Lo que no es un checkpoint del entrenamiento no lo escribe este módulo. Las
predicciones sobre el conjunto de test son el caso a tener presente: no son un
registro de cómo se entrenó el modelo, sino el dato que consume la comparativa
entre arquitecturas, así que de ellas se ocupa `predictions.py` y viven en
`models/data/predictions`, fuera de la carpeta del cuaderno.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def save_training_summary(
    summary_path,
    training_history,
    test_metrics=None,
    model_path=None,
    history_path=None,
    batch_size=None
):
    """
    Escribe el resumen del entrenamiento.

    Parameters:
        summary_path (str | Path): Ruta del fichero JSON del resumen.
        training_history (dict[str, list[float]]): Nombre de métrica ->
            lista con su valor en cada época, sobre los conjuntos de
            entrenamiento y de validación. Es el atributo `history` que
            devuelve `fit`.
        test_metrics (dict[str, float]): Nombre de métrica -> valor sobre el
            conjunto de test, si ya se ha evaluado.
        model_path (str | Path): Ruta del modelo al que acompaña el resumen.
        history_path (str | Path): Ruta del historial por época.
        batch_size (int): Tamaño de lote usado en el entrenamiento.

    Returns:
        dict: El resumen escrito, con las claves 'model', 'history', 'updated',
            'batch_size', 'epochs_completed', 'best_epoch' y 'test_metrics'.
    """
    val_loss = training_history.get("val_loss")

    summary = {
        "model": Path(model_path).name if model_path else None,
        "history": Path(history_path).name if history_path else None,
        "updated": datetime.now().isoformat(timespec="seconds"),
        "batch_size": batch_size,
        "epochs_completed": len(training_history.get("loss", [])),
        # Época con mejor validación (1-indexada), la que conserva el modelo
        "best_epoch": int(np.argmin(val_loss)) + 1 if val_loss else None,
        "test_metrics": (
            {key: float(value) for key, value in test_metrics.items()}
            if test_metrics else None
        ),
    }

    Path(summary_path).write_text(json.dumps(summary, indent=2))

    return summary


def load_training_summary(summary_path):
    """
    Lee el resumen del entrenamiento guardado.

    Parameters:
        summary_path (str | Path): Ruta del fichero JSON del resumen.

    Returns:
        dict: El resumen del entrenamiento, con las claves 'model', 'history',
            'updated', 'batch_size', 'epochs_completed', 'best_epoch' y
            'test_metrics'.
    """
    return json.loads(Path(summary_path).read_text())


def load_training_history(history_path):
    """
    Lee el historial por época guardado por `CSVLogger`.

    Parameters:
        history_path (str | Path): Ruta del fichero CSV del historial.

    Returns:
        dict[str, list[float]]: Nombre de métrica -> lista con su valor en cada
            época, sobre los conjuntos de entrenamiento y de validación. Es la
            misma estructura que el atributo `history` que devuelve `fit`.
    """
    history_table = pd.read_csv(history_path)

    # La columna 'epoch' es implícita en el historial de Keras
    return history_table.drop(columns="epoch").to_dict("list")


def load_checkpoint(model_path, history_path, summary_path, custom_objects=None):
    """
    Recupera el modelo y el registro de entrenamiento de una sesión anterior.

    Parameters:
        model_path (str | Path): Ruta del fichero `.keras` del modelo.
        history_path (str | Path): Ruta del fichero CSV del historial.
        summary_path (str | Path): Ruta del fichero JSON del resumen.
        custom_objects (dict[str, type]): Nombre de clase -> la clase, para las
            capas propias del modelo. El `.keras` guarda de ellas el nombre,
            pero no el código, así que Keras necesita recibirlas para
            reconstruir un modelo que las use.

    Returns:
        tuple: El modelo entrenado (`keras.Model`), el historial por época
            (`dict[str, list[float]]`, métrica -> valor en cada época) y el
            resumen del entrenamiento (`dict`).
    """
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    return (
        model,
        load_training_history(history_path),
        load_training_summary(summary_path)
    )


def save_test_data(test_data_path, **arrays):
    """
    Guarda las variables del conjunto de test que consumen las celdas
    posteriores al entrenamiento.

    Parameters:
        test_data_path (str | Path): Ruta del fichero `.npz`.
        **arrays (numpy.ndarray): Arrays del conjunto de test a guardar, cada
            uno con el nombre de la variable del cuaderno a la que corresponde
            ('X_test', 'y_test', 'sdss_wavelength'...).

    Returns:
        Path: Ruta del fichero escrito.
    """
    np.savez(test_data_path, **arrays)

    return Path(test_data_path)


def load_test_data(test_data_path):
    """
    Lee las variables del conjunto de test guardadas.

    Parameters:
        test_data_path (str | Path): Ruta del fichero `.npz`.

    Returns:
        dict[str, numpy.ndarray]: Nombre de la variable del cuaderno -> su
            array del conjunto de test, con la forma y el `dtype` originales.
    """
    with np.load(test_data_path) as test_arrays:
        return {name: test_arrays[name] for name in test_arrays.files}

