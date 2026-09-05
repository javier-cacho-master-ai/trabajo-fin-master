"""
Conjuntos de entrenamiento, validación y test de los que parten los cuadernos.

Servicio compartido por todos los cuadernos de entrenamiento de `models`. Todos
arrancan del mismo `.npz`, el que deja el preprocesado en `data/splits`, así que
localizarlo y leerlo vive aquí y no repetido en la celda de datos de cada
cuaderno.

Los arrays se piden por su nombre porque el fichero pasa de los ochocientos
megabytes y ningún cuaderno usa todo lo que lleva dentro: leerlo entero cargaría
en memoria cientos de megabytes de errores de Gaia y de varianzas inversas de
SDSS que solo usan algunos.

Lo leído queda en una caché de módulo, de modo que una celda pueda pedir los
arrays que necesita sin volver a leer del disco los que ya cargó otra. Es lo que
permite ejecutar suelta cualquier sección de entrenamiento de un cuaderno: la
celda que abre la sección pide los arrays de los que parte, y si ya los cargó
otra celda de la sesión, los recupera de la caché en lugar del disco.
"""

import numpy as np

from services.paths import DATA_DIR

SPLITS_DIR = DATA_DIR / "splits"

# Nombre del array -> su contenido, para no leer dos veces el mismo del disco
_loaded_arrays = {}


def splits_path():
    """
    Ruta del `.npz` con los conjuntos que deja el preprocesado.

    El nombre del fichero puede llevar la fecha de generación como prefijo, y de
    haber una versión sin duplicados es la que se usa.

    Returns:
        Path: Ruta del `.npz` de datos.

    Raises:
        FileNotFoundError: Si no hay ningún `.npz` de datos en `data/splits`.
    """
    candidates = (
        sorted(SPLITS_DIR.glob("*processed_data_no_duplicates.npz"))
        + sorted(SPLITS_DIR.glob("*processed_data.npz"))
    )

    if not candidates:
        raise FileNotFoundError(f"No hay ningún `.npz` de datos en {SPLITS_DIR}")

    return candidates[0]


def load_splits(*names):
    """
    Lee del `.npz` de datos los arrays pedidos.

    Parameters:
        *names (str): Nombres de los arrays a leer ('X_train', 'y_test',
            'sdss_wavelength'...). Sin ninguno se leen todos.

    Returns:
        dict[str, numpy.ndarray]: Nombre del array -> su contenido, con la forma
            y el `dtype` originales.
    """
    with np.load(splits_path()) as split_arrays:
        # Abrir el `.npz` solo lee su índice: cada array se lee al pedirlo
        wanted = names or tuple(split_arrays.files)

        _loaded_arrays.update({
            name: split_arrays[name]
            for name in wanted
            if name not in _loaded_arrays
        })

    return {name: _loaded_arrays[name] for name in wanted}
