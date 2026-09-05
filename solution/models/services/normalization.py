"""
Normalización de los espectros de Gaia y de SDSS.

Servicio compartido por todos los cuadernos de entrenamiento de `models`. Los
espectros llegan en unidades de flujo físico, con amplitudes que cambian en
varios órdenes de magnitud de una estrella a otra, así que todos los modelos
normalizan su entrada y su objetivo antes de entrenar y deshacen después la
normalización sobre la predicción. Lo único que cambia de un modelo a otro es el
estadístico con el que lo hacen, y aquí están reunidos los tres del trabajo:

- **Escala por espectro con la mediana** (`normalize_per_spectrum`): cada
  espectro se divide entre la mediana de su flujo absoluto y queda con mediana
  1, de modo que el modelo aprende la forma del espectro y no el brillo de la
  estrella. Es la normalización del modelo denso de referencia, y la que usan
  también el recurrente, el de la SNR y los convolucionales.
- **Escala por espectro con el máximo** (`statistic="max"`): la misma idea
  dividiendo entre el máximo del flujo absoluto, que deja el espectro dentro de
  [-1, 1] a costa de que un solo píxel —una línea de emisión, un pico de
  ruido— fije la escala de todo el espectro.
- **Tipificación** (`normalize_standard`): se resta una media y se divide entre
  una desviación típica calculadas sobre el conjunto de entrenamiento. Sin
  `axis` salen unos estadísticos globales, los del modelo denso con
  normalización global; con `axis=0` salen unos por píxel, los que el modelo de
  la SNR aplica a la relación señal-ruido que concatena a su entrada.

Las dos familias se deshacen igual de fácil, pero no con lo mismo: la escala por
espectro con `denormalize` y la tipificación con `denormalize_standard`.

No se usa scikit-learn, que sería la respuesta corta a un módulo de
normalización, porque sus transformadores no cubren el caso de este trabajo. Los
que guardan lo aprendido y saben deshacerlo —`StandardScaler`, `MaxAbsScaler`,
`RobustScaler`— trabajan por columna, es decir, por punto de la malla de
longitudes de onda, mientras que aquí la escala es de cada espectro entero, por
fila. El único que trabaja por fila, `Normalizer`, no guarda las escalas ni
tiene `inverse_transform`, y sin ellas no hay forma de devolver una predicción a
las unidades del flujo. Ninguno, además, escala por la mediana de la fila. Con
scikit-learn fuera, tampoco entra como dependencia del proyecto.

El objetivo se normaliza siempre con **su propia** escala, la del espectro de
SDSS, nunca con la de la entrada de Gaia. Así todos los modelos comparten la
misma definición del objetivo y sus métricas de entrenamiento se pueden
comparar entre sí. A cambio, deshacer la normalización de una predicción pide la
escala del espectro real de SDSS: sobre el conjunto de test sale de su `.npz`,
que la guarda junto al resto de variables.
"""

import numpy as np

# Estadístico de la escala por espectro -> la función que lo calcula. Los dos
# ignoran los NaN, que en estos espectros no aparecen, pero saldrían de una
# interpolación fuera de rango o de un píxel enmascarado.
SPECTRUM_STATISTICS = {"median": np.nanmedian, "max": np.nanmax}


def spectrum_scale(spectra, statistic="median"):
    """
    Calcula la escala de cada espectro, con la que se normaliza fila a fila.

    Parameters:
        spectra (numpy.ndarray): Espectros, uno por fila.
        statistic (str): 'median' para la mediana del flujo absoluto de cada
            espectro o 'max' para su máximo.

    Returns:
        numpy.ndarray: Escala de cada espectro, en una columna, con la que
            dividir o multiplicar el array de espectros por difusión.
    """
    reduce_statistic = SPECTRUM_STATISTICS[statistic]

    return reduce_statistic(np.abs(spectra), axis=1, keepdims=True)


def normalize(spectra, scale):
    """
    Normaliza unos espectros con una escala ya calculada.

    Parameters:
        spectra (numpy.ndarray): Espectros, uno por fila.
        scale (numpy.ndarray): Escala de cada espectro, en una columna.

    Returns:
        numpy.ndarray: Los espectros normalizados, con la forma de la entrada.
    """
    return spectra / scale


def denormalize(spectra, scale):
    """
    Deshace la normalización por escala, con la que se devuelve una predicción
    a las unidades del flujo.

    Parameters:
        spectra (numpy.ndarray): Espectros normalizados, uno por fila.
        scale (numpy.ndarray): Escala con la que se normalizaron, en una
            columna.

    Returns:
        numpy.ndarray: Los espectros en unidades de flujo, con la forma de la
            entrada.
    """
    return spectra * scale


def normalize_per_spectrum(spectra, statistic="median"):
    """
    Normaliza cada espectro con su propia escala.

    Es el atajo de las celdas de entrenamiento, que necesitan a la vez los
    espectros normalizados y la escala con la que deshacerlo más adelante.

    Parameters:
        spectra (numpy.ndarray): Espectros, uno por fila.
        statistic (str): 'median' para la mediana del flujo absoluto de cada
            espectro o 'max' para su máximo.

    Returns:
        tuple: Los espectros normalizados (`numpy.ndarray`, con la forma de la
            entrada) y la escala de cada uno (`numpy.ndarray`, en una columna).
    """
    scale = spectrum_scale(spectra, statistic)

    return normalize(spectra, scale), scale


def normalize_splits(train, val, test, statistic="median"):
    """
    Normaliza de una vez los tres conjuntos, cada espectro con su escala.

    De las tres escalas solo devuelve la del conjunto de test, que es la única
    que hace falta después de esta llamada: las de entrenamiento y validación se
    consumen al normalizar y no vuelven a aparecer, mientras que la de test se
    guarda en el `.npz` porque sin ella no se puede devolver una predicción a
    las unidades del flujo.

    Parameters:
        train (numpy.ndarray): Espectros del conjunto de entrenamiento.
        val (numpy.ndarray): Espectros del conjunto de validación.
        test (numpy.ndarray): Espectros del conjunto de test.
        statistic (str): 'median' para la mediana del flujo absoluto de cada
            espectro o 'max' para su máximo.

    Returns:
        tuple: Los tres conjuntos normalizados y la escala del de test, cuatro
            `numpy.ndarray` en ese orden.
    """
    train_norm, _ = normalize_per_spectrum(train, statistic)
    val_norm, _ = normalize_per_spectrum(val, statistic)
    test_norm, test_scale = normalize_per_spectrum(test, statistic)

    return train_norm, val_norm, test_norm, test_scale


def standard_stats(values, axis=None):
    """
    Calcula la media y la desviación típica con las que tipificar.

    Salen siempre del conjunto de entrenamiento: los de validación y test se
    tipifican con estos mismos, no con los suyos.

    Parameters:
        values (numpy.ndarray): Valores del conjunto de entrenamiento, un
            espectro por fila.
        axis (int | None): Eje sobre el que reducir. `None` da unos
            estadísticos globales, escalares; `axis=0` da uno por píxel.

    Returns:
        tuple: La media y la desviación típica (dos `numpy.ndarray`, escalares
            si `axis` es `None`).
    """
    return np.nanmean(values, axis=axis), np.nanstd(values, axis=axis)


def normalize_standard(values, mean, std):
    """
    Tipifica unos valores con la media y la desviación típica de
    `standard_stats`.

    Parameters:
        values (numpy.ndarray): Valores a tipificar, uno por fila.
        mean (numpy.ndarray): Media del conjunto de entrenamiento.
        std (numpy.ndarray): Desviación típica del conjunto de entrenamiento.

    Returns:
        numpy.ndarray: Los valores tipificados, con la forma de la entrada.
    """
    return (values - mean) / std


def denormalize_standard(values, mean, std):
    """
    Deshace la tipificación, con la que se devuelve una predicción a las
    unidades del flujo.

    Parameters:
        values (numpy.ndarray): Valores tipificados, uno por fila.
        mean (numpy.ndarray): Media con la que se tipificaron.
        std (numpy.ndarray): Desviación típica con la que se tipificaron.

    Returns:
        numpy.ndarray: Los valores en unidades de flujo, con la forma de la
            entrada.
    """
    return values * std + mean


def standardize_splits(train, val, test, axis=None):
    """
    Tipifica de una vez los tres conjuntos con los estadísticos del de
    entrenamiento.

    A diferencia de la escala por espectro, la media y la desviación típica sí
    se devuelven: son las mismas para los tres conjuntos y hacen falta después
    para deshacer la tipificación de una predicción.

    Parameters:
        train (numpy.ndarray): Valores del conjunto de entrenamiento, de los
            que salen los estadísticos.
        val (numpy.ndarray): Valores del conjunto de validación.
        test (numpy.ndarray): Valores del conjunto de test.
        axis (int | None): Eje sobre el que reducir. `None` da unos
            estadísticos globales, escalares; `axis=0` da uno por píxel.

    Returns:
        tuple: Los tres conjuntos tipificados, la media y la desviación típica,
            cinco `numpy.ndarray` en ese orden.
    """
    mean, std = standard_stats(train, axis=axis)

    return (
        normalize_standard(train, mean, std),
        normalize_standard(val, mean, std),
        normalize_standard(test, mean, std),
        mean,
        std
    )
