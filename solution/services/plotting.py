"""
Gráficas compartidas por los cuadernos.

Además del espectro suelto, dibuja la comparativa que repetían todos los
cuadernos de entrenamiento —espectro observado por SDSS, espectro predicho y
espectro de Gaia que entra en el modelo— sobre los objetos de
`grouped_samples_selected_objects.csv`, en lugar de sobre índices sueltos del
conjunto de test escritos a mano en cada cuaderno.

Sale una figura por objeto, guardada en `models/images` con un nombre estable
que la memoria cita: allí se agrupan las de cada zona del diagrama HR en una
misma figura, con el nombre del grupo en el pie.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Las carpetas de datos e imágenes cuelgan de `solution`, así que se deducen de
# la posición de este fichero y no del directorio desde el que arranque el
# kernel del cuaderno.
SOLUTION_DIR = Path(__file__).resolve().parents[1]
GROUPED_SAMPLES_DIR = SOLUTION_DIR / "models" / "data" / "grouped_samples"
SELECTED_OBJECTS_PATH = GROUPED_SAMPLES_DIR / "grouped_samples_selected_objects.csv"
IMAGES_DIR = SOLUTION_DIR / "models" / "images"

# Tipos de objeto de SIMBAD, en castellano: el código escueto que devuelve el
# catálogo ('LM*', 'RR*') no se entiende. Los que no estén aquí salen sólo con
# su código.
OTYPE_LABELS = {
    "LM*": "estrella de baja masa"
  , "RR*": "variable RR Lyrae"
  , "QSO": "cuásar"
  , "WD*": "enana blanca"
}

# Paleta de Okabe-Ito: las tres series se distinguen también con deuteranopía y
# protanopía, al contrario que el azul, naranja y verde por defecto de
# Matplotlib, y todas contrastan sobre blanco al imprimir la memoria.
OBSERVED_COLOR  = "#0072b2"
PREDICTED_COLOR = "#d55e00"
INPUT_COLOR     = "#009e73"

SUBTITLE_COLOR = "#52514e"

# Tamaño base de la letra, el 'em' de la figura: los demás tamaños son relativos
# a él —Matplotlib entiende 'large' o 'small'— y también lo son las medidas del
# lienzo, de modo que cambiarlo reescala la figura entera.
BASE_FONT_SIZE = 12

FIGURE_WIDTH_EM = 56
FIGURE_HEIGHT_EM = 20

# Hueco entre el nombre del objeto y su gráfica, donde caben los identificadores
TITLE_PAD_EM = 2

# Un punto es 1/72 de pulgada: es la única conversión del módulo, y hace falta
# porque Matplotlib sólo admite pulgadas para el tamaño del lienzo
POINTS_PER_INCH = 72

FIGURE_STYLE = {
    "font.size":  BASE_FONT_SIZE
  , "axes.grid":  True
  , "grid.alpha": 0.2
}


# ---------------------------------------------------------------------------
# Puras – sin E/S
# ---------------------------------------------------------------------------


def _object_subtitle(selected_object):
    """
    Segunda línea del título: el tipo que SIMBAD asigna al objeto y sus
    identificadores de catálogo, debajo del nombre oficial.
    """
    otype = str(selected_object["simbad_otype"])
    parts = [
        # El nombre que encabeza la figura y el tipo salen los dos de SIMBAD,
        # así que la atribución va aquí y no repetida al pie
        f"SIMBAD: {OTYPE_LABELS.get(otype, otype)} ({otype})"
      , f"Gaia DR3 {selected_object['source_id']}"
        # La terna placa-MJD-fibra es la referencia con la que SDSS nombra sus
        # espectros, más legible que el identificador numérico
      , "SDSS "
        f"{int(selected_object['plate'])}-"
        f"{int(selected_object['mjd'])}-"
        f"{int(selected_object['fiberID'])}"
    ]

    return "  ·  ".join(parts)


def _test_indices(source_ids, test_source_ids):
    """
    Localiza cada objeto seleccionado dentro del conjunto de test.

    Parameters:
        source_ids (Sequence[int]): Identificadores de Gaia de los objetos.
        test_source_ids (numpy.ndarray): Identificadores del conjunto de test.

    Returns:
        list[int | None]: Índice de cada objeto en el conjunto de test, o None
            si no está en él.
    """
    positions = []
    for source_id in source_ids:
        matches = np.flatnonzero(test_source_ids == np.int64(source_id))
        positions.append(int(matches[0]) if len(matches) else None)

    return positions


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------


def plot_physical_spectrum(spectrum, title: str = "Espectro"):
    # Seaborn no es dependencia del proyecto, así que se importa aquí y no en la
    # cabecera: el resto del módulo se usa desde los cuadernos sin tenerlo
    import seaborn as sns
    from matplotlib.collections import LineCollection

    sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.linestyle": "--"})
    plt.figure(figsize=(12, 5))

    x = spectrum.spectral_axis.value
    y = spectrum.flux.value

    # Create line segments for the gradient
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap="turbo", linewidth=1)
    lc.set_array(x)
    plt.gca().add_collection(lc)

    plt.xlabel(f"Longitud de onda ({spectrum.spectral_axis.unit})")
    plt.ylabel(f"Flujo ({spectrum.flux.unit})")
    plt.title(title)

    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max() * 1.05)  # Add slight padding to the top
    plt.tight_layout()
    plt.show()


def plot_selected_objects(selection_path, test_data_path, predictions_path, model_name):
    """
    Dibuja la comparativa de espectros de cada objeto seleccionado en el CSV.

    Sale una figura por objeto —espectro observado por SDSS, espectro predicho
    y espectro de Gaia—, encabezada por su nombre celeste oficial y, debajo,
    su tipo y sus identificadores de catálogo. Cada una se guarda en
    `models/images` como `<modelo>_<grupo>_<source_id>.png`.

    La celda que la llama se ejecuta suelta: la función parte del CSV de objetos
    y de los `.npz` del conjunto de test y de las predicciones, así que no
    necesita ni el modelo cargado ni las variables que dejen en memoria las
    demás celdas del cuaderno.

    Parameters:
        selection_path (str | Path): CSV de objetos seleccionados, el que deja
            `services/simbad.py` con el nombre de cada objeto en SIMBAD.
        test_data_path (str | Path): `.npz` del conjunto de test del cuaderno.
        predictions_path (str | Path): `.npz` de predicciones del modelo.
        model_name (string): Nombre del modelo, el que llevan sus ficheros
            ('dense-global', 'cnn-v2', 'gru'...). Nombra las imágenes.

    Returns:
        list[Path]: Rutas de las imágenes guardadas, una por objeto.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(selection_path)

    with np.load(test_data_path) as test_data:
        gaia_wavelength = test_data["gaia_wavelength"]
        sdss_wavelength = test_data["sdss_wavelength"]
        test_source_ids = test_data["X_id_test"]
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]

    with np.load(predictions_path) as predictions:
        y_pred = predictions["y_pred"]

    selection["test_index"] = _test_indices(selection["source_id"], test_source_ids)

    image_paths = []
    inches_per_em = BASE_FONT_SIZE / POINTS_PER_INCH

    for _, selected_object in selection.iterrows():
        if pd.isna(selected_object["test_index"]):
            print(
                f"Aviso: {selected_object['simbad_main_id']} "
                f"(Gaia DR3 {selected_object['source_id']}) no está en el conjunto de test"
            )
            continue

        index = int(selected_object["test_index"])

        with plt.rc_context(FIGURE_STYLE):
            figure, axis = plt.subplots(
                figsize=(FIGURE_WIDTH_EM * inches_per_em, FIGURE_HEIGHT_EM * inches_per_em)
              , layout="constrained"
            )

            axis.plot(
                sdss_wavelength, y_test[index]
              , label="SDSS observado", color=OBSERVED_COLOR, alpha=0.85
            )
            axis.plot(
                sdss_wavelength, y_pred[index], label="SDSS predicho", color=PREDICTED_COLOR
            )
            axis.plot(
                gaia_wavelength, X_test[index], label="Gaia (entrada)", color=INPUT_COLOR
            )

            # El nombre oficial del objeto encabeza la figura; el tipo y los
            # identificadores van debajo, en menor tamaño
            axis.set_title(
                selected_object["simbad_main_id"]
              , fontsize="large", fontweight="bold", pad=TITLE_PAD_EM * BASE_FONT_SIZE
            )
            axis.text(
                0.5, 1.02, _object_subtitle(selected_object)
              , transform=axis.transAxes, ha="center", va="bottom"
              , fontsize="small", color=SUBTITLE_COLOR
            )

            axis.set_xlabel("Longitud de onda [Å]")
            axis.set_ylabel("Flujo")

            # La leyenda la coloca el propio motor de composición bajo la
            # gráfica, sin coordenadas escritas a mano
            figure.legend(loc="outside lower center", ncol=3, frameon=False)

            image_path = (
                IMAGES_DIR / f"{model_name}_{selected_object['group']}"
                f"_{selected_object['source_id']}.png"
            )
            figure.savefig(image_path, dpi=150)
            image_paths.append(image_path)

            plt.show()

    print(f"Imágenes guardadas en {IMAGES_DIR}")

    return image_paths
