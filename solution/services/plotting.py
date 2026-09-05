"""
Gráficas compartidas por los cuadernos.

Además del espectro suelto, reúne la comparativa de espectros que repetían
todos los cuadernos de entrenamiento: la misma figura de espectro real,
espectro predicho y espectro de Gaia que dibujaba `plot_prediction_example`,
pero sobre los objetos de `grouped_samples_selected_objects.csv` en lugar de
sobre índices sueltos del conjunto de test escritos a mano en cada cuaderno.

Así todos los modelos se ilustran con los mismos objetos —una pareja por cada
zona del diagrama HR—, identificados por su nombre celeste oficial, y las
figuras quedan guardadas en `models/images` con un nombre estable que la
memoria puede citar.
"""

from pathlib import Path
import re

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

# Nombre de cada grupo, por la zona de la secuencia que ocupan sus objetos. El
# color BP-RP crece hacia la derecha del diagrama, es decir, hacia las
# temperaturas más bajas: el grupo de la izquierda es el más caliente
# (Teff ~6 400 K) y el más luminoso, y el de la derecha, el más frío
# (Teff ~3 100 K) y el más débil.
#
# La magnitud absoluta del grupo de la izquierda hay que tomarla con reservas:
# sus paralajes no llegan a 0,01 mas, así que no son significativos y el módulo
# de distancia que sale de ellos tampoco. De ahí que ahí aparezcan objetos
# lejanos —RR Lyrae, cuásares— como si fueran muy luminosos.
GROUP_LABELS = {
    "main_sequence_left":    "Secuencia central, alta luminosidad y alta temperatura"
  , "main_sequence_central": "Secuencia central"
  , "main_sequence_right":   "Secuencia central, baja luminosidad y baja temperatura"
  , "white_dwarfs":          "Enanas blancas"
}

# Grupos cuyo nombre ya dice de qué objetos se trata: repetir ahí el tipo de
# SIMBAD sobra
GROUPS_NAMED_BY_TYPE = frozenset({"white_dwarfs"})

# Tipos de objeto de SIMBAD, en castellano: en el título del grupo el código
# escueto que devuelve el catálogo ('LM*', 'RR*') no se entiende. Se acompaña
# del código entre paréntesis, que es como lo publica el catálogo. Los tipos
# que no estén aquí salen sólo con su código.
OTYPE_LABELS = {
    "LM*": "estrella de baja masa"
  , "RR*": "variable RR Lyrae"
  , "QSO": "cuásar"
  , "WD*": "enana blanca"
}

# La procedencia del nombre y del tipo de cada objeto, al pie de la figura: así
# no hay que repetirla en cada panel
SIMBAD_CREDIT = "Nombre y tipo de cada objeto según SIMBAD"

# Paleta de Okabe-Ito: las tres series se distinguen también con deuteranopía y
# protanopía, al contrario que el azul, naranja y verde por defecto de
# Matplotlib, y todas contrastan sobre blanco al imprimir la memoria.
OBSERVED_COLOR  = "#0072b2"
PREDICTED_COLOR = "#d55e00"
INPUT_COLOR     = "#009e73"

SUBTITLE_COLOR = "#52514e"


# ---------------------------------------------------------------------------
# Puras – sin E/S
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convierte un nombre de SIMBAD en un nombre de fichero manejable."""
    return re.sub(r"[^0-9A-Za-z]+", "-", str(text)).strip("-")


def _group_label(group: str) -> str:
    return GROUP_LABELS.get(group, str(group).replace("_", " "))


def _otype_labels(group: str, group_objects: pd.DataFrame) -> str:
    """
    Tipos de objeto de un grupo, sin repetirlos y en el orden en que aparecen.
    Casi siempre es uno solo; el grupo de paralaje pequeño mezcla dos. Sale
    vacío cuando el nombre del grupo ya dice de qué objetos se trata.
    """
    if group in GROUPS_NAMED_BY_TYPE:
        return ""

    labels = []
    for otype in group_objects["simbad_otype"]:
        if str(otype) in ("", "nan"):
            continue

        label = OTYPE_LABELS.get(otype)
        label = f"{label} ({otype})" if label else str(otype)

        if label not in labels:
            labels.append(label)

    return " y ".join(labels)


def _object_subtitle(selected_object: pd.Series) -> str:
    """
    Segunda línea del título: los identificadores de catálogo, debajo del
    nombre oficial. Ni la zona del diagrama ni el tipo de objeto se repiten
    aquí: encabezan la figura entera.
    """
    parts = [f"Gaia DR3 {selected_object['source_id']}"]

    # La terna placa-MJD-fibra es la referencia con la que SDSS nombra sus
    # espectros, más legible que el identificador numérico
    if not pd.isna(selected_object.get("plate")):
        parts.append(
            "SDSS "
            f"{int(selected_object['plate'])}-"
            f"{int(selected_object['mjd'])}-"
            f"{int(selected_object['fiberID'])}"
        )

    return "  ·  ".join(parts)


def _test_indices(source_ids, test_source_ids):
    """
    Localiza cada objeto seleccionado dentro del conjunto de test.

    Parameters:
        source_ids (Sequence[int]): Identificadores de Gaia de los objetos.
        test_source_ids (numpy.ndarray): Identificadores del conjunto de test.

    Returns:
        list[int | None]: Índice de cada objeto en el conjunto de test, o
            None si no está en él.
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


def plot_selected_objects(
    selection_path
  , test_data_path
  , predictions_path
  , model_name
  , images_dir=IMAGES_DIR
  , prediction_key="y_pred"
):
    """
    Dibuja, para cada zona del diagrama HR, la comparativa de espectros de los
    objetos seleccionados en el CSV.

    Sale una figura por zona, con un panel por objeto: el espectro observado por
    SDSS, el que predice el modelo y el de Gaia que entra en él. Cada panel se
    encabeza con el nombre celeste oficial del objeto y, debajo, sus
    identificadores de catálogo.

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
            ('dense-global', 'cnn-v2', 'gru'...). Encabeza las figuras y nombra
            las imágenes.
        images_dir (str | Path): Carpeta donde guardar las imágenes. Se crea si
            no existe.
        prediction_key (string): Nombre del array de predicciones dentro de su
            `.npz`.

    Returns:
        list[Path]: Rutas de las imágenes guardadas, una por zona.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(selection_path)

    with np.load(test_data_path) as test_data:
        gaia_wavelength = test_data["gaia_wavelength"]
        sdss_wavelength = test_data["sdss_wavelength"]
        test_source_ids = test_data["X_id_test"]
        X_test          = test_data["X_test"]
        y_test          = test_data["y_test"]

    with np.load(predictions_path) as predictions:
        y_pred = predictions[prediction_key]

    selection["test_index"] = _test_indices(selection["source_id"], test_source_ids)

    missing = selection[selection["test_index"].isna()]
    for _, absent_object in missing.iterrows():
        print(
            f"Aviso: {absent_object['simbad_main_id']} "
            f"(Gaia DR3 {absent_object['source_id']}) no está en el conjunto de test"
        )

    image_paths = []

    for group, group_objects in selection.dropna(subset=["test_index"]).groupby("group", sort=False):
        figure, axes = plt.subplots(
            len(group_objects), 1
          , figsize=(11, 3.6 * len(group_objects))
          , squeeze=False
        )

        for axis, (_, selected_object) in zip(axes[:, 0], group_objects.iterrows()):
            index = int(selected_object["test_index"])

            axis.plot(
                sdss_wavelength, y_test[index]
              , label="SDSS observado", color=OBSERVED_COLOR, linewidth=1.0, alpha=0.85
            )
            axis.plot(
                sdss_wavelength, y_pred[index]
              , label="SDSS predicho", color=PREDICTED_COLOR, linewidth=1.2
            )
            axis.plot(
                gaia_wavelength, X_test[index]
              , label="Gaia (entrada)", color=INPUT_COLOR, linewidth=1.4
            )

            # El nombre oficial del objeto encabeza el panel; los
            # identificadores de Gaia y SDSS van debajo, en menor tamaño
            axis.set_title(
                selected_object["simbad_main_id"]
              , fontsize=12, fontweight="bold", pad=20
            )
            axis.text(
                0.5, 1.015, _object_subtitle(selected_object)
              , transform=axis.transAxes, ha="center", va="bottom"
              , fontsize=9, color=SUBTITLE_COLOR
            )

            axis.set_xlabel("Longitud de onda [Å]")
            axis.set_ylabel("Flujo")
            axis.grid(alpha=0.2)

        # El pie de la figura lleva la leyenda y, debajo, la procedencia de los
        # nombres. Las dos van en fracción de figura, así que la altura que se
        # les reserva se reparte entre ambas
        footer_height = 0.42 / figure.get_figheight()

        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles, labels
          , loc="lower center", bbox_to_anchor=(0.5, 0.45 * footer_height)
          , ncol=3, frameon=False
        )
        figure.text(
            0.5, 0.1 * footer_height, SIMBAD_CREDIT
          , ha="center", va="bottom", fontsize=8, color=SUBTITLE_COLOR
        )

        otypes = _otype_labels(group, group_objects)

        figure.suptitle(
            " — ".join(
                part for part in
                (_group_label(group), otypes, f"modelo {model_name}") if part
            )
          , fontsize=13
        )
        figure.tight_layout(rect=(0, footer_height, 1, 0.97))

        image_path = images_dir / f"{model_name}_{group}.png"
        figure.savefig(image_path, dpi=150)
        image_paths.append(image_path)

        plt.show()

    print(f"Imágenes guardadas en {images_dir}")

    return image_paths
