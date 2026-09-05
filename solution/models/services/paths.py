"""
Rutas de los ficheros que dejan los cuadernos de `models`.

Servicio compartido por todos los cuadernos de entrenamiento. Todos escriben y
leen los mismos ficheros con los mismos nombres, así que la convención que los
nombra vive aquí y no repetida en la celda de rutas de cada cuaderno:

- `<carpeta>/<modelo>.keras`: los pesos de la mejor época de validación.
- `<carpeta>/training/<modelo>_training_history.csv`: el historial por época.
- `<carpeta>/training/<modelo>_training_summary.json`: el resumen del
  entrenamiento.
- `data/predictions/<modelo>_predictions.npz`: las predicciones sobre el
  conjunto de test.
- `data/predictions/ispec_<modelo>.csv`: la tabla del análisis con iSpec de esas
  predicciones. Las dos últimas no son un checkpoint del entrenamiento, sino el
  dato que consume la comparativa entre arquitecturas, así que viven fuera de la
  carpeta del cuaderno.
- `models/data/test/<cuaderno>_test_data.npz`: el conjunto de test que consumen
  las celdas de análisis. Lo nombra el cuaderno y no el modelo porque un
  cuaderno cuyos modelos comparten normalización guarda uno solo para todos.

Las carpetas de las rutas devueltas quedan creadas, de modo que el cuaderno
pueda escribir en ellas sin comprobarlo.

La carpeta `solution` se deduce de la posición de este fichero, así que el
cuaderno no necesita pasarla. Sí necesita localizarla por su cuenta antes de
importar este módulo: `services` vive en `models`, y esa carpeta no está en
`sys.path` hasta que el propio cuaderno la añade.
"""

from pathlib import Path
from typing import NamedTuple

# `services` cuelga de `models`, que cuelga de `solution`
SOLUTION_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = SOLUTION_DIR / "data"
MODELS_DIR = SOLUTION_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
TEST_DATA_DIR = MODELS_DIR / "data" / "test"


class ModelPaths(NamedTuple):
    """Ficheros de un modelo, en el orden en que los deja el cuaderno."""

    model: Path
    history: Path
    summary: Path
    predictions: Path
    ispec: Path


def model_paths(model_name, model_dir):
    """
    Rutas de los ficheros de un modelo entrenado.

    Parameters:
        model_name (str): Nombre del modelo, el que llevan todos sus ficheros
            ('cnn-v2', 'dense-median', 'gru'...).
        model_dir (str): Carpeta del cuaderno dentro de `models` ('cnn',
            'dense', 'rnn'...).

    Returns:
        ModelPaths: Rutas del modelo, de su historial, de su resumen, de sus
            predicciones y de su tabla de iSpec, en ese orden y con sus
            carpetas ya creadas.
    """
    notebook_dir = MODELS_DIR / model_dir
    training_dir = notebook_dir / "training"

    training_dir.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    return ModelPaths(
        model=notebook_dir / f"{model_name}.keras",
        history=training_dir / f"{model_name}_training_history.csv",
        summary=training_dir / f"{model_name}_training_summary.json",
        predictions=PREDICTIONS_DIR / f"{model_name}_predictions.npz",
        ispec=PREDICTIONS_DIR / f"ispec_{model_name}.csv",
    )


def test_data_path(notebook_name):
    """
    Ruta del `.npz` con el conjunto de test que guarda un cuaderno.

    Parameters:
        notebook_name (str): Nombre con el que el cuaderno guarda su conjunto
            de test, uno por cada preparación distinta ('cnn-v2', 'rnn',
            'dense-median'...).

    Returns:
        Path: Ruta del `.npz`, con su carpeta ya creada.
    """
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    return TEST_DATA_DIR / f"{notebook_name}_test_data.npz"
