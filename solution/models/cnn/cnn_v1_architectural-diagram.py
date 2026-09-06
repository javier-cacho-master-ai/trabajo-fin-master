# %% [markdown]
# # Diagramas de arquitectura del modelo convolucional 1D
#
# Figuras de la arquitectura del modelo entrenado en `cnn_v1_training.ipynb`, para la memoria. Las dibuja `services/architecture_diagram.py`, el módulo que comparten todos los cuadernos de diagramas de `models`:
#
# - La vista `graph` que **visualkeras** saca del modelo ya entrenado, sin escribir la estructura a mano.
#
# La arquitectura descrita es la de `cnn_v1_training.ipynb`, así que hay que repasarla si cambia la red.

# %%
import sys

from pathlib import Path

# 'services' está tanto en 'solution' como en 'models', así que las dos carpetas
# van a 'sys.path'. 'solution' se localiza subiendo desde el directorio de
# trabajo, sin suponer dónde arranca el kernel: su marca es 'pyproject.toml'.
DIRECTORIES = (Path.cwd(), *Path.cwd().parents)
CANDIDATES = (*DIRECTORIES, *(directory / "solution" for directory in DIRECTORIES))
SOLUTION_DIR = next(
    candidate for candidate in CANDIDATES if (candidate / "pyproject.toml").is_file()
)

sys.path.insert(0, str(SOLUTION_DIR / "models"))
sys.path.insert(0, str(SOLUTION_DIR))

from services.architecture_diagram import DIAGRAMS_DIR, draw_diagrams
from services.paths import model_paths

MODEL_PATH = model_paths("cnn_v1", "cnn").model

print("Carpeta de trabajo:", SOLUTION_DIR)
print("Figuras en:", DIAGRAMS_DIR)

# %% [markdown]
# El `.keras` guarda la arquitectura junto con los pesos. El resumen sirve para comprobar contra el modelo entrenado las formas y el número de filtros que se describen aquí. La vista de visualkeras sí sale del modelo cargado.

# %%
import tensorflow as tf

model_cnn = tf.keras.models.load_model(MODEL_PATH)

model_cnn.summary()

# %% [markdown]
# La arquitectura de la red es la siguiente:
#
# - La red trabaja siempre sobre la malla de SDSS, sin submuestrear: los 2666 puntos se conservan de principio a fin, y por eso la red es una cadena y no una U como la de la v2.
# - Tras la convolución inicial de 64 filtros vienen los **cuatro bloques residuales**, todos iguales, que la figura agrupa en una sola caja: lo que cambia entre ellos son los pesos, no la forma.
# - La convolución final de un filtro genera la **corrección**, y la **conexión residual global** —la flecha discontinua— le suma el flujo interpolado: la red aprende la corrección, no el espectro entero.
#
# Cada bloque residual son dos convoluciones, el atajo que se les suma y la activación.
#
# Y cómo se lee la vista `graph` de visualkeras, que salen del `.keras` y por tanto enseñan la red tal y como quedó y no como se pretendía que fuera:
#
# - `graph`: las neuronas de cada capa y las conexiones entre ellas, resumidas con puntos suspensivos.

# %%
draw_diagrams(model_cnn, "cnn_v1", views=("graph", "layered"))
