# %% [markdown]
# # Diagramas de arquitectura del modelo denso
#
# Figuras de la arquitectura de los modelos entrenados en `dense_training.ipynb`, para la memoria. Las dibuja `models/services/architecture_diagram.py`, el módulo que comparten todos los cuadernos de diagramas de `models`:
#
# - La vista `graph` que **visualkeras** saca del modelo ya entrenado, sin escribir la estructura a mano.
#
# El cuaderno de entrenamiento deja **tres** modelos —`dense-global`, `dense-median` y `dense-max`—, pero los tres son la misma red: lo único que cambia entre ellos es la normalización con la que se les da el espectro. Por eso hay una sola tanda de figuras, nombrada `dense`, y por eso la celda siguiente comprueba contra los modelos entrenados que las tres arquitecturas coinciden de verdad.
#
# La arquitectura descrita es la de `dense_training.ipynb`, así que hay que repasarla si cambia la red.

# %%
import sys

from pathlib import Path

# 'services' cuelga de 'models', así que esa carpeta va a 'sys.path'. 'solution'
# se localiza subiendo desde el directorio de trabajo, sin suponer dónde arranca
# el kernel: su marca es 'pyproject.toml'.
DIRECTORIES = (Path.cwd(), *Path.cwd().parents)
CANDIDATES = (*DIRECTORIES, *(directory / "solution" for directory in DIRECTORIES))
SOLUTION_DIR = next(
    candidate for candidate in CANDIDATES if (candidate / "pyproject.toml").is_file()
)

sys.path.insert(0, str(SOLUTION_DIR / "models"))

from services.architecture_diagram import DIAGRAMS_DIR, draw_diagrams, layer_signature
from services.paths import model_paths

MODEL_PATHS = {
    "dense-global": model_paths("dense-global", "dense").model
  , "dense-median": model_paths("dense-median", "dense").model
  , "dense-max":    model_paths("dense-max", "dense").model
}

print("Carpeta de trabajo:", SOLUTION_DIR)
print("Figuras en:", DIAGRAMS_DIR)

# %% [markdown]
# El `.keras` guarda la arquitectura junto con los pesos. El resumen sirve para comprobar contra los modelos entrenados las formas y el número de neuronas que se describen aquí. La vista de visualkeras sí sale del modelo cargado.
#
# La comparación se hace sobre la clase y la forma de salida de cada capa, no sobre la configuración completa: Keras numera los nombres de las capas de forma correlativa dentro de la sesión, así que las tres redes traen nombres distintos —`dense_5`, `dense_12`…— aunque sean idénticas.

# %%
import tensorflow as tf

models = {name: tf.keras.models.load_model(path) for name, path in MODEL_PATHS.items()}
signatures = {name: layer_signature(model) for name, model in models.items()}

# Una sola tanda de figuras para los tres modelos solo vale si son la misma red
assert len(set(signatures.values())) == 1, signatures

models["dense-global"].summary()

# %% [markdown]
# La arquitectura de la red es la siguiente:
#
# - La entrada son los **201 puntos** del espectro XP de Gaia, que la red toma como un vector suelto: sin convoluciones ni recurrencia, ninguna neurona sabe que dos puntos del vector son longitudes de onda vecinas. La relación entre puntos contiguos, si hace falta, tiene que aprenderla de los datos.
# - Vienen después **cuatro capas ocultas** —una de 512 y tres de 1024— que alternan dos activaciones: `LeakyReLU` con pendiente 0,01 para el lado negativo y `ELU`. Todas son capas completamente conectadas, sin regularización ni normalización por lotes.
# - La salida es una capa **lineal de 2666 neuronas**, el espectro de SDSS normalizado entero: la red produce de una vez todos los puntos de la malla de SDSS, en lugar de recorrerla. Ella sola se lleva 2,73 de los 5,46 millones de parámetros de la red, la mitad del modelo.
#
# Lo único que separa a los tres modelos del cuaderno es la normalización con la que se les da el espectro, que es la misma que después deshacen sobre la predicción. Las tres están en `models/services/normalization.py`, y ninguna forma parte de la red: por eso no aparecen en las figuras.
#
# Y cómo se lee la vista `graph` de visualkeras, que salen del `.keras` y por tanto enseñan la red tal y como quedó y no como se pretendía que fuera:
#
# - `graph`: las neuronas de cada capa y las conexiones entre ellas, resumidas con puntos suspensivos. Es la vista que enseña que en esta red **todo está conectado con todo**, que es justamente lo que la distingue de las demás arquitecturas del trabajo.

# %%
# Los nombres de capa de esta red son cortos, así que sus columnas van más
# apretadas que las comunes: la figura sale menos apaisada y, al reducirla, sus
# pies se leen mayores
draw_diagrams(models["dense-global"], "dense", graph={"layer_spacing": 150})
