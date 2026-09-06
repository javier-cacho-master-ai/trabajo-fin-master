# %% [markdown]
# # Diagramas de arquitectura del modelo denso con la SNR
#
# Figuras de la arquitectura del modelo entrenado en `snr_training.ipynb`, para la memoria. Las dibuja `models/services/architecture_diagram.py`, el módulo que comparten todos los cuadernos de diagramas de `models`:
#
# - La vista `graph` que **visualkeras** saca del modelo ya entrenado, sin escribir la estructura a mano.
#
# Este modelo es la red densa de `dense_training.ipynb` con **la SNR del espectro de Gaia añadida a la entrada**: las capas son exactamente las mismas y lo único que cambia es que la entrada pasa de 201 a 402 números. La celda siguiente lo comprueba contra los dos modelos entrenados.
#
# La arquitectura descrita es la de `snr_training.ipynb`, así que hay que repasarla si cambia la red.

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

# El modelo a dibujar y el denso con el que se compara
MODEL_PATH = model_paths("snr", "snr").model
DENSE_MODEL_PATH = model_paths("dense-median", "dense").model

print("Carpeta de trabajo:", SOLUTION_DIR)
print("Figuras en:", DIAGRAMS_DIR)

# %% [markdown]
# El `.keras` guarda la arquitectura junto con los pesos. El resumen sirve para comprobar contra el modelo entrenado las formas y el número de neuronas que se describen aquí. La vista de visualkeras sí sale del modelo cargado.
#
# La comparación con el modelo denso se hace sobre la clase y la forma de salida de cada capa: es justamente la entrada lo que distingue a las dos redes, y el resto tiene que coincidir para que lo que aquí se describe sea cierto.

# %%
import tensorflow as tf

model_snr = tf.keras.models.load_model(MODEL_PATH)
model_dense = tf.keras.models.load_model(DENSE_MODEL_PATH)

# Las dos redes son la misma salvo por la entrada, que aquí lleva la SNR
# concatenada al espectro, y esta descripción solo vale si es cierto
assert layer_signature(model_snr) == layer_signature(model_dense), (
    layer_signature(model_snr), layer_signature(model_dense)
)
assert model_snr.inputs[0].shape[1] == 2 * model_dense.inputs[0].shape[1]

model_snr.summary()

# %% [markdown]
# La arquitectura de la red es la siguiente:
#
# - La entrada son **402 números**: los 201 puntos del espectro XP de Gaia y, concatenados detrás, los 201 valores de la SNR de ese mismo espectro. Cada mitad se normaliza a su manera antes de entrar en la red: el espectro se divide entre la mediana de su flujo absoluto, espectro a espectro, mientras que la SNR se tipifica punto a punto con la media y la desviación típica del conjunto de entrenamiento.
# - Las **cuatro capas ocultas** —una de 512 y tres de 1024, alternando `LeakyReLU` con pendiente 0,01 y `ELU`— y la **salida lineal de 2666** son exactamente las del modelo denso. La red no sabe que la segunda mitad de su entrada es una medida de calidad y no un flujo: eso tiene que aprenderlo de los datos, como cualquier otra relación entre las entradas.
# - Los 201 números de más de la entrada añaden 103 000 parámetros a la primera capa oculta, un 1,9 % de los 5,56 millones del modelo. Es todo el coste de la prueba.
#
# Y cómo se lee la vista `graph` de visualkeras, que salen del `.keras` y por tanto enseñan la red tal y como quedó y no como se pretendía que fuera:
#
# - `graph`: las neuronas de cada capa y las conexiones entre ellas, resumidas con puntos suspensivos.

# %%
# Las columnas, apretadas como las de la red densa: es la misma red y sus pies
# son igual de cortos
draw_diagrams(model_snr, "snr", graph={"layer_spacing": 150})
