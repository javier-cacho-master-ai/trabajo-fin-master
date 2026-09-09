# %% [markdown]
# # Diagramas de arquitectura del modelo convolucional 1D v2 (U-Net residual)
#
# Figuras de la arquitectura del modelo entrenado en `cnn-v2_training.ipynb`, para la memoria. Las dibuja `models/services/architecture_diagram.py`, el módulo que comparten todos los cuadernos de diagramas de `models`:
#
# - La vista `graph` que **visualkeras** saca del modelo ya entrenado, sin escribir la estructura a mano. Aquí son justamente esas tiras, así que van con las cajas más juntas y sin rótulos, para que al menos la forma de la red se distinga.
# - La vista **`layered`**: las 95 capas apiladas en plano y pegadas unas a otras. Es la única que cabe entera en el ancho de una página, porque la `graph`, con una columna por capa, sale como una tira de miles de píxeles de ancho.
#
# La arquitectura descrita es la de `cnn-v2_training.ipynb`, así que hay que repasarla si cambia la red.

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

from PIL import ImageFont

from services.architecture_diagram import DIAGRAMS_DIR, FONT_PATH, draw_diagrams
from services.paths import model_paths

MODEL_PATH = model_paths("cnn-v2", "cnn").model

print("Carpeta de trabajo:", SOLUTION_DIR)
print("Figuras en:", DIAGRAMS_DIR)

# %% [markdown]
# El `.keras` guarda la arquitectura junto con los pesos. El resumen sirve para comprobar contra el modelo entrenado las formas y el número de filtros que se describen aquí. La vista de visualkeras sí sale del modelo cargado.

# %%
import tensorflow as tf

model_cnn_v2 = tf.keras.models.load_model(MODEL_PATH)

model_cnn_v2.summary()

# %% [markdown]
# La arquitectura de la red es la siguiente:
#
# - La rama izquierda es el **codificador**: cada nivel es un bloque residual con atención de canal, y entre niveles una convolución de paso 2 que reduce a la mitad la longitud del espectro (2672 → 1336 → 668 → 334 puntos). El número de filtros crece al bajar (64 → 96 → 128 → 160) para compensar la pérdida de resolución.
# - Abajo está el **cuello de botella**, dos bloques residuales de 160 filtros que ven el espectro completo a resolución 1/8.
# - La rama derecha es el **decodificador**, simétrico: sobremuestreo por 2, concatenación de la conexión de salto del nivel correspondiente y un bloque residual, hasta volver a la malla original.
# - Las **conexiones de salto** unen cada nivel del codificador con el del decodificador que le corresponde, y devuelven a este el detalle fino que el submuestreo pierde.
# - Arriba, la **conexión residual global**: la red aprende la corrección y la suma al flujo interpolado, no genera el espectro desde cero.
#
# Cada bloque residual con atención de canal son las dos convoluciones, la atención de canal (*squeeze-and-excitation*), que pondera cada filtro con un factor aprendido, y el atajo que las suma.
#
# Y cómo se leen las dos vistas de visualkeras, que salen del `.keras` y por tanto enseñan la red tal y como quedó y no como se pretendía que fuera:
#
# - `graph`: las neuronas de cada capa y las conexiones entre ellas, resumidas con puntos suspensivos.
# - `layered`: las 95 capas en plano, una barra por capa y del color de lo que hace. Es la que cuenta cuántas capas tiene de verdad la red. La altura de cada barra es proporcional al número de filtros o de unidades de la capa, así que crece del primer nivel del codificador al cuello de botella, y en ella se distingue el patrón que se repite en los ocho bloques residuales.

# %%
# visualkeras hace cada entrada de la leyenda tan ancha como su rótulo y luego
# las va partiendo en filas, así que con doce clases de capa salen desiguales.
# Midiendo todos los rótulos como el más largo, las entradas salen del mismo
# ancho y su propio reparto las deja alineadas en columnas.
LEGEND_FONT = ImageFont.truetype(FONT_PATH, 48)
LEGEND_BOX = (
    0
  , 0
  , round(max(map(LEGEND_FONT.getlength, (type(l).__name__ for l in model_cnn_v2.layers))))
    # El alto sí es el de la letra: es el que le deja sitio a las jotas y las ges
  , LEGEND_FONT.getbbox("Ag")[3]
)
LEGEND_FONT.getbbox = lambda text, *args, **kwargs: LEGEND_BOX

# Las 95 capas del modelo no caben en las figuras con las opciones comunes: con
# el hueco que estas dejan para el rótulo de cada capa, las vistas salen de más
# de diez mil píxeles de ancho. Son las únicas figuras del trabajo que
# necesitan retoque, así que se le pasa aquí y no al servicio.
draw_diagrams(
    model_cnn_v2, "cnn-v2"
  , views=("graph", "layered")
    # Sin los recuadros que rotulan cada capa ni la separación que piden sus
    # pies: con 95 capas, la figura saldría de más de veinte mil píxeles de
    # ancho y los pies, reducidos a una columna, no se leerían de todos modos
  , graph={
        "layer_spacing": 60, "node_size": 20, "ellipsize_after": 4
      , "layered_groups": [], "padding": 24
    }
    # Sin rotular y con las capas muy juntas: son 95, y sus formas, reducidas
    # al ancho de una página, saldrían en menos de dos puntos. El hueco no baja
    # de 8 porque es también el que la leyenda deja entre cada color y su texto
    # La leyenda, con letra aún mayor: esta figura sale dos veces y media más
    # ancha que las demás, así que se reduce otro tanto en la página
  , layered={"spacing": 8, "text_callable": None, "font": LEGEND_FONT}
)
