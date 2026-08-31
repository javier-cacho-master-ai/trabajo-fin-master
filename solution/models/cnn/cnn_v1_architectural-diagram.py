# %% [markdown]
# # Diagrama de arquitectura del modelo convolucional 1D
#
# Figura de la arquitectura del modelo entrenado en `cnn_v1_training.ipynb`, para la memoria. Es un **esquema por etapas**, con el mismo planteamiento que el de la v2: una caja por etapa, no por capa, porque una caja por capa sale como una tira demasiado apaisada para una página.
#
# El esquema se escribe en Graphviz, que ya entra con `pydot`. La estructura es la de `cnn_v1_training.ipynb`, así que hay que repasarla si cambia la red.

# %%
from pathlib import Path

# Localizamos la carpeta 'solution' subiendo desde el directorio de trabajo, sin
# suponer dónde arranca el kernel: sirve tanto la carpeta de este cuaderno como
# 'models' o la raíz del repositorio. Su marca es el fichero 'pyproject.toml'.
SOLUTION_DIR = next(
    candidate
    for parent in (Path.cwd(), *Path.cwd().parents)
    for candidate in (parent, parent / "solution")
    if (candidate / "pyproject.toml").is_file()
)

MODELS_DIR = SOLUTION_DIR / "models"
CNN_DIR = MODELS_DIR / "cnn"

# Modelo a dibujar y figura que genera este cuaderno
MODEL_PATH = CNN_DIR / "cnn_v1.keras"
DIAGRAM_PATH = CNN_DIR / "cnn_v1_architectural-diagram.png"

print("Carpeta de trabajo:", SOLUTION_DIR)

# %% [markdown]
# El `.keras` guarda la arquitectura junto con los pesos. El resumen no genera la figura —el esquema está escrito a mano—, pero sirve para comprobar contra el modelo entrenado las formas y el número de filtros que rotula el diagrama.

# %%
import tensorflow as tf

model_cnn = tf.keras.models.load_model(MODEL_PATH)

model_cnn.summary()

# %% [markdown]
# Cómo se lee el esquema:
#
# - La red trabaja siempre sobre la malla de SDSS, sin submuestrear: los 2666 puntos se conservan de principio a fin, y por eso el esquema es una cadena y no una U como el de la v2.
# - Tras la convolución inicial de 64 filtros vienen los **cuatro bloques residuales**, todos iguales, que la figura agrupa en una sola caja: lo que cambia entre ellos son los pesos, no la forma.
# - La convolución final de un filtro genera la **corrección**, y la **conexión residual global** —la flecha discontinua— le suma el flujo interpolado: la red aprende la corrección, no el espectro entero.
#
# El recuadro de la derecha desarrolla uno de los bloques residuales: las dos convoluciones, el atajo que se les suma y la activación.

# %%
import pydot

from IPython.display import Image

DOT = """digraph cnn_residual {
    rankdir=TB
    bgcolor="white"
    splines=polyline
    dpi=150
    nodesep=0.35
    ranksep=0.45
    node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5b6b7c"
          fontname="DejaVu Sans" fontsize=11 margin="0.18,0.10"]
    edge [fontname="DejaVu Sans" fontsize=9 color="#5b6b7c"]

    entrada [label="flujo interpolado\n2666 × 1" fillcolor="#dfe7ef"]
    conv0   [label="Conv1D 64 (k9) + ELU\n64 @ 2666"]
    bloques [label="4 × bloque residual\n64 @ 2666" fillcolor="#d7e3f0"]
    conv1   [label="Conv1D 1 (k5)\ncorrección, 2666"]
    salida  [label="⊕ flujo interpolado\n2666" fillcolor="#dfe7ef"]

    entrada -> conv0 -> bloques -> conv1 -> salida
    entrada -> salida [constraint=false style=dashed color="#9aa4ad"
                       label="  conexión residual global"]

    subgraph cluster_bloque {
        label="bloque residual"
        fontname="DejaVu Sans" fontsize=11 fontcolor="#5b6b7c"
        color="#c3ccd6" style=rounded margin=12

        b_in  [label="x\n64 @ 2666" fillcolor="#dfe7ef"]
        b_c1  [label="Conv1D 64 (k7) + ELU"]
        b_c2  [label="Conv1D 64 (k7)"]
        b_add [label="⊕" shape=circle width=0.3 fixedsize=true fillcolor="#dfe7ef"]
        b_out [label="ELU\n64 @ 2666" fillcolor="#dfe7ef"]

        b_in -> b_c1 -> b_c2 -> b_add -> b_out
        b_in -> b_add [constraint=false style=dashed color="#9aa4ad" label="  atajo"]
    }
}"""

(graph,) = pydot.graph_from_dot_data(DOT)
graph.write_png(str(DIAGRAM_PATH))

print("Diagrama guardado en:", DIAGRAM_PATH)

Image(str(DIAGRAM_PATH))
