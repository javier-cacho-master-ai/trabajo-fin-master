"""
Diagramas de arquitectura de los modelos entrenados en `models`.

Servicio compartido por los cuadernos que dibujan diagramas. Todas las figuras
las dibuja visualkeras a partir del `.keras`, de modo que enseñan la red tal y
como quedó entrenada y no como se pretendía que fuera: aquí no se describe
ninguna arquitectura a mano. Van a `models/images/architectural_diagrams`, con
el nombre del modelo por delante y el de la vista detrás,
`<modelo>_<vista>-diagram.png`. Salen las de `DEFAULT_VIEWS`, salvo que se pidan
otras por su nombre:

- `graph`: las neuronas de cada capa y las conexiones entre ellas, con la clase
  y la forma de cada una al pie.
- `layered`: las capas apiladas en plano de la entrada a la salida, una barra
  por capa y con la leyenda de colores. La pide quien tenga tantas capas que no
  quepan en la anterior.

Los colores no los reparte visualkeras: los fija `LAYER_COLORS` por lo que hace
cada capa, de modo que signifiquen lo mismo en las figuras de todos los modelos.
"""

import visualkeras

from IPython.display import Image, display
from keras.layers import InputLayer
from matplotlib import font_manager
from PIL import ImageFont

from services.paths import MODELS_DIR

# `models` lo deduce `paths.py` de su propia posición, de modo que las figuras
# se localicen desde ahí y no desde el directorio en el que arranque el kernel
# de quien llama
DIAGRAMS_DIR = MODELS_DIR / "images" / "architectural_diagrams"

# ---------------------------------------------------------------------------
# Vistas de visualkeras
# ---------------------------------------------------------------------------

# Paleta de Okabe-Ito, la misma que las gráficas de
# `solution/services/plotting.py`: los colores contrastan bien y las capas 
# se agrupan por lo que hacen y no por su clase, que es lo que interesa comparar
# de una red a otra.
INPUT_COLOR       = "#dfe7ef"  # la entrada, en un gris azulado
DENSE_COLOR       = "#56b4e9"  # capas completamente conectadas
CONVOLUTION_COLOR = "#009e73"  # convoluciones
RECURRENT_COLOR   = "#0072b2"  # capas recurrentes
ACTIVATION_COLOR  = "#e69f00"  # activaciones
MERGE_COLOR       = "#d55e00"  # uniones de ramas: sumas, concatenaciones...
RESHAPE_COLOR     = "#cc79a7"  # cambios de forma: recortes, remuestreos...
POOLING_COLOR     = "#f0e442"  # resúmenes de la secuencia entera
OTHER_COLOR       = "#eef2f7"  # cualquier clase que no esté en la lista

# Las clases van por su nombre para que entren también las capas escritas a
# mano, sin que este módulo tenga que importarlas. Los tensores de entrada y de
# salida que la vista `graph` añade por su cuenta no son capas del modelo, así
# que los colorea visualkeras y no esta tabla.
LAYER_COLORS = {
    "InputLayer":             INPUT_COLOR
  , "Dense":                  DENSE_COLOR
  , "Conv1D":                 CONVOLUTION_COLOR
  , "GRU":                    RECURRENT_COLOR
  , "LSTM":                   RECURRENT_COLOR
  , "SimpleRNN":              RECURRENT_COLOR
  , "Bidirectional":          RECURRENT_COLOR
  , "Activation":             ACTIVATION_COLOR
  , "LeakyReLU":              ACTIVATION_COLOR
  , "ELU":                    ACTIVATION_COLOR
  , "ReLU":                   ACTIVATION_COLOR
  , "Add":                    MERGE_COLOR
  , "Concatenate":            MERGE_COLOR
  , "Multiply":               MERGE_COLOR
  , "Reshape":                RESHAPE_COLOR
  , "Cropping1D":             RESHAPE_COLOR
  , "ZeroPadding1D":          RESHAPE_COLOR
  , "UpSampling1D":           RESHAPE_COLOR
  , "GlobalAveragePooling1D": POOLING_COLOR
  , "AttentionPooling":       POOLING_COLOR
}

# Nombre de cada clase de capa en castellano, para los rótulos. Se quedan como
# están los nombres propios —las funciones de activación y las siglas—, que no
# se traducen. La clase que no esté aquí se rotula con su nombre de Keras.
LAYER_NAMES = {
    "InputLayer":             "Entrada"
  , "Dense":                  "Densa"
  , "Conv1D":                 "Convolución 1D"
  , "Bidirectional":          "Bidireccional"
  , "Activation":             "Activación"
  , "Add":                    "Suma"
  , "Concatenate":            "Concatenación"
  , "Multiply":               "Multiplicación"
  , "Reshape":                "Cambio de forma"
  , "Cropping1D":             "Recorte 1D"
  , "ZeroPadding1D":          "Relleno de ceros 1D"
  , "UpSampling1D":           "Sobremuestreo 1D"
  , "GlobalAveragePooling1D": "Media global 1D"
  , "AttentionPooling":       "Atención"
  , "SimpleRNN":              "RNN simple"
}

# Sin tipografía visualkeras rotula con una de mapa de bits ilegible al
# imprimir. DejaVu Sans entra con Matplotlib, así que todas las figuras de un
# modelo salen con la misma letra. Los recuadros de la vista `graph` la quieren
# por su ruta y no ya cargada: solo admiten una cadena o la de mapa de bits.
FONT_SIZE = 12
FONT_PATH = font_manager.findfont("Calibri")
FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# La vista `graph` sale mucho más apaisada que las demás —una columna por capa,
# con las conexiones entre ellas—, así que hay que reducirla mucho más para
# encajarla en un ancho dado. Sus pies van en un cuerpo mayor para que, ya
# reducidos, se lean como los rótulos de las otras figuras.
GRAPH_FONT_SIZE = 18 

# La vista `layered` se lee por su leyenda, y con el cuerpo común esta sale
# ilegible en cuanto se reduce la figura: la vista entera lleva la letra mayor.
LEGEND_FONT = ImageFont.truetype(FONT_PATH, 26)

# Recuadro con el que la vista `graph` rotula cada capa: en gris y sin relleno,
# para que se vean las conexiones que lo cruzan
GROUP_STYLE = {
    "fill":         (255, 255, 255, 0)
  , "outline":      "#c3ccd6"
  , "padding":      12
  , "text_spacing": 10
  , "font":         FONT_PATH
  , "font_size":    GRAPH_FONT_SIZE
  , "font_color":   "black"
}


def layer_shape(layer):
    """
    Forma de salida de una capa, sin el eje del lote y ya escrita.

    Parameters:
        layer (keras.layers.Layer): Capa de un modelo cargado.

    Returns:
        str: Las dimensiones separadas por aspas ('2666 × 64').
    """
    return " × ".join(map(str, layer.output.shape[1:]))


def layer_signature(model):
    """
    Clase y forma de salida de cada capa, con las que comparar dos modelos.

    Parameters:
        model (keras.Model): Modelo entrenado, recién cargado de su '.keras'.

    Returns:
        tuple: Una pareja (clase, forma de salida) por capa.
    """
    return tuple((type(layer).__name__, layer_shape(layer)) for layer in model.layers)


def layer_name(layer):
    """
    Nombre de una capa en castellano, el de `LAYER_NAMES` por su clase.

    Parameters:
        layer (keras.layers.Layer): Capa de un modelo cargado.

    Returns:
        str: El nombre traducido, o el de la clase si no está en la tabla.
    """
    return LAYER_NAMES.get(type(layer).__name__, type(layer).__name__)


def _layer_label(layer):
    """
    Rótulo de una capa: su nombre y su forma de salida, en una sola línea.

    En una y no en dos porque visualkeras mide el rótulo entero como si fuese
    una línea: partido, lo mide más ancho de lo que es y lo centra desplazado
    a la izquierda de la capa que rotula.
    """
    return f"{layer_name(layer)} {layer_shape(layer)}"


def _shape_label(index, layer):
    """
    Rótulo de la vista `layered`: la forma de salida, partida en líneas.

    La clase la da la leyenda, así que el rótulo solo lleva la forma. Y va
    partida porque es su ancho el que separa una capa de la siguiente: en
    varias líneas la figura sale bastante menos apaisada y, al reducirla, el
    rótulo se lee mayor. Aquí sí se puede partir, al contrario que en los pies
    de la vista `graph`: esta mide cada línea por separado.

    Parameters:
        index (int): Posición de la capa, que exige visualkeras y no se usa.
        layer (keras.layers.Layer): Capa a rotular.

    Returns:
        tuple: El rótulo y si va encima de la caja, como lo espera visualkeras.
    """
    return layer_shape(layer).replace(" × ", "\n× "), False


def layer_groups(model):
    """
    Recuadros con los que la vista `graph` rotula las capas de un modelo.

    Es la única vista que no admite rótulos por capa, así que la clase y la
    forma de salida de cada una entran como el pie de un recuadro alrededor de
    sus neuronas. Los tensores de entrada y de salida no llevan: los añade la
    propia vista y no son capas del modelo.

    Parameters:
        model (keras.Model): Modelo entrenado, recién cargado de su '.keras'.

    Returns:
        list: Un recuadro por capa, como los espera el parámetro
            `layered_groups` de visualkeras.
    """
    return [
        {"layers": [layer], "name": _layer_label(layer), **GROUP_STYLE}
        for layer in model.layers
    ]


# Entre el vector de unos cientos de puntos de una entrada y los miles por
# decenas de filtros de una convolución hay varios órdenes de magnitud, y a
# escala no caben en la misma figura: con la escala logarítmica se sigue viendo
# cuál es mayor sin que se desborde. La escala multiplica al logaritmo decimal
# (y a un 20 interno), así que con 5 un tensor de unos pocos miles queda en unos
# 340 píxeles y el menor en 18. Los vectores se dibujan a lo alto, como columnas.
SIZING_OPTIONS = {
    "sizing_mode":         "logarithmic"
  , "one_dim_orientation": "y"
  , "scale_xy":            5
  , "scale_z":             5
  , "min_xy":              18
  , "min_z":               18
  , "max_xy":              480
  , "max_z":               260
}

VIEW_OPTIONS = {
    # `ellipsize_after` es cuántas neuronas dibuja de cada capa antes de
    # resumir el resto con unos puntos suspensivos: una capa de 1024 saldría si
    # no como una columna de 1024 círculos. Los rótulos los ponen los recuadros
    # de `layer_groups`, porque esta vista no admite `text_callable`.
    "graph": {
        "show_neurons":    True
      , "ellipsize_after": 6
      , "node_size":       32
        # Los pies son más anchos que la columna que rotulan, así que las
        # columnas van separadas por lo que ocupa un nombre de capa largo.
        # Quien los tenga cortos puede apretarlas más.
      , "layer_spacing":  100 
        # visualkeras calcula el alto del lienzo con las columnas y le suma
        # este margen arriba y abajo; los pies cuelgan por debajo de la
        # columna más alta, así que el margen es también su hueco
      , "padding":         60
      , "connector_width": 2
    }
    # La pila de capas de la entrada a la salida, cada una como una barra cuya
    # altura sigue al tamaño de su tensor. Es la vista que cuenta cuántas capas
    # hay en la red, y la que sigue cabiendo cuando son tantas que ni se
    # pueden rotular: entonces se lee con la leyenda de colores.
  , "layered": {
        **SIZING_OPTIONS
      , "legend":      True
        # En plano y no en volumen: si las barras van pegadas unas a otras, las
        # caras en perspectiva de cada una tapan la siguiente, y lo que hay que
        # leer es su altura, cómo crece y mengua el tensor al atravesar la red
      , "draw_volume": False
        # Sin los embudos entre capas: en cuanto hay muchas, sus líneas tapan
        # las barras
      , "draw_funnel": False
        # El hueco entre capas, que es también el que la leyenda deja entre
        # cada color y su texto, y el que separa un rótulo del siguiente: no
        # baja de 50 porque con la letra de `LEGEND_FONT` los rótulos se pisan
      , "spacing":     50
      , "padding":     24
      , "text_callable": _shape_label
      , "font":        LEGEND_FONT
    }
}

# Las vistas que se dibujan si no se piden otras
DEFAULT_VIEWS = ("graph",)

# Opciones que dependen del modelo y no pueden ir con las demás, que son las
# mismas para todos. Las vistas que no estén aquí no llevan ninguna.
VIEW_EXTRAS = {"graph": lambda model: {"layered_groups": layer_groups(model)}}


def color_map(model):
    """
    Colores de las capas de un modelo, los de `LAYER_COLORS` por su clase.

    Parameters:
        model (keras.Model): Modelo entrenado, recién cargado de su '.keras'.

    Returns:
        dict: Relleno de cada clase de capa, como lo espera visualkeras, que
            lo busca por la clase misma y no por su nombre. Entra también la
            entrada, que un modelo secuencial no cuenta entre sus capas.
    """
    classes = {InputLayer, *map(type, model.layers)}

    return {
        cls: {"fill": LAYER_COLORS.get(cls.__name__, OTHER_COLOR)} for cls in classes
    }


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def diagram_path(model_name, kind):
    """
    Ruta de una figura de un modelo.

    Parameters:
        model_name (str): Nombre del modelo, el que llevan sus ficheros, o el
            de la familia cuando varios modelos comparten arquitectura.
        kind (str): Nombre de la vista, una de las claves de `VIEW_OPTIONS`.

    Returns:
        Path: Fichero PNG de la figura, con su carpeta ya creada.
    """
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    return DIAGRAMS_DIR / f"{model_name}_{kind}-diagram.png"


def draw_view(model, model_name, view, **overrides):
    """
    Dibuja una de las vistas de visualkeras de un modelo y la guarda.

    Parameters:
        model (keras.Model): Modelo entrenado, recién cargado de su '.keras'.
        model_name (str): Nombre del modelo, el mismo que en `diagram_path`.
        view (str): Vista, una de las claves de `VIEW_OPTIONS`.
        **overrides: Opciones de visualkeras que sustituyen a las de la vista,
            para el modelo que no salga bien con las comunes.

    Returns:
        Path: Ruta de la figura guardada.
    """
    path = diagram_path(model_name, view)
    render = getattr(visualkeras, f"{view}_view")
    extras = VIEW_EXTRAS.get(view, lambda model: {})(model)

    render(
        model
      , to_file=str(path)
      , color_map=color_map(model)
      , **{**VIEW_OPTIONS[view], **extras, **overrides}
    )

    return path


def draw_diagrams(model, model_name, views=DEFAULT_VIEWS, **overrides):
    """
    Dibuja las figuras de un modelo y las muestra donde se le llame.

    Parameters:
        model (keras.Model): Modelo entrenado, recién cargado de su '.keras'.
        model_name (str): Nombre del modelo, el mismo que en `diagram_path`.
        views (tuple): Vistas de visualkeras a dibujar, claves de
            `VIEW_OPTIONS`. Por omisión, las de `DEFAULT_VIEWS`.
        **overrides: Opciones por vista, cada una con el nombre de la vista por
            clave y un diccionario de opciones de visualkeras por valor.

    Returns:
        dict: Ruta de cada figura, con el nombre de su vista por clave.
    """
    paths = {
        view: draw_view(model, model_name, view, **overrides.get(view, {}))
        for view in views
    }

    for path in paths.values():
        print("Diagrama guardado en:", path)
        display(Image(str(path)))

    return paths
