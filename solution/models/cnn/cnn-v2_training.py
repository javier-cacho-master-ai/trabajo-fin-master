# %% [markdown]
# # Modelo convolucional 1D v2 (U-Net residual)
#
# Versión mejorada del modelo convolucional 1D. Mantiene el planteamiento de superresolución del modelo base (entrada interpolada sobre la malla de SDSS y conexión residual global) e incorpora tres mejoras:
#
# 1. **Pérdida ponderada por la varianza inversa de SDSS** (`y_ivar`): los píxeles ruidosos del espectro objetivo pesan menos y los píxeles inválidos (`ivar = 0`) se enmascaran, de modo que la red aprende el espectro subyacente y no la realización concreta del ruido.
# 2. **Canal de entrada adicional con el error de flujo de Gaia** (`X_err`): la red sabe qué zonas de su entrada son fiables.
# 3. **Mejora de datos con ruido realista**: en cada época se perturba el flujo de Gaia con ruido gaussiano de amplitud igual a su error observacional, lo que multiplica de forma efectiva el conjunto de entrenamiento.
#
# La arquitectura pasa de una pila de bloques residuales a una **U-Net 1D** con bloques residuales y atención de canal (*squeeze-and-excitation*). Con tres niveles de submuestreo, el campo receptivo cubre todo el espectro, por lo que la red puede corregir tanto el detalle de las líneas como el desajuste de continuo a gran escala entre Gaia y SDSS.
#

# %%
import sys

import numpy as np
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

DATA_DIR = SOLUTION_DIR / "data"
MODELS_DIR = SOLUTION_DIR / "models"
CNN_DIR = MODELS_DIR / "cnn"

# 'services', 'model_functions' e 'iSpec_functions' están en 'models', así que se
# importan desde su carpeta. iSpec tampoco se instala como dependencia, pero de
# montarlo se encarga 'iSpec_functions', que lo añade a 'sys.path' al importarse.
sys.path.insert(0, str(MODELS_DIR))

# Ficheros de checkpoint. Esta celda no carga datos, de modo que la recarga de
# una sesión anterior solo necesita ejecutar esta celda y la de recarga.
MODEL_PATH = CNN_DIR / "cnn-v2.keras"
HISTORY_PATH = CNN_DIR / "cnn-v2_training_history.csv"
SUMMARY_PATH = CNN_DIR / "cnn-v2_training_summary.json"
TEST_DATA_PATH = CNN_DIR / "cnn-v2_test_data.npz"

# Predicciones sobre el conjunto de test. No son un checkpoint del
# entrenamiento, sino el dato que consume la comparativa entre arquitecturas,
# así que van en 'data/predictions' y no en esta misma carpeta.
PREDICTIONS_PATH = DATA_DIR / "predictions" / "cnn-v2_predictions.npz"

print("Carpeta de trabajo:", SOLUTION_DIR)


# %%
# El nombre del fichero puede llevar la fecha de generación como prefijo
data_dir = DATA_DIR / "splits"
candidates = (
    sorted(data_dir.glob("*processed_data_no_duplicates.npz"))
    + sorted(data_dir.glob("*processed_data.npz"))
)
processed_path = candidates[0]
print("Usando:", processed_path)

data = np.load(processed_path)
print(data.files)


# %%
# Datos de entrada Gaia: flujo y error de flujo
X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

X_err_train = data["X_err_train"]
X_err_val = data["X_err_val"]
X_err_test = data["X_err_test"]

# Datos objetivo SDSS: flujo y varianza inversa
y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

y_ivar_train = data["y_ivar_train"]
y_ivar_val = data["y_ivar_val"]
y_ivar_test = data["y_ivar_test"]

# IDs de Gaia y SDSS
X_id_test = data["X_id_test"]
y_id_test = data["y_id_test"]

# Longitudes de onda
gaia_wavelength = data["gaia_wavelength"]
sdss_wavelength = data["sdss_wavelength"]

print("Gaia:", X_train.shape, "| SDSS:", y_train.shape)


# %% [markdown]
# Normalizamos cada par de espectros con la mediana del flujo de Gaia, igual que en el modelo base. El error de Gaia se normaliza con la misma escala para que siga siendo coherente con su flujo.
#
# Los pesos por píxel se derivan de la varianza inversa de SDSS: al dividir el flujo por la escala `s`, la varianza queda dividida por `s²`, luego la varianza inversa del flujo normalizado es `ivar · s²`. Dentro de cada espectro normalizamos los pesos para que su media sobre los píxeles válidos sea 1: así todos los objetos contribuyen por igual a la pérdida y los pesos solo redistribuyen la importancia entre píxeles. Un recorte superior evita que unos pocos píxeles de varianza muy baja dominen el entrenamiento; los píxeles con `ivar = 0` quedan enmascarados con peso 0.
#

# %%
# Escala por espectro: mediana del flujo de Gaia
X_scale_train = np.nanmedian(np.abs(X_train), axis=1, keepdims=True)
X_scale_val = np.nanmedian(np.abs(X_val), axis=1, keepdims=True)
X_scale_test = np.nanmedian(np.abs(X_test), axis=1, keepdims=True)

X_train_norm = X_train / X_scale_train
X_val_norm = X_val / X_scale_val
X_test_norm = X_test / X_scale_test

X_err_train_norm = X_err_train / X_scale_train
X_err_val_norm = X_err_val / X_scale_val
X_err_test_norm = X_err_test / X_scale_test

# El objetivo se normaliza con la misma escala que su entrada
y_train_norm = y_train / X_scale_train
y_val_norm = y_val / X_scale_val
y_test_norm = y_test / X_scale_test


def build_pixel_weights(y_ivar, scale, max_weight=10.0):
    # Varianza inversa del flujo normalizado
    weights = y_ivar * scale**2
    # Media 1 sobre los píxeles válidos de cada espectro
    valid = np.where(weights > 0, weights, np.nan)
    weights = weights / np.nanmean(valid, axis=1, keepdims=True)
    return np.clip(weights, 0.0, max_weight).astype(np.float32)


w_train = build_pixel_weights(y_ivar_train, X_scale_train)
w_val = build_pixel_weights(y_ivar_val, X_scale_val)
w_test = build_pixel_weights(y_ivar_test, X_scale_test)

print("Fracción de píxeles enmascarados (ivar = 0):", (w_train == 0).mean())


# %% [markdown]
# La interpolación lineal de Gaia sobre la malla de SDSS es una operación lineal fija, por lo que puede expresarse como una matriz de `(n_sdss, n_gaia)`. Esto permite aplicarla dentro del flujo de datos de TensorFlow, de modo que el ruido de la mejora de datos se añade sobre el espectro nativo de Gaia (donde es estadísticamente correcto) y la interpolación se hace después, en cada época.
#

# %%
from services.checkpoints import save_test_data


def build_interp_matrix(source_wavelength, target_wavelength):
    idx = np.searchsorted(source_wavelength, target_wavelength)
    idx = np.clip(idx, 1, len(source_wavelength) - 1)
    left = idx - 1
    right = idx
    t = (target_wavelength - source_wavelength[left]) / (
        source_wavelength[right] - source_wavelength[left]
    )
    # Fuera del rango de Gaia se mantiene el valor extremo, igual que np.interp
    t = np.clip(t, 0.0, 1.0)
    matrix = np.zeros(
        (len(target_wavelength), len(source_wavelength)), dtype=np.float32
    )
    rows = np.arange(len(target_wavelength))
    matrix[rows, left] = 1.0 - t
    matrix[rows, right] = t
    return matrix


interp_matrix = build_interp_matrix(gaia_wavelength, sdss_wavelength)

# Comprobación: la matriz reproduce np.interp
reference = np.interp(sdss_wavelength, gaia_wavelength, X_train_norm[0])
assert np.allclose(X_train_norm[0] @ interp_matrix.T, reference, atol=1e-5)


def interpolate_pair(flux_norm, err_norm):
    flux_interp = (flux_norm @ interp_matrix.T).astype(np.float32)
    err_interp = (err_norm @ interp_matrix.T).astype(np.float32)
    return flux_interp, err_interp


X_val_flux_interp, X_val_err_interp = interpolate_pair(X_val_norm, X_err_val_norm)
X_test_flux_interp, X_test_err_interp = interpolate_pair(X_test_norm, X_err_test_norm)

print(X_val_flux_interp.shape, "->", y_val_norm.shape)

# Guardamos las variables del conjunto de test que consumen las celdas
# posteriores al entrenamiento, para poder ejecutarlas sin repetir estas celdas
save_test_data(
    TEST_DATA_PATH,
    X_test=X_test,
    y_test=y_test,
    y_ivar_test=y_ivar_test,
    X_id_test=X_id_test,
    y_id_test=y_id_test,
    gaia_wavelength=gaia_wavelength,
    sdss_wavelength=sdss_wavelength,
    X_scale_test=X_scale_test,
    X_test_flux_interp=X_test_flux_interp,
    X_test_err_interp=X_test_err_interp
)


# %% [markdown]
# Flujos de datos de entrenamiento y evaluación. El de entrenamiento añade en cada época ruido gaussiano al flujo de Gaia con la amplitud de su error observacional y después interpola; los de validación y test solo interpolan. Cada elemento produce las dos entradas del modelo (flujo interpolado y error interpolado), el objetivo y los pesos por píxel.
#

# %%
import tensorflow as tf

batch_size = 64
interp_matrix_tf = tf.constant(interp_matrix)


def project_to_sdss(flux, err, target, weight):
    flux_interp = tf.linalg.matvec(interp_matrix_tf, flux)
    err_interp = tf.linalg.matvec(interp_matrix_tf, err)
    inputs = (flux_interp[:, None], err_interp[:, None])
    return inputs, target[:, None], weight


def augment_with_noise(flux, err, target, weight):
    noise = tf.random.normal(tf.shape(flux)) * err
    return flux + noise, err, target, weight


def as_tensor_slices(flux, err, target, weight):
    return tf.data.Dataset.from_tensor_slices((
        flux.astype(np.float32),
        err.astype(np.float32),
        target.astype(np.float32),
        weight,
    ))


train_ds = (
    as_tensor_slices(X_train_norm, X_err_train_norm, y_train_norm, w_train)
    .shuffle(4096)
    .map(augment_with_noise, num_parallel_calls=tf.data.AUTOTUNE)
    .map(project_to_sdss, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    as_tensor_slices(X_val_norm, X_err_val_norm, y_val_norm, w_val)
    .map(project_to_sdss, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    as_tensor_slices(X_test_norm, X_err_test_norm, y_test_norm, w_test)
    .map(project_to_sdss, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)


# %% [markdown]
# Arquitectura U-Net 1D:
#
# - **Codificador**: convolución inicial (64 filtros, núcleo 9) seguida de tres niveles con un bloque residual y submuestreo por convolución con paso 2 (64 → 96 → 128 → 160 filtros).
# - **Cuello de botella**: dos bloques residuales de 160 filtros que ven el espectro completo a resolución 1/8.
# - **Decodificador**: tres niveles de sobremuestreo con concatenación de la conexión del codificador correspondiente y un bloque residual.
# - Cada bloque residual incluye atención de canal (*squeeze-and-excitation*).
# - La convolución final de un filtro genera la corrección y la **conexión residual global** le suma el flujo interpolado, igual que en el modelo base.
#
# Como 2666 no es múltiplo de 8, se añaden 3 puntos de relleno por cada lado (2672 = 8 · 334) y la corrección se recorta a la longitud original antes de la suma global.
#
# La pérdida es Huber ponderada: los pesos por píxel entran como `sample_weight` temporal, de forma que las métricas `weighted_mae` y `weighted_mse` reflejan el error que optimiza la red, y `mae` y `mse` el error sin ponderar. Se usan las mismas métricas que en el resto de modelos de la memoria (denso, RNN y convolucional v1), de modo que los resultados son comparables entre sí.
#

# %%
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

n_sdss = len(sdss_wavelength)
padding = (3, 3)


def se_block(x, filters, ratio=8):
    excite = layers.GlobalAveragePooling1D()(x)
    excite = layers.Dense(filters // ratio, activation="elu")(excite)
    excite = layers.Dense(filters, activation="sigmoid")(excite)
    excite = layers.Reshape((1, filters))(excite)
    return layers.Multiply()([x, excite])


def residual_se_block(x, filters, kernel_size=7):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="elu")(x)
    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = se_block(x, filters)
    x = layers.Add()([x, shortcut])
    return layers.Activation("elu")(x)


def downsample(x, filters):
    return layers.Conv1D(filters, 5, strides=2, padding="same", activation="elu")(x)


def upsample_and_merge(x, skip, filters):
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters, 5, padding="same", activation="elu")(x)
    x = layers.Concatenate()([x, skip])
    return layers.Conv1D(filters, 5, padding="same", activation="elu")(x)


flux_input = layers.Input(shape=(n_sdss, 1), name="interp_flux")
err_input = layers.Input(shape=(n_sdss, 1), name="interp_err")

x = layers.Concatenate()([flux_input, err_input])
x = layers.ZeroPadding1D(padding)(x)
x = layers.Conv1D(64, 9, padding="same", activation="elu")(x)

enc1 = residual_se_block(x, 64)      # 2672 puntos
x = downsample(enc1, 96)             # 1336
enc2 = residual_se_block(x, 96)
x = downsample(enc2, 128)            # 668
enc3 = residual_se_block(x, 128)
x = downsample(enc3, 160)            # 334

x = residual_se_block(x, 160)
x = residual_se_block(x, 160)

x = upsample_and_merge(x, enc3, 128)  # 668
x = residual_se_block(x, 128)
x = upsample_and_merge(x, enc2, 96)   # 1336
x = residual_se_block(x, 96)
x = upsample_and_merge(x, enc1, 64)   # 2672
x = residual_se_block(x, 64)

correction = layers.Conv1D(1, 5, padding="same")(x)
correction = layers.Cropping1D(padding)(correction)

# Conexión residual global: salida = flujo interpolado + corrección
outputs = layers.Add()([flux_input, correction])

model_cnn_v2 = models.Model(
    [flux_input, err_input], outputs, name="cnn_unet_1d"
)

model_cnn_v2.compile(
    optimizer=Adam(learning_rate=3e-4, clipnorm=1.0),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=["mae", "mse"],
    weighted_metrics=["mae", "mse"],
)

model_cnn_v2.summary()


# %% [markdown]
# El proceso deja cuatro ficheros de checkpoint en esta misma carpeta, uno por cada forma de dato, con los que las celdas de evaluación y análisis pueden ejecutarse en una sesión nueva sin repetir el entrenamiento ni volver a cargar el `.npz` de datos completo:
#
# - `cnn-v2.keras`: el modelo con la mejor pérdida de validación, guardado cada vez que mejora.
# - `cnn-v2_training_history.csv`: una fila por época con todas las métricas, escrita al final de cada época. Es una **tabla**, así que la escribe directamente el callback `CSVLogger` de Keras, sin código propio, y queda legible y comparable entre versiones del modelo. Un CSV además admite los `NaN` de una época divergente, que en JSON no serían válidos.
# - `cnn-v2_training_summary.json`: el resumen del entrenamiento (mejor época, tamaño de lote y métricas de test). Son **datos sueltos y heterogéneos** que no caben en una tabla ni en un contenedor de arrays, y en JSON siguen siendo legibles y versionables en Git.
# - `cnn-v2_test_data.npz`: las variables del conjunto de test que consumen las celdas posteriores (`X_test`, `y_test`, la varianza inversa de SDSS, los identificadores, las longitudes de onda, la escala y los espectros interpolados). Son **arrays** de 177 MB en total, para los que `.npz` es el único formato razonable de los tres: conserva forma y `dtype` sin código de conversión y se escribe y lee en menos de un segundo, mientras que en JSON o CSV los mismos datos ocuparían unos 800 MB de texto. Sobre todo, conserva los identificadores de Gaia como `int64`: son de hasta 19 dígitos y más de la mitad no se representan de forma exacta en el `float64` al que los llevaría un CSV o un JSON leído como decimal.
#
# La lógica de guardado y recarga vive en `models/services/checkpoints.py`, compartida por todos los cuadernos de entrenamiento. De las predicciones se ocupa `models/services/predictions.py`, que las guarda aparte, en `data/predictions/cnn-v2_predictions.npz`: no son un checkpoint del entrenamiento sino el dato que consume la comparativa entre arquitecturas.
#

# %%
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from services.checkpoints import save_training_summary

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=1e-4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

# Checkpoint del modelo: conserva los pesos de la mejor época de validación
save_best_epoch_model = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# Checkpoint del historial: añade una fila al final de cada época
save_training_history = CSVLogger(HISTORY_PATH)

history = model_cnn_v2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[save_best_epoch_model, early_stop, reduce_lr, save_training_history],
    verbose=1
)

# `EarlyStopping` restaura los mejores pesos: los fijamos en el checkpoint final
model_cnn_v2.save(MODEL_PATH)

training_history = history.history
save_training_summary(
    SUMMARY_PATH,
    training_history,
    model_path=MODEL_PATH,
    history_path=HISTORY_PATH,
    batch_size=batch_size
)


# %% [markdown]
# Recarga desde el checkpoint. Esta celda recupera de los cuatro ficheros todas las variables que necesitan las celdas siguientes: el modelo, el historial, el resumen con las métricas de test y las variables del conjunto de test. Ejecutando la primera celda del cuaderno —la de rutas, que no carga datos— y después esta, el resto del análisis funciona sin reentrenar y sin volver a cargar el `.npz` de datos completo.
#
# La única celda que queda fuera es la de evaluación: `test_ds` es un flujo de datos de TensorFlow, que no se puede serializar en ninguno de estos formatos. No hace falta, porque su resultado, `test_metrics`, es justo lo que guarda el resumen.
#

# %%
from services.checkpoints import load_checkpoint, load_test_data

model_cnn_v2, training_history, training_summary = load_checkpoint(
    MODEL_PATH, HISTORY_PATH, SUMMARY_PATH
)

test_metrics = training_summary["test_metrics"]

# Variables del conjunto de test que consumen las celdas siguientes
test_data = load_test_data(TEST_DATA_PATH)

X_test = test_data["X_test"]
y_test = test_data["y_test"]
y_ivar_test = test_data["y_ivar_test"]
X_id_test = test_data["X_id_test"]
y_id_test = test_data["y_id_test"]
gaia_wavelength = test_data["gaia_wavelength"]
sdss_wavelength = test_data["sdss_wavelength"]
X_scale_test = test_data["X_scale_test"]
X_test_flux_interp = test_data["X_test_flux_interp"]
X_test_err_interp = test_data["X_test_err_interp"]

print("Checkpoint de", training_summary["updated"])
print(
    "Épocas completadas:", training_summary["epochs_completed"],
    "| mejor época:", training_summary["best_epoch"]
)
print("Métricas de test:", test_metrics)


# %%
# Importamos funciones a usar para analizar resultados
from model_functions import (
    plot_training_metrics,
    plot_prediction_example,
    plot_worst_best_predictions,
    calculate_all_chi2
)

plot_training_metrics(training_history)


# %%
test_metrics = model_cnn_v2.evaluate(
    test_ds,
    verbose=1,
    return_dict=True
)

# Las métricas de test se añaden al resumen del entrenamiento
save_training_summary(
    SUMMARY_PATH,
    training_history,
    test_metrics,
    model_path=MODEL_PATH,
    history_path=HISTORY_PATH,
    batch_size=batch_size
)

print(test_metrics)


# %%
from services.predictions import predict_test_set, save_predictions

# Predicciones sobre el conjunto de test. La celda parte del `.keras` del modelo
# y del `.npz` de test, así que se ejecuta suelta tras la de rutas: no
# reentrena, no vuelve a cargar el `.npz` de datos completo y no necesita las
# variables que dejen en memoria las demás celdas del cuaderno.
y_pred = predict_test_set(
    MODEL_PATH,
    TEST_DATA_PATH,
    build_input=lambda test: [
        test["X_test_flux_interp"][..., None],
        test["X_test_err_interp"][..., None]
    ],
    # La normalización se deshace solo con la escala de la entrada de Gaia
    denormalize=lambda y_norm, test: y_norm[..., 0] * test["X_scale_test"]
)

# Las guardamos para la comparativa entre arquitecturas, en su propio cuaderno
save_predictions(PREDICTIONS_PATH, y_pred=y_pred)


# %%
# Mismos objetos usados en las comparativas de la memoria
for i in [123, 191]:
    plot_prediction_example(
        i,
        sdss_wavelength,
        gaia_wavelength,
        y_test,
        y_pred,
        y_id_test,
        X_test,
        X_id_test
    )


# %%
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Comparación invariante a escala: cada espectro se normaliza con su propia
# mediana, de modo que el ranking refleja el error de forma y no el desajuste
# de calibración absoluta entre Gaia y SDSS
y_test_shape = y_test / np.nanmedian(np.abs(y_test), axis=1, keepdims=True)
y_pred_shape = y_pred / np.nanmedian(np.abs(y_pred), axis=1, keepdims=True)
X_test_shape = X_test / np.nanmedian(np.abs(X_test), axis=1, keepdims=True)

plot_worst_best_predictions(
    y_test_shape,
    y_pred_shape,
    X_test_shape,
    X_id_test,
    sdss_wavelength,
    gaia_wavelength,
    10
)


# %% [markdown]
# Chi cuadrado normalizado frente al espectro real de SDSS, la métrica con la que se comparan entre sí todas las arquitecturas del trabajo: la diferencia de flujo se pondera con la varianza inversa de SDSS y se divide entre el número de puntos válidos (aquellos cuya varianza inversa es mayor que cero). Se calcula sobre el flujo sin normalizar, de modo que los valores son comparables con los de los modelos denso y recurrente.
#

# %%
chi2_values = calculate_all_chi2(y_test, y_pred, y_ivar_test)

print(f"Mediana: {np.median(chi2_values): .2f}")


# %%
# El modelo ya está guardado por el checkpoint; lo reescribimos por si esta
# sesión ha continuado el entrenamiento a partir de un checkpoint anterior
model_cnn_v2.save(MODEL_PATH)


# %% [markdown]
# Validación física con iSpec: analizamos una muestra de espectros de test comparando los parámetros estelares (Teff, log g, [M/H]) derivados del espectro SDSS real frente a los derivados del espectro predicho por la red.
#
# El proceso se reparte en tres celdas independientes —importaciones, recursos y análisis— para que repetir el análisis, que es con diferencia lo más lento, no vuelva a cargar los recursos.
#
# Las predicciones son las de la celda de predicción, que carga el modelo guardado (`cnn-v2.keras`) y no requiere haber entrenado en esta sesión.

# %%
# Recargamos el módulo si ha habido modificaciones
# %load_ext autoreload
# %autoreload 2

# Funciones de iSpec (el módulo está en 'models', ya añadido a sys.path)
from iSpec_functions import (
    load_ispec_resources,
    load_gaia_table,
    analyze_sample_real_vs_pred,
    analyze_ispec_errors
)

# %%
# Recursos de iSpec y tabla de Gaia. Las dos cargas están cacheadas en
# 'iSpec_functions', de modo que repetir esta celda no vuelve a leer de disco
resources = load_ispec_resources()

gaia_data_df = load_gaia_table()

# %%
# Analizamos con iSpec una muestra de espectros reales frente a predichos
results_df_cnn_v2 = analyze_sample_real_vs_pred(
    y_test,
    y_pred,
    X_id_test,
    gaia_data_df,
    sdss_wavelength,
    resources,
    sample_size=250
)

error_df_cnn_v2 = analyze_ispec_errors(results_df_cnn_v2)
