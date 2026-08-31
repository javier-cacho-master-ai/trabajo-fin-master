# %% [markdown]
# # Modelo convolucional 1D
# 
# Entrenamiento del modelo convolucional descrito en la memoria (apartado *Modelo convolucional 1D*).
# 
# Planteamiento de superresolución:
# 
# 1. El espectro de Gaia se interpola linealmente sobre la malla de longitudes de onda de SDSS, de modo que entrada y salida quedan alineadas punto a punto.
# 2. Una red convolucional residual aprende la **corrección** que hay que aplicar sobre el espectro interpolado (conexión residual global), en lugar de generar el espectro completo desde cero.
# 3. Cada par de espectros se normaliza con la mediana del flujo de Gaia, por lo que la normalización puede deshacerse usando solo la entrada (aplicable a observaciones nuevas sin conocer el espectro de SDSS).

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
ISPEC_DIR = SOLUTION_DIR / "lib" / "iSpec"

# 'services' está en la carpeta de este cuaderno, 'model_functions' e
# 'iSpec_functions' en 'models' e iSpec no se instala como dependencia: todos se
# importan desde su carpeta
sys.path[:0] = [str(CNN_DIR), str(MODELS_DIR), str(ISPEC_DIR)]

# Ficheros de checkpoint. Esta celda no carga datos, de modo que la recarga de
# una sesión anterior solo necesita ejecutar esta celda y la de recarga.
MODEL_PATH = CNN_DIR / "cnn_v1.keras"
HISTORY_PATH = CNN_DIR / "cnn_v1_training_history.csv"
SUMMARY_PATH = CNN_DIR / "cnn_v1_training_summary.json"
TEST_DATA_PATH = CNN_DIR / "cnn_v1_test_data.npz"

# Tamaño de lote del entrenamiento, guardado también en el resumen
batch_size = 64

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
# Datos de entrada Gaia
X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

# Datos objetivo SDSS
y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

# IDs de Gaia y SDSS
X_id_test = data["X_id_test"]
y_id_test = data["y_id_test"]

# Varianza inversa de SDSS, con la que se pondera el chi cuadrado
y_ivar_test = data["y_ivar_test"]

# Longitudes de onda
gaia_wavelength = data["gaia_wavelength"]
sdss_wavelength = data["sdss_wavelength"]

print("Gaia:", X_train.shape, "| SDSS:", y_train.shape)

# %% [markdown]
# Normalizamos cada par de espectros con la mediana del flujo de Gaia. Como ambos flujos están en las mismas unidades y los pares con escalas incoherentes ya se descartaron en el preprocesado, el espectro de SDSS normalizado queda en una escala cercana a 1.

# %%
# Escala por espectro: mediana del flujo de Gaia
X_scale_train = np.nanmedian(np.abs(X_train), axis=1, keepdims=True)
X_scale_val = np.nanmedian(np.abs(X_val), axis=1, keepdims=True)
X_scale_test = np.nanmedian(np.abs(X_test), axis=1, keepdims=True)

X_train_norm = X_train / X_scale_train
X_val_norm = X_val / X_scale_val
X_test_norm = X_test / X_scale_test

# El objetivo se normaliza con la misma escala que su entrada
y_train_norm = y_train / X_scale_train
y_val_norm = y_val / X_scale_val
y_test_norm = y_test / X_scale_test

# %% [markdown]
# Interpolamos el espectro de Gaia sobre la malla de longitudes de onda de SDSS. Esta versión interpolada es la entrada de la red y también la base sobre la que se aplica la corrección residual.

# %%
from services.persistence import save_test_data


def interpolate_to_sdss_grid(spectra):
    return np.stack([
        np.interp(sdss_wavelength, gaia_wavelength, spectrum)
        for spectrum in spectra
    ])

X_train_interp = interpolate_to_sdss_grid(X_train_norm)
X_val_interp = interpolate_to_sdss_grid(X_val_norm)
X_test_interp = interpolate_to_sdss_grid(X_test_norm)

print(X_train_interp.shape, "->", y_train_norm.shape)

# Guardamos las variables del conjunto de test que consumen las celdas
# posteriores al entrenamiento, para poder ejecutarlas sin repetir estas celdas
save_test_data(
    TEST_DATA_PATH,
    X_test=X_test,
    y_test=y_test,
    y_test_norm=y_test_norm,
    y_ivar_test=y_ivar_test,
    X_id_test=X_id_test,
    gaia_wavelength=gaia_wavelength,
    sdss_wavelength=sdss_wavelength,
    X_scale_test=X_scale_test,
    X_test_interp=X_test_interp
)

# %% [markdown]
# Arquitectura: capa convolucional inicial (64 filtros, núcleo 9), cuatro bloques residuales (dos convoluciones de 64 filtros con núcleo 7 y activación ELU, más un atajo) y una convolución final de un filtro que genera la corrección. La conexión residual global suma la entrada interpolada a dicha corrección.

# %%
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

n_sdss = y_train.shape[1]


def residual_block(x, filters=64, kernel_size=7):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="elu")(x)
    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.Add()([x, shortcut])
    return layers.Activation("elu")(x)


inputs = layers.Input(shape=(n_sdss, 1))

x = layers.Conv1D(64, 9, padding="same", activation="elu")(inputs)

for _ in range(4):
    x = residual_block(x)

correction = layers.Conv1D(1, 5, padding="same")(x)

# Conexión residual global: salida = entrada interpolada + corrección
outputs = layers.Add()([inputs, correction])
outputs = layers.Reshape((n_sdss,))(outputs)

model_cnn = models.Model(inputs, outputs, name="cnn_residual_1d")

model_cnn.compile(
    optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=["mae", "mse"]
)

model_cnn.summary()

# %% [markdown]
# El proceso deja cuatro ficheros de checkpoint en esta misma carpeta, uno por cada forma de dato, con los que las celdas de evaluación y análisis pueden ejecutarse en una sesión nueva sin repetir el entrenamiento ni volver a cargar el `.npz` de datos completo:
#
# - `cnn_v1.keras`: el modelo con la mejor pérdida de validación, guardado cada vez que mejora.
# - `cnn_v1_training_history.csv`: una fila por época con todas las métricas, escrita al final de cada época. Es una **tabla**, así que la escribe directamente el callback `CSVLogger` de Keras, sin código propio, y queda legible y comparable entre versiones del modelo. Un CSV además admite los `NaN` de una época divergente, que en JSON no serían válidos.
# - `cnn_v1_training_summary.json`: el resumen del entrenamiento (mejor época, tamaño de lote y métricas de test). Son **datos sueltos y heterogéneos** que no caben en una tabla ni en un contenedor de arrays, y en JSON siguen siendo legibles y versionables en Git.
# - `cnn_v1_test_data.npz`: las variables del conjunto de test que consumen las celdas posteriores. Son **arrays** de más de cien megabytes en total, para los que `.npz` es el único formato razonable de los tres: conserva forma y `dtype` sin código de conversión y se escribe y lee en menos de un segundo, mientras que en JSON o CSV los mismos datos ocuparían varias veces más en texto. Sobre todo, conserva los identificadores de Gaia como `int64`: son de hasta 19 dígitos y más de la mitad no se representan de forma exacta en el `float64` al que los llevaría un CSV o un JSON leído como decimal.
#
# La lógica de guardado y recarga vive en `services/persistence.py`.
#

# %%
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from services.persistence import save_training_summary

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
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

history = model_cnn.fit(
    X_train_interp[..., None],
    y_train_norm,
    validation_data=(X_val_interp[..., None], y_val_norm),
    epochs=50,
    batch_size=batch_size,
    callbacks=[save_best_epoch_model, early_stop, reduce_lr, save_training_history],
    verbose=1
)

# `EarlyStopping` restaura los mejores pesos: los fijamos en el checkpoint final
model_cnn.save(MODEL_PATH)

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

# %%
from services.persistence import load_checkpoint, load_test_data

model_cnn, training_history, training_summary = load_checkpoint(
    MODEL_PATH, HISTORY_PATH, SUMMARY_PATH
)

test_metrics = training_summary["test_metrics"]

# Variables del conjunto de test que consumen las celdas siguientes
test_data = load_test_data(TEST_DATA_PATH)

X_test = test_data["X_test"]
y_test = test_data["y_test"]
y_test_norm = test_data["y_test_norm"]
y_ivar_test = test_data["y_ivar_test"]
X_id_test = test_data["X_id_test"]
gaia_wavelength = test_data["gaia_wavelength"]
sdss_wavelength = test_data["sdss_wavelength"]
X_scale_test = test_data["X_scale_test"]
X_test_interp = test_data["X_test_interp"]

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
    plot_worst_best_predictions,
    calculate_all_chi2
)

plot_training_metrics(training_history)


# %%
from services.persistence import save_training_summary

test_metrics = model_cnn.evaluate(
    X_test_interp[..., None],
    y_test_norm,
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
y_pred_norm = model_cnn.predict(X_test_interp[..., None])

# Deshacemos la normalización solo con la escala de la entrada de Gaia
y_pred = y_pred_norm * X_scale_test

# %%
# Mismos objetos usados en las comparativas de la memoria

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

# %%
chi2_values = calculate_all_chi2(y_test, y_pred, y_ivar_test)

print(f"Mediana: {np.median(chi2_values): .2f}")

# %%
# El modelo ya está guardado por el checkpoint; lo reescribimos por si esta
# sesión ha continuado el entrenamiento a partir de un checkpoint anterior
model_cnn.save(MODEL_PATH)

# %% [markdown]
# Validación física con iSpec: analizamos una muestra de espectros de test comparando los parámetros estelares (Teff, log g, [M/H]) derivados del espectro SDSS real frente a los derivados del espectro predicho por la red.
# 
# Esta celda carga el modelo guardado (`cnn_v1.keras`) y genera sus propias predicciones, por lo que no requiere haber entrenado en esta sesión: basta con ejecutar antes las celdas de preparación de datos (carga, normalización e interpolación).

# %%
# Funciones de iSpec (el módulo está en 'models', ya añadido a sys.path)
from iSpec_functions import (
    load_ispec_resources,
    analyze_sample_real_vs_pred,
    analyze_ispec_errors
)

import tensorflow as tf
from astropy.table import Table

# Cargamos el modelo entrenado y generamos las predicciones sobre test
model_cnn = tf.keras.models.load_model(MODEL_PATH)

y_pred_norm = model_cnn.predict(X_test_interp[..., None])

# Deshacemos la normalización solo con la escala de la entrada de Gaia
y_pred = y_pred_norm * X_scale_test

# Cargamos los recursos de iSpec
resources = load_ispec_resources()

# Cargamos la tabla de Gaia guardada en local
gaia_data_path = DATA_DIR / "gaia_data.ecsv"

gaia_data_table = Table.read(
    gaia_data_path,
    format="ascii.ecsv"
)

# Pasamos a dataframe
gaia_data_df = gaia_data_table.to_pandas()

# Analizamos con iSpec una muestra de espectros reales frente a predichos
results_df_cnn = analyze_sample_real_vs_pred(
    y_test,
    y_pred,
    X_id_test,
    gaia_data_df,
    sdss_wavelength,
    resources,
    sample_size=250
)

error_df_cnn = analyze_ispec_errors(results_df_cnn)


