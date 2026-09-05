"""
Configuración común del entrenamiento de los modelos.

Servicio compartido por todos los cuadernos de entrenamiento de `models`. Todos
entrenan con los mismos cuatro callbacks y con la misma configuración —lo que
cambia de un modelo a otro es la arquitectura y la normalización, no cuándo
parar ni cuándo bajar la tasa de aprendizaje—, así que armarlos vive aquí y no
repetido en la celda de entrenamiento de cada modelo.

Reunirlos además hace que esa celda dependa solo de lo que tiene delante: el
modelo ya compilado y sus datos normalizados. Antes la parada temprana y la
bajada de la tasa de aprendizaje se creaban una sola vez en una celda suelta muy
por encima, de modo que entrenar de nuevo un modelo obligaba a volver a pasar
por ella.
"""

from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


def training_callbacks(model_path, history_path):
    """
    Callbacks con los que se entrena cada modelo.

    Parameters:
        model_path (str | Path): Ruta del fichero `.keras` en el que guardar el
            modelo de la mejor época de validación.
        history_path (str | Path): Ruta del fichero CSV del historial por época.

    Returns:
        list: Los cuatro callbacks del entrenamiento: el checkpoint del modelo,
            la parada temprana, la bajada de la tasa de aprendizaje y el
            historial por época.
    """
    return [
        # Checkpoint del modelo: conserva los pesos de la mejor época
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        # Corta el entrenamiento cuando la validación pasa cinco épocas sin
        # mejorar, y deja en el modelo los pesos de la mejor de ellas
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-4,
            restore_best_weights=True
        ),

        # Baja la tasa de aprendizaje a la mitad cuando la validación se estanca
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),

        # Checkpoint del historial: añade una fila al final de cada época
        CSVLogger(history_path)
    ]
