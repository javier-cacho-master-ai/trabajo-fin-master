import matplotlib.pyplot as plt
import numpy as np

def plot_training_metrics(model_history, metrics=None):
    epochs = range(1, len(model_history["loss"]) + 1)

    # Por defecto, las métricas registradas por el modelo: las que tienen su
    # pareja de validación, de modo que sirve para cualquier compilación
    if metrics is None:
        metrics = [
            metric for metric in model_history
            if not metric.startswith("val_") and f"val_{metric}" in model_history
        ]

    for metric in metrics:
        plt.figure(figsize=(9, 4))
        plt.plot(epochs, model_history[metric], label=f"Métrica {metric} de entrenamiento")
        plt.plot(epochs, model_history[f"val_{metric}"], label=f"Métrica {metric} de validación")

        plt.xlabel("Época")
        plt.ylabel(metric.upper())
        plt.title(f"Evolución de {metric}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


def calculate_chi2(y_real, y_pred, y_ivar):

    valid = (
        np.isfinite(y_ivar)
        & (y_ivar > 0)
    )
    chi2 = np.sum((pow((y_real[valid] - y_pred[valid]), 2))* y_ivar[valid])

    return chi2 / len(y_real[valid])


def calculate_all_chi2(y_real, y_pred, y_ivar, plot=True):
    chi2_values = []

    for i in range(len(y_real)):
        chi2_values.append(calculate_chi2(y_real[i], y_pred[i], y_ivar[i]))

    chi2_values = np.asarray(chi2_values)

    if plot:
        plt.figure(figsize=(8, 5))

        plt.hist(
            chi2_values,
            bins=40,
            alpha=0.7
        )

        median_chi2 = np.median(chi2_values)
        maximum_chi2 = np.max(chi2_values)
        minimum_chi2 = np.min(chi2_values)

        plt.axvline(
            median_chi2,
            linestyle="--",
            linewidth=2,
            label=f"Mediana = {median_chi2:.2f}"
        )

        plt.xlabel("chi cuadrado")
        plt.ylabel("Número de espectros")
        plt.title("Distribución del chi cuadrado entre espectros reales y predichos")

        plt.legend()
        plt.show()

        print(f"Mínimo: {minimum_chi2: .2f}")
        print(f"Máximo: {maximum_chi2: .2f}")

    return chi2_values


def plot_hr_chi2_hexbin(
    bp_rp,
    g_mag,
    parallax,
    chi2_red,
    gridsize=60,
    mincnt=1,
    figsize=(8, 8)
):

    bp_rp = np.asarray(bp_rp)
    g_mag = np.asarray(g_mag)
    parallax = np.asarray(parallax)
    chi2_red = np.asarray(chi2_red)

    # Valores válidos
    valid = (
        np.isfinite(bp_rp)
        & np.isfinite(g_mag)
        & np.isfinite(parallax)
        & np.isfinite(chi2_red)
        & (parallax > 0)
        & (chi2_red > 0)
    )

    # Corrección de la distancia en la magnitud absoluta G
    abs_g_mag = (g_mag[valid]+ 5 * np.log10(parallax[valid]) - 10)

    chi2_valid = chi2_red[valid]

    plt.figure(figsize=figsize)

    hb = plt.hexbin(
        bp_rp[valid],
        abs_g_mag,
        C=chi2_valid,
        gridsize=gridsize,
        reduce_C_function=np.median,
        mincnt=mincnt
    )

    cbar = plt.colorbar(hb)

    cbar.set_label(
        r"Mediana de $\chi^2_{\mathrm{red}}$"
    )

    plt.gca().invert_yaxis()

    plt.xlabel("BP - RP")
    plt.ylabel(r"$M_G$")

    plt.title(
        r"Distribución del $\chi^2$ en el diagrama HR"
    )

    plt.tight_layout()
    plt.show()