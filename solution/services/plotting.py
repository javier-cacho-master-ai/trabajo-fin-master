import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from specutils import Spectrum


def plot_physical_spectrum(spectrum: Spectrum, title: str = "Espectro"):
    sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.linestyle": "--"})
    plt.figure(figsize=(12, 5))

    x = spectrum.spectral_axis.value
    y = spectrum.flux.value

    # Create line segments for the gradient
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap="turbo", linewidth=1)
    lc.set_array(x)
    plt.gca().add_collection(lc)

    plt.xlabel(f"Longitud de onda ({spectrum.spectral_axis.unit})")
    plt.ylabel(f"Flujo ({spectrum.flux.unit})")
    plt.title(title)

    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max() * 1.05)  # Add slight padding to the top
    plt.tight_layout()
    plt.show()
