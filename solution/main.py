# %% [markdown]
#  # Uso de redes generativas para la mejora de la señal astronómica
#
# El presente "notebook" documenta la extracción y emparejamiento de los datos de Gaia y SDSS.
#
# Puesto que SDSS tiene menos datos con espectrografía que los que proporciona el crossmatch de Gaia, para mostrar los gráficos usaremos un ejemplo aleatorio de SDSS para mostrar la misma espectrometría para ese objeto en Gaia.cÑw
#
# El procedimiento será el siguiente:
# 1. De Gaia, obtener los mejores crossmatches de Gaia y SDSS
# 2. De SDSS, obtener la referencia de la tabla de SDSS con la espectrometría (SpecObj)
# 3. De SDSS, seleccionar un ejemplo aleatorio
# 4. De SDSS, obtener la espectrometría para el ejemplo obtenido en 3.
# 5. De Gaia, obtener la espectrometría para el objeto obtenido en 3.
# 6. De Gaia y SDSS, mostrar los gráficos para el objecto obtenido en 3

# %% [markdown]
#  ## Extracción de los datos

# %% [markdown]
#  ### Obtención de una muestra de objetos

# %% [markdown]
#   #### Crossmatches Gaia SDSS
# 
#   Se seleccionan aleatoriamente los mejores crossmatches entre Gaia y SDSS.

# %%
from pathlib import Path
from astroquery.gaia import Gaia
from astropy.table import Table

gaia_query_path = Path("services", "queries", "gaia_sdss_random_subset.adql")
gaia_query_template = gaia_query_path.read_text(encoding="utf-8")
gaia_query = gaia_query_template.format(start_index=10000, end_index=20000)

print(f"Executing Step 1: {gaia_query_path.name}...")
gaia_query_job = Gaia.launch_job_async(gaia_query)
gaia_job_results = gaia_query_job.get_results() or Table()

display(gaia_job_results)


# %% [markdown]
#   #### SDSS - Datos de referencia de objetos

# %%
from astroquery.sdss import SDSS
from astropy.io import ascii
from astropy.table import Table
import services.querying as sq

gaia_crossmatch_ids = gaia_job_results["original_ext_source_id"]
query_filter_ids = ",".join(gaia_crossmatch_ids.astype(str))

sdss_query_path = Path("services", "queries", "sdss.adql")
sdss_query_template = sdss_query_path.read_text(encoding="utf-8")
sdss_query_template_sanitized = sq.sanitize_adql(sdss_query_template)
sdss_query = sdss_query_template_sanitized.format(filter_ids=query_filter_ids)

sdss_query_result = SDSS.query_sql(sdss_query, data_release=13)

display(sdss_query_result)



# %% [markdown]
#   ### Obtención de los datos de espectrometría

# %% [markdown]
#   #### SDSS - Selección ejemplo aleatorio

# %%
import random as rnd

sdss_sample_object = rnd.choice(sdss_query_result)

display(sdss_sample_object)


# %% [markdown]
#  #### SDSS - Espectrometría para el ejemplo

# %%

sdss_spectra_response = (
    SDSS.get_spectra(
        plate=sdss_sample_object["plate"],
        mjd=sdss_sample_object["mjd"],
        fiberID=sdss_sample_object["fiberID"],
        data_release=13,
    )
    or []
)

# help(sdss_spectra_response)
# dir(sdss_spectra_response)
# print(sdss_spectra_response)

sdss_fits_file = sdss_spectra_response[0]
sdss_spectra_hdu = sdss_fits_file[
    1
]  # La primera tabla después de la cabecera contiene los datos
sdss_spectra_data = sdss_spectra_hdu.data

display(sdss_spectra_data)


# %% [markdown]
#  #### Gaia - Espectrometría para el ejemplo
# 
#  Obtenemos la referencia en Gaia para el objecto de muestra de SDSS

# %%
gaia_sample_object = gaia_job_results[
    gaia_job_results["original_ext_source_id"] == sdss_sample_object["bestObjID"]
]

display(gaia_sample_object)

gaia_sample_object_source_id = gaia_sample_object["source_id"].item()

display(type(gaia_sample_object_source_id))
display(gaia_sample_object_source_id)


# %% [markdown]
#  Y obtenemos la espctrometría para esa referencia.

# %%
import gaiaxpy
import astropy.units as u
from specutils import Spectrum

# 1. Calibrate functionally using gaiaxpy
source_ids = [gaia_sample_object_source_id]
calibrated_spectra, sampling_grid = gaiaxpy.calibrate(source_ids)

# 2. Extract arrays and attach Astropy units
flux_array = calibrated_spectra["flux"].iloc[0]
wavelength_qty = sampling_grid * u.nm
flux_qty = flux_array * (u.W / (u.m**2 * u.nm))

# 3. Instantiate the standard specutils Spectrum1D object
gaia_object_spectrum = Spectrum(flux=flux_qty, spectral_axis=wavelength_qty)

display(gaia_object_spectrum)


# %% [markdown]
#  ### Gráficos comparativos

# %% [markdown]
#  #### SDSS - Flujo y Longitud de Onda para una objeto

# %%
from services.sdss import parse_to_si
from services.plotting import plot_physical_spectrum

wavelength = sdss_spectra_data["loglam"]
flux = sdss_spectra_data["flux"]

sdss_spectrum_si = parse_to_si(wavelength, flux)

plot_physical_spectrum(
    spectrum=sdss_spectrum_si,
    title=f"Espectro de SDSS (Unidades SI) - objID: {sdss_sample_object['bestObjID']})",
)



# %% [markdown]
#  #### Gaia - Flujo y Longitud de Onda para una objeto

# %%
import services.plotting as spl
from specutils import Spectrum
import astropy.units as u

spl.plot_physical_spectrum(
    spectrum=gaia_object_spectrum, title="Espectro de Gaia para un objeto"
)

# # %% [markdown]
# # #### Gaia - Flujo y Longitud de Onda para múltiples objetos

# # %%
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = pd.concat(
#     [
#         v[0].to_table().to_pandas().assign(source=key.split()[-1].replace(".xml", ""))
#         for key, v in gaia_spectra_data.items()
#     ],
#     ignore_index=True,
# )


# %% [markdown]
# ## Comparación de espectros del pipeline
#
# Carga una pareja aleatoria Gaia-SDSS de los datos ya extraídos por el pipeline
# y muestra ambos espectros lado a lado.

# %%
import random
import h5py
import numpy as np
import astropy.units as u
from pathlib import Path
from specutils import Spectrum
from services.plotting import plot_physical_spectrum

_h5_path = Path("dataset_extraction_pipeline") / "data" / "06_training_data" / "training.h5"

with h5py.File(str(_h5_path), "r") as _f:
    _i              = random.randrange(_f["gaia_source_id"].shape[0])
    _gaia_source_id = int(_f["gaia_source_id"][_i])
    _sdss_obj_id    = int(_f["sdss_obj_id"][_i])
    _gaia_wave      = _f["gaia_wavelength_nm"][:]
    _sdss_wave      = _f["sdss_wavelength_aa"][:]
    _gaia_flux      = _f["X"][_i]
    _sdss_flux      = _f["y"][_i]

sdss_spectrum = Spectrum(
    spectral_axis=_sdss_wave * u.AA
  , flux=_sdss_flux * u.Unit("erg / (cm2 s AA)") * 1e-17
)
gaia_spectrum = Spectrum(
    spectral_axis=_gaia_wave * u.nm
  , flux=_gaia_flux * u.Unit("W / (m2 nm)")
)

plot_physical_spectrum(sdss_spectrum, title=f"SDSS  objID={_sdss_obj_id}")
plot_physical_spectrum(gaia_spectrum, title=f"Gaia  source_id={_gaia_source_id}")

# %% [markdown]
# ### Comparación conjunta SDSS vs Gaia (mismo objeto, espectros superpuestos)

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.linestyle": "--"})

# Convert both spectra to Å and normalize flux to [0, 1] for overlay
_sdss_wave_aa = sdss_spectrum.spectral_axis.to(u.AA).value
_gaia_wave_aa = gaia_spectrum.spectral_axis.to(u.AA).value

def _norm(y):
    lo, hi = y.min(), y.max()
    return (y - lo) / (hi - lo) if hi > lo else y - lo

_sdss_flux_n = _norm(sdss_spectrum.flux.value)
_gaia_flux_n = _norm(gaia_spectrum.flux.value)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(_sdss_wave_aa, _sdss_flux_n, color="steelblue",  linewidth=0.9, label=f"SDSS  objID={_sdss_obj_id}")
ax.plot(_gaia_wave_aa, _gaia_flux_n, color="darkorange", linewidth=0.9, label=f"Gaia  source_id={_gaia_source_id}")
ax.set_xlabel("Longitud de onda (Å)")
ax.set_ylabel("Flujo normalizado [0–1]")
ax.set_title(f"Espectro SDSS vs Gaia — SDSS objID={_sdss_obj_id} / Gaia source_id={_gaia_source_id}")
ax.legend()
plt.tight_layout()
plt.show()

# %%
