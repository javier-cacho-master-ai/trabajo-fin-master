# %% [markdown]
#  # Uso de redes generativas para la mejora de la señal astronómica
# 
#  El presente jupyter notebook contiene el código usado para el entrenamiento y creación del modelo generativo.
# 

# %% [markdown]
# # Extracción de los datos

# %% [markdown]
#  ### Datos filtrados de Gaia con SDSS

# %%
from pathlib import Path
from astroquery.gaia import Gaia
from astropy.table import Table

gaia_query_path = Path("services", "queries", "gaia_sdss_random_subset.adql")
gaia_query_template = gaia_query_path.read_text(encoding="utf-8")
gaia_query = gaia_query_template.format(start_index=0, end_index=10000)

print(f"Executing Step 1: {gaia_query_path.name}...")
gaia_query_job = Gaia.launch_job_async(gaia_query)
gaia_job_results = gaia_query_job.get_results() or Table()

display(gaia_job_results)

# %% [markdown]
#  ### Datos espectrales de Gaia
# 
# 
# 
#  Para los datos espectrales debemos usar el [DataLink service](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html#datalink-service-public-and-authenticated)

# %% [markdown]
#  #### Obtención de los datos

# %%
source_ids = ",".join(gaia_job_results['source_id'].astype(str))

xp_data = Gaia.load_data(
    ids=source_ids,
    data_release="Gaia DR3",
    retrieval_type="XP_SAMPLED",
    # retrieval_type="XP_CONTINUOUS",
    data_structure="INDIVIDUAL",
    format="votable",
    dump_to_file=False, # Datos en memoria
)

display(xp_data)


# %% [markdown]
#  #### Graficado de los datos

# %%

## Ejemplo de múltiples sources
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.concat(
    [v[0].to_table().to_pandas().assign(
        source=key.split()[-1].replace('.xml', '')
    ) for key, v in xp_data.items()],
    ignore_index=True
)

sns.lineplot(data=df, x='wavelength', y='flux', hue='source')
plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 1. Build a flat DataFrame for a single spectrum (first source in xp_data)
first_key = next(iter(xp_data))
raw = xp_data[first_key][0].to_table().to_pandas()
spectrum_df = pd.DataFrame({
    'Wavelengt': raw['wavelength'],
    'Flux': raw['flux']
})

# 2. Set the aesthetic theme globally (same as your example)
sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.linestyle": "--"})

plt.figure(figsize=(12, 5))

# 3. Plot a line whose colour changes from blue (short λ) to red (long λ) 
#    (Seaborn does not provide this directly; we use a LineCollection)
x = spectrum_df['Wavelengt'].values
y = spectrum_df['Flux'].values

norm = Normalize(vmin=x.min(), vmax=x.max())
cmap = plt.get_cmap('coolwarm')          # blue -> red

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=0.8)
lc.set_array(x)                           # colour each segment by its x-start
plt.gca().add_collection(lc)

# 4. Axis labels and limits
plt.xlabel('Longitud de Onda (nm)')
plt.ylabel('Flujo (W / (nm m²))')
plt.xlim(x.min(), x.max())
plt.ylim(y.min(), y.max())

# Optional colour bar to show the mapping
cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca())
cbar.set_label('Wavelength (nm)')

plt.tight_layout()
plt.show()



# %%
import re
from astroquery.sdss import SDSS
from astropy.io import ascii
from astropy.table import Table

def sanitize_adql(raw_query: str) -> str:
    """Removes SQL comments and compresses whitespace into single spaces."""
    # 1. Remove block comments (/* ... */)
    # The re.DOTALL flag allows '.' to match newline characters
    no_blocks = re.sub(r'/\*.*?\*/', '', raw_query, flags=re.DOTALL)
    
    # 2. Remove inline comments (-- ...)
    no_inlines = re.sub(r'--.*', '', no_blocks)
    
    # 3. Compress remaining whitespace (tabs, newlines, multiple spaces)
    return re.sub(r'\s+', ' ', no_inlines).strip()

gaia_crossmatch_ids = gaia_job_results["original_ext_source_id"]
query_filter_ids = ",".join(gaia_crossmatch_ids.astype(str))

sdss_query_path = Path("services", "queries", "sdss.adql")
sdss_query_template = sdss_query_path.read_text(encoding="utf-8")
sdss_query_template_sanitized = sanitize_adql(sdss_query_template)
sdss_query = sdss_query_template_sanitized.format(filter_ids=query_filter_ids)

sdss_query_result = SDSS.query_sql(sdss_query, data_release=13)

# print(sdss_query_result.keys.filter['flux'])

display(sdss_query_result)



# %%
import astropy.units as u

spectra = SDSS.get_spectra(
    plate=sdss_query_result["plate"][0],
    mjd=sdss_query_result["mjd"][0],
    fiberID=sdss_query_result["fiberID"][0],
    data_release=13,
)

data = spectra[0][1].data


wavelength = 0.1 * 10 ** data["loglam"]
flux = data["flux"] * 1e-17 * u.erg / (u.cm**2 * u.s * u.AA)
 
plt.figure(figsize=(12, 5))
plt.plot(wavelength, flux)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Flux")
plt.show()

# %% [markdown]
#  Mostrar el Gráfico

# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 1. We must force the raw FITS arrays into a DataFrame for Seaborn
spectrum_df = pd.DataFrame({
    'Wavelength': 10 ** spec_data['loglam'],
    'Flux': spec_data['flux']
})

# 2. Set the aesthetic theme globally (this replaces plt.grid and background tweaks)
sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.linestyle": "--"})

plt.figure(figsize=(12, 5))

# 3. The Seaborn plot command
sns.lineplot(
    data=spectrum_df, 
    x='Wavelength', 
    y='Flux', 
    color='black', 
    linewidth=0.8
)

plt.ylabel(r"Flux ($10^{-17} \text{ erg/cm}^2\text{/s/\AA}$)")
plt.xlim(3800, 9200) 
plt.show()


