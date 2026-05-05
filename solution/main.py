# %% [markdown]
# # Uso de redes generativas para la mejora de la señal astronómica
#
# El presente jupyter notebook contiene el código usado para el entrenamiento y creación del modelo generativo.
#

# %%
import sys

from astroquery.gaia import Gaia
from astropy.table import Table
from pathlib import Path
from datetime import datetime

gaia_query_path = Path("queries", "gaia_panstarrs1_random_subset.adql")
gaia_query_template = gaia_query_path.read_text(encoding="utf-8")
gaia_query = gaia_query_template.format(start_index=0, end_index=100)

print(f"Loaded query from {gaia_query_path.name}. Executing on ESA servers...")

print(f"Executing Step 1: {gaia_query_path.name}...")
gaia_query_job = Gaia.launch_job_async(gaia_query)
gaia_job_results = gaia_query_job.get_results() or Table()

display(gaia_job_results)

# %% 
import requests
from astropy.io import ascii
import panstarrs1 as pan

# -----------------------------------------------------------------
# 1. Query a few Pan‑STARRS1 objects by their objID
# -----------------------------------------------------------------
# For reference check https://mast.stsci.edu/api/v0/MastApiTutorial.html and https://ps1images.stsci.edu/ps1_dr2_api.html
# Replace these with the `original_ext_source_id` values from your Gaia cross‑match

gaia_crossmatch_ids = gaia_job_results["original_ext_source_id"][:100]  # take the first 100 for testing

# Define the request
request = {
    "service": "Mast.Catalogs.Filtered.Tic",
    "format": "json",
    "params": {
        "columns": "*",
        "filters": pan.set_filters({
            "": gaia_crossmatch_ids # The key must match the column name in the database (e.g., 'ID' or 'obsid')
        })
    }
}

# Run the query using your existing function
headers, content = pan.mast_query(request)

display(content)
# %% [markdown]
#
