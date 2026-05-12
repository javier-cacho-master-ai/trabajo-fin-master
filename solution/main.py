# %% [markdown]
# # Uso de redes generativas para la mejora de la señal astronómica
#
# El presente jupyter notebook contiene el código usado para el entrenamiento y creación del modelo generativo.
#

# %%
from astroquery.gaia import Gaia
from astropy.table import Table
from pathlib import Path


gaia_query_path = Path("services","queries", "gaia_panstarrs1_random_subset.adql")
gaia_query_template = gaia_query_path.read_text(encoding="utf-8")
gaia_query = gaia_query_template.format(start_index=0, end_index=100)

print(f"Executing Step 1: {gaia_query_path.name}...")
gaia_query_job = Gaia.launch_job_async(gaia_query)
gaia_job_results = gaia_query_job.get_results() or Table()

display(gaia_job_results)

# %%
import requests
from astropy.io import ascii
from astropy.table import Table

# -----------------------------------------------------------------
# 1. Query a few Pan‑STARRS1 objects by their objID
# -----------------------------------------------------------------
# For reference check https://mast.stsci.edu/api/v0/MastApiTutorial.html and https://ps1images.stsci.edu/ps1_dr2_api.html
# Replace these with the `original_ext_source_id` values from your Gaia cross‑match

gaia_crossmatch_ids = gaia_job_results[
    "original_ext_source_id"
]  # take the first 100 for testing

import requests

# 1. The dedicated endpoint for Pan-STARRS DR2 'mean' catalog
url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.json"

# Print the top-level keys to understand the structure
response = requests.get(url)
data = response.json()
display(data.keys())


# # 2. Extract valid IDs, convert to string, and join with commas
# # (This handles the MaskedColumn safely by ignoring masked values)
# if hasattr(gaia_crossmatch_ids, "mask"):
#     valid_ids = gaia_crossmatch_ids[~gaia_crossmatch_ids.mask].tolist()
#     id_string = ",".join(map(str, valid_ids))
# else:
#     valid_ids = gaia_crossmatch_ids

# # # 3. Build the payload
# payload = {
#     "objID": valid_ids
#     # # Ask for specific columns to avoid server timeouts
#     # "columns": [
#     #     "objID",
#     #     "ra",
#     #     "dec",
#     #     "gMeanPSFMag",
#     #     "rMeanPSFMag",
#     #     "iMeanPSFMag"
#     # ]
# }

# # # 4. Send the POST request directly to MAST
# response = requests.post(url, json=payload)
# ps1_json = response.json()

# display(response.content)

# # 5. Convert the JSON response into an Astropy Table
# if "data" in ps1_json:
#     ps1_table = Table(ps1_json["data"])
#     display(ps1_table)
# else:
#     print("API Error or no data returned:")
#     print(ps1_json)


    # %% [markdown]
#
