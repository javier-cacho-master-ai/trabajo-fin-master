# %% [markdown]
# # Uso de redes generativas para la mejora de la señal astronómica
# 
# El presente jupyter notebook contiene el código usado para el entrenamiento y creación del modelo generativo.
# 

# %%
from astroquery.gaia import Gaia
from astropy.table import Table
from pathlib import Path

# 1. Read the raw ADQL string from your file
query_path = Path("queries", "cross-match-apogee.adql")
adql_query = query_path.read_text(encoding="utf-8")

print(f"Loaded query from {query_path.name}. Executing on ESA servers...")

# 2. Launch the job
job = Gaia.launch_job_async(adql_query)

# 3. Retrieve the results
job_results = job.get_results() or Table()  # Ensure we have an empty table if no results
df = job_results.to_pandas() 

print(f"Match complete! Found {len(df)} stars.")
# df.to_csv("training_data.csv", index=False)

display(df.head())
# %% [markdown]
# 


