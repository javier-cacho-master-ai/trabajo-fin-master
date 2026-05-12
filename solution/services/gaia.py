from astroquery.gaia import Gaia
from astropy.table import Table
from pathlib import Path
from datetime import datetime

def fetch_gaia_data(query_folder, query_file_name):
  """
  Obtiene los datos de Gaia ejecutando una consulta ADQL.
  
  Parameters:
    query_folder (string):
      Ruta que contiene el fichero de la consulta.
    query_file_name (string): Nombre del fichero de la consulta.

  Returns:
    astropy.table: Tabla con los datos obtenidos.
  """  

  
  gaia_query_path = Path("queries", "gaia_panstarrs1_random_subset.adql")
  gaia_query_template = gaia_query_path.read_text(encoding="utf-8")
  gaia_query = gaia_query_template.format(start_index=0, end_index=100)

  print(f"Loaded query from {gaia_query_path.name}. Executing on ESA servers...")

  print(f"Executing Step 1: {gaia_query_path.name}...")
  gaia_query_job = Gaia.launch_job_async(gaia_query)

  return gaia_query_job.get_results() or Table()
