"""
Resolución de nombres científicos en SIMBAD.

Toma las muestras agrupadas por tipo de objeto celeste que hay en
``models/data/grouped_samples``, elige unos pocos objetos al azar de cada grupo
y les asigna su nombre científico (``main_id``) consultando SIMBAD.  El
resultado se guarda en ``grouped_samples_selected_objects.csv``, dentro de ese
mismo directorio.

La resolución se hace en dos etapas:

  1. Cruce por identificador.  SIMBAD mantiene en su tabla ``ident`` las
     referencias cruzadas de cada objeto, así que una única consulta TAP
     resuelve un lote entero de ``source_id`` de Gaia DR3.  Es mucho más
     rápido que llamar a ``Simbad.query_object`` una vez por objeto y, además,
     devuelve de paso el tipo de objeto (``otype``).
  2. Cruce por coordenadas.  No todas las fuentes de Gaia están catalogadas en
     SIMBAD bajo su identificador de Gaia DR3, así que las que fallan en la
     etapa anterior tienen una segunda oportunidad: se busca el objeto conocido
     más cercano dentro de un radio pequeño.

Todos los objetos del fichero de salida tienen nombre en SIMBAD: la cobertura
varía mucho entre grupos (las enanas blancas están al completo, mientras que
sólo una quinta parte de la secuencia principal central aparece), así que el
muestreo baraja cada grupo y descarta los objetos anónimos, sustituyéndolos por
otros sacados al azar del mismo grupo hasta reunir los pedidos.

Uso:
    python services/simbad.py [--sample-size 2] [--seed 42]
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
from astroquery.simbad import Simbad

# El directorio de muestras cuelga del proyecto, así que se deduce de la
# posición de este fichero y no del directorio desde el que se lance el script.
SOLUTION_DIR = Path(__file__).resolve().parents[1]
GROUPED_SAMPLES_DIR = SOLUTION_DIR / "models" / "data" / "grouped_samples"
OUTPUT_FILE_NAME = "grouped_samples_selected_objects.csv"

# Límite prudente para la cláusula IN() del TAP de SIMBAD.
SIMBAD_ID_BATCH_SIZE = 200

# Radio de búsqueda del cruce por coordenadas, en grados (2 segundos de arco),
# y número máximo de candidatos a los que se aplica esa segunda oportunidad.
COORDINATE_MATCH_RADIUS_DEG  = 2.0 / 3600.0
COORDINATE_MATCH_MAX_ATTEMPTS = 25

DEFAULT_SAMPLE_SIZE = 2
DEFAULT_RANDOM_SEED = 42

# Columnas del catálogo que se arrastran hasta el fichero de salida.
CATALOG_COLUMNS = (
    "source_id"
  , "ra"
  , "dec"
  , "parallax"
  , "phot_g_mean_mag"
  , "bp_rp"
  , "abs_g_mag"
  , "plate"
  , "mjd"
  , "fiberID"
)

OUTPUT_COLUMNS = (
    "group"
  , *CATALOG_COLUMNS
  , "simbad_main_id"
  , "simbad_otype"
  , "name_source"
  , "match_separation_arcsec"
)


# ---------------------------------------------------------------------------
# Puras – sin E/S
# ---------------------------------------------------------------------------


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _quoted_gaia_identifiers(source_ids: Sequence[str]) -> str:
    return ", ".join(f"'Gaia DR3 {source_id}'" for source_id in source_ids)


def discover_group_files(directory: Path) -> list[Path]:
    """Devuelve los CSV de grupos del directorio, excluyendo el de salida."""
    return sorted(
        path for path in directory.glob("*.csv")
        if path.name != OUTPUT_FILE_NAME
    )


def _empty_record() -> dict[str, object]:
    return {
        "simbad_main_id":          None
      , "simbad_otype":            None
      , "name_source":            "unresolved"
      , "match_separation_arcsec": None
    }


# ---------------------------------------------------------------------------
# Consultas a SIMBAD
# ---------------------------------------------------------------------------


def resolve_names_by_identifier(source_ids: Sequence[str]) -> dict[str, dict[str, object]]:
    """
    Resuelve un lote de ``source_id`` de Gaia DR3 mediante la tabla de
    identificadores de SIMBAD.

    Parameters:
        source_ids (Sequence[str]):
            Identificadores de Gaia DR3, como cadenas.

    Returns:
        dict: Diccionario ``source_id`` -> datos de SIMBAD, sólo con los
        objetos encontrados.
    """
    resolved: dict[str, dict[str, object]] = {}

    for batch in _batched(source_ids, SIMBAD_ID_BATCH_SIZE):
        query = (
            "SELECT identifier.id AS query_id, basic.main_id, basic.otype "
            "FROM ident AS identifier "
            "JOIN basic ON basic.oid = identifier.oidref "
            f"WHERE identifier.id IN ({_quoted_gaia_identifiers(batch)})"
        )
        result_table = Simbad.query_tap(query)
        if result_table is None:
            continue

        for row in result_table:
            # query_id llega como 'Gaia DR3 <source_id>'.
            source_id = str(row["query_id"]).rsplit(" ", 1)[-1]
            resolved[source_id] = {
                "simbad_main_id":          str(row["main_id"])
              , "simbad_otype":            str(row["otype"])
              , "name_source":             "gaia_dr3_ident"
              , "match_separation_arcsec": 0.0
            }

    return resolved


def resolve_name_by_coordinates(
    ra: float, dec: float, radius_deg: float = COORDINATE_MATCH_RADIUS_DEG
) -> dict[str, object]:
    """
    Busca el objeto de SIMBAD más cercano a unas coordenadas dadas.

    Parameters:
        ra (float):  Ascensión recta en grados (ICRS).
        dec (float): Declinación en grados (ICRS).
        radius_deg (float): Radio de búsqueda en grados.

    Returns:
        dict: Datos de SIMBAD del objeto más cercano, o un registro vacío si no
        hay ninguno dentro del radio.
    """
    query = (
        "SELECT TOP 1 basic.main_id, basic.otype, "
        f"DISTANCE(POINT('ICRS', basic.ra, basic.dec), POINT('ICRS', {ra}, {dec})) AS separation "
        "FROM basic "
        f"WHERE CONTAINS(POINT('ICRS', basic.ra, basic.dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1 "
        "ORDER BY separation"
    )
    result_table = Simbad.query_tap(query)
    if result_table is None or not len(result_table):
        return _empty_record()

    row = result_table[0]
    return {
        "simbad_main_id":          str(row["main_id"])
      , "simbad_otype":            str(row["otype"])
      , "name_source":             "coordinate_match"
      , "match_separation_arcsec": float(row["separation"]) * 3600.0
    }


# ---------------------------------------------------------------------------
# Selección de objetos
# ---------------------------------------------------------------------------


def select_named_objects(
    frame: pd.DataFrame
  , group: str
  , sample_size: int
  , seed: int
) -> pd.DataFrame:
    """
    Elige al azar ``sample_size`` objetos de un grupo que tengan nombre en
    SIMBAD.

    Baraja el grupo entero y va resolviendo lotes de candidatos por
    identificador, quedándose con los que SIMBAD conoce y descartando el resto:
    cada objeto anónimo se sustituye por otro sacado al azar del mismo grupo.
    Si el grupo se agota sin reunir los objetos pedidos, se reintenta con el
    cruce por coordenadas sobre los candidatos anónimos.

    Parameters:
        frame (pd.DataFrame): Catálogo del grupo.
        group (string):       Nombre del grupo (el del fichero, sin extensión).
        sample_size (int):    Número de objetos a seleccionar.
        seed (int):           Semilla para que el muestreo sea reproducible.

    Returns:
        pd.DataFrame: Filas seleccionadas, todas con nombre de SIMBAD.

    Raises:
        ValueError: Si el grupo no tiene suficientes objetos catalogados.
    """
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled["source_id"] = shuffled["source_id"].astype(str)

    selected: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []

    for batch in _batched(shuffled.index.to_list(), SIMBAD_ID_BATCH_SIZE):
        candidates = shuffled.loc[batch]
        resolved   = resolve_names_by_identifier(candidates["source_id"].to_list())

        for _, candidate in candidates.iterrows():
            record = {
                "group": group
              , **{column: candidate.get(column) for column in CATALOG_COLUMNS}
            }
            match = resolved.get(candidate["source_id"])

            if match is None:
                unresolved.append(record)
            else:
                selected.append(record | match)
                if len(selected) == sample_size:
                    return pd.DataFrame(selected, columns=list(OUTPUT_COLUMNS))

    # El grupo se ha agotado sin reunir los objetos pedidos: se da una segunda
    # oportunidad a los candidatos anónimos buscándolos por posición, por si
    # SIMBAD los tiene catalogados bajo otro identificador.
    for record in unresolved[:COORDINATE_MATCH_MAX_ATTEMPTS]:
        if len(selected) == sample_size:
            break

        match = resolve_name_by_coordinates(float(record["ra"]), float(record["dec"]))
        if match["simbad_main_id"] is not None:
            selected.append(record | match)

    if len(selected) < sample_size:
        raise ValueError(
            f"El grupo '{group}' sólo tiene {len(selected)} objetos con nombre en "
            f"SIMBAD, y se pedían {sample_size}."
        )

    return pd.DataFrame(selected, columns=list(OUTPUT_COLUMNS))


def build_selection(
    directory: Path = GROUPED_SAMPLES_DIR
  , sample_size: int = DEFAULT_SAMPLE_SIZE
  , seed: int = DEFAULT_RANDOM_SEED
) -> pd.DataFrame:
    """
    Construye la tabla de objetos seleccionados para todos los grupos.

    Parameters:
        directory (Path):  Directorio con los CSV de muestras agrupadas.
        sample_size (int): Objetos a seleccionar por grupo.
        seed (int):        Semilla del muestreo.

    Returns:
        pd.DataFrame: Selección de todos los grupos concatenada.
    """
    group_files = discover_group_files(directory)
    if not group_files:
        raise FileNotFoundError(f"No hay ficheros de grupos en {directory}")

    selections = []
    for group_file in group_files:
        group = group_file.stem
        print(f"Resolviendo {sample_size} objetos del grupo '{group}'...")

        selection = select_named_objects(
            pd.read_csv(group_file), group, sample_size, seed
        )
        for _, row in selection.iterrows():
            print(f"  {row['source_id']} -> {row['simbad_main_id']} ({row['simbad_otype']})")

        selections.append(selection)

    return pd.concat(selections, ignore_index=True)


def save_selection(selection: pd.DataFrame, directory: Path = GROUPED_SAMPLES_DIR) -> Path:
    """Guarda la selección en el CSV de salida y devuelve su ruta."""
    output_path = directory / OUTPUT_FILE_NAME
    selection.to_csv(output_path, index=False)

    return output_path


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--directory", type=Path, default=GROUPED_SAMPLES_DIR
      , help="Directorio con los CSV de muestras agrupadas."
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE
      , help="Número de objetos a seleccionar por grupo."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_RANDOM_SEED
      , help="Semilla del muestreo aleatorio."
    )
    arguments = parser.parse_args()

    selection   = build_selection(arguments.directory, arguments.sample_size, arguments.seed)
    output_path = save_selection(selection, arguments.directory)

    print(f"\nGuardados {len(selection)} objetos en {output_path}")


if __name__ == "__main__":
    main()
