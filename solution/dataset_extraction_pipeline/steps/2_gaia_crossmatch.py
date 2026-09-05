"""
Cruce inverso Gaia – Paso 2.

Lee los metadatos SDSS SpecObj del paso 1, divide los bestObjID en fragmentos
y consulta el TAP de Gaia en sentido inverso: dado un conjunto de IDs SDSS
fotométricos, devuelve las fuentes Gaia correspondientes que además tienen
espectros BP/RP continuos.

El resultado de cada fragmento se enriquece con los metadatos espectrales SDSS
(plate, mjd, fiberID, redshift) y se escribe como fichero ECSV.  Gracias a que
partimos del catálogo espectroscópico SDSS, la tasa de coincidencia Gaia es muy
alta (~20-50 %) frente al ~1 % del flujo fotométrico anterior.
"""

from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path

from astropy.table import Table, vstack
from astroquery.gaia import Gaia

from dataset_extraction_pipeline._retry import retry
from dataset_extraction_pipeline.config import PipelineConfig

_QUERIES_DIR = Path(__file__).parent.parent / "queries"

# Schema canónico de la tabla de salida (usado para fragmentos vacíos).
_CROSSMATCH_DTYPES: dict[str, type] = {
    "source_id":                    np.int64
  , "ra":                           np.float64
  , "dec":                          np.float64
  , "phot_bp_mean_mag":             np.float32
  , "phot_rp_mean_mag":             np.float32
  , "phot_bp_mean_flux_over_error": np.float32
  , "phot_rp_mean_flux_over_error": np.float32
  , "astrometric_excess_noise":     np.float32
  , "bp_rp":                        np.float32
  , "teff_gspphot":                 np.float32
  , "logg_gspphot":                 np.float32
  , "mh_gspphot":                   np.float32
  , "vbroad":                       np.float32
  , "alphafe_gspspec":              np.float32
  , "original_ext_source_id":       np.int64
  , "angular_distance":             np.float32
  , "plate":                        np.int32
  , "mjd":                          np.int32
  , "fiberID":                      np.int32
  , "redshift":                     np.float32
}


# ---------------------------------------------------------------------------
# Puras – sin E/S
# ---------------------------------------------------------------------------


def _load_tables(directory: Path, pattern: str) -> Table:
    files = sorted(directory.glob(pattern))
    if not files:
        return Table()
    tables = [Table.read(str(f), format="ascii.ecsv") for f in files]
    non_empty = [t for t in tables if len(t)]
    return vstack(non_empty) if non_empty else Table()


def _chunks(items: list, size: int) -> list[list]:
    it = iter(items)
    return list(iter(lambda: list(islice(it, size)), []))


def _chunk_path(output_dir: Path, index: int) -> Path:
    return output_dir / f"chunk_{index:05d}.ecsv"


def _is_pending(output_dir: Path, indexed: tuple[int, list]) -> bool:
    index, _ = indexed
    return not _chunk_path(output_dir, index).exists()


def _empty_table() -> Table:
    return Table({col: np.array([], dtype=dt) for col, dt in _CROSSMATCH_DTYPES.items()})


def _merge_sdss_metadata(
    gaia_table: Table
  , sdss_by_id: dict[str, dict]
) -> Table:
    """Add SDSS spectroscopic columns to a Gaia result table."""
    ids = [str(sid) for sid in gaia_table["original_ext_source_id"]]
    gaia_table["plate"]    = np.array([int(sdss_by_id[i]["plate"])    for i in ids], dtype=np.int32)
    gaia_table["mjd"]      = np.array([int(sdss_by_id[i]["mjd"])      for i in ids], dtype=np.int32)
    gaia_table["fiberID"]  = np.array([int(sdss_by_id[i]["fiberID"])  for i in ids], dtype=np.int32)
    gaia_table["redshift"] = np.array([float(sdss_by_id[i]["redshift"]) for i in ids], dtype=np.float32)
    return gaia_table


# ---------------------------------------------------------------------------
# Con efectos – E/S en el límite
# ---------------------------------------------------------------------------


@retry()
def _fetch_gaia(query_template: str, ids: list[str]) -> Table:
    job = Gaia.launch_job_async(query_template.format(sdss_ids=",".join(ids)))
    return job.get_results() or Table()


def _fetch_and_save(
    output_dir: Path
  , query_template: str
  , sdss_by_id: dict[str, dict]
  , indexed: tuple[int, list]
) -> Path | None:
    index, ids = indexed
    path = _chunk_path(output_dir, index)

    gaia_table = _fetch_gaia(query_template, ids)

    if not len(gaia_table):
        _empty_table().write(str(path), format="ascii.ecsv", overwrite=True)
        print(f"[gaia_crossmatch] fragmento {index:05d}: sin coincidencias")
        return None

    # Filter to rows whose SDSS ID is in our metadata dict (defensive)
    mask       = [str(sid) in sdss_by_id for sid in gaia_table["original_ext_source_id"]]
    gaia_table = gaia_table[mask]

    if not len(gaia_table):
        _empty_table().write(str(path), format="ascii.ecsv", overwrite=True)
        return None

    merged = _merge_sdss_metadata(gaia_table, sdss_by_id)
    merged.write(str(path), format="ascii.ecsv", overwrite=True)
    print(f"[gaia_crossmatch] fragmento {index:05d}: {len(merged):>5,} pares  →  {path.name}")
    return path


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def is_complete(config: PipelineConfig) -> bool:
    """Devuelve True cuando existe un fichero de fragmento para cada lote de IDs SDSS."""
    specobj_files = list(config.sdss_specobj_dir.glob("batch_*.ecsv"))
    if not specobj_files:
        return False
    specobj_table = _load_tables(config.sdss_specobj_dir, "batch_*.ecsv")
    if not len(specobj_table):
        return False
    n_chunks = len(_chunks(list(specobj_table["bestObjID"].astype(str)), config.sdss_id_batch_size))
    return all(_chunk_path(config.gaia_crossmatch_dir, i).exists() for i in range(n_chunks))


def run(config: PipelineConfig) -> list[Path]:
    output_dir = config.gaia_crossmatch_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    specobj_table = _load_tables(config.sdss_specobj_dir, "batch_*.ecsv")
    if not len(specobj_table):
        print("[gaia_crossmatch] AVISO: sin datos SDSS SpecObj — ejecuta primero el paso 1")
        return []

    all_ids   = list(specobj_table["bestObjID"].astype(str))
    sdss_by_id = {str(row["bestObjID"]): row for row in specobj_table}

    query_template = (_QUERIES_DIR / "gaia_by_sdss_ids.adql").read_text(encoding="utf-8")

    indexed_chunks = list(enumerate(_chunks(all_ids, config.sdss_id_batch_size)))
    pending = list(filter(partial(_is_pending, output_dir), indexed_chunks))
    cached  = [_chunk_path(output_dir, i) for i, _ in indexed_chunks if not _is_pending(output_dir, (i, []))]

    print(f"[gaia_crossmatch] {len(pending)} fragmento(s) por obtener, {len(cached)} ya en caché")

    fetch = partial(_fetch_and_save, output_dir, query_template, sdss_by_id)

    with ThreadPoolExecutor(max_workers=config.gaia_max_workers) as pool:
        results = list(pool.map(fetch, pending))

    return cached + [p for p in results if p is not None]
