"""
Descarga de espectros SDSS – Paso 3.

Lee la tabla de cruce Gaia-SDSS del paso 2 y descarga en paralelo el espectro
FITS de cada objeto.  Cada fichero FITS se escribe en disco de inmediato tras su
descarga; una ejecución fallida reanuda desde el primer espectro ausente.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import NamedTuple

from astropy.table import Table, vstack
from astroquery.sdss import SDSS

from pipeline._retry import retry
from pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Tipos del dominio
# ---------------------------------------------------------------------------


class _SpecRequest(NamedTuple):
    obj_id:   str
    plate:    int
    mjd:      int
    fiber_id: int


# ---------------------------------------------------------------------------
# Puras – sin E/S
# ---------------------------------------------------------------------------


def _load_tables(directory: Path, pattern: str) -> Table:
    files = sorted(directory.glob(pattern))
    if not files:
        return Table()
    tables     = [Table.read(str(f), format="ascii.ecsv") for f in files]
    non_empty  = [t for t in tables if len(t)]
    return vstack(non_empty) if non_empty else Table()


def _to_request(row) -> _SpecRequest:
    return _SpecRequest(
        obj_id=str(row["original_ext_source_id"])
      , plate=int(row["plate"])
      , mjd=int(row["mjd"])
      , fiber_id=int(row["fiberID"])
    )


def _spectrum_path(output_dir: Path, req: _SpecRequest) -> Path:
    return output_dir / f"{req.obj_id}.fits"


def _is_pending(output_dir: Path, req: _SpecRequest) -> bool:
    return not _spectrum_path(output_dir, req).exists()


def _unique_by_obj(requests: Iterable[_SpecRequest]) -> list[_SpecRequest]:
    """Conserva una sola petición por objID; el fichero de salida se nombra sólo
    por objID, de modo que los duplicados del cruce competirían por el mismo
    fichero (PermissionError en Windows al escribir en paralelo)."""
    return list({req.obj_id: req for req in requests}.values())


# ---------------------------------------------------------------------------
# Con efectos – E/S en el límite
# ---------------------------------------------------------------------------


@retry()
def _get_spectra(plate: int, mjd: int, fiber_id: int, data_release: int):
    return SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiber_id, data_release=data_release) or []


def _fetch_and_save(output_dir: Path, data_release: int, req: _SpecRequest) -> Path | None:
    try:
        hdulists = _get_spectra(req.plate, req.mjd, req.fiber_id, data_release)
    except Exception as exc:
        print(f"[sdss_spectra] descarta {req.obj_id} tras reintentos: {exc}")
        return None
    if not hdulists:
        print(f"[sdss_spectra] sin espectro: objID={req.obj_id}")
        return None
    path = _spectrum_path(output_dir, req)
    hdulists[0].writeto(str(path), overwrite=True)
    print(f"[sdss_spectra] guardado  →  {path.name}")
    return path


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def is_complete(config: PipelineConfig) -> bool:
    """Devuelve True cuando cada par Gaia-SDSS tiene su fichero FITS en disco."""
    if not list(config.gaia_crossmatch_dir.glob("chunk_*.ecsv")):
        return False
    table = _load_tables(config.gaia_crossmatch_dir, "chunk_*.ecsv")
    return bool(len(table)) and all(
        _spectrum_path(config.sdss_spectra_dir, _to_request(row)).exists()
        for row in table
    )


def run(config: PipelineConfig, crossmatch_dir: Path | None = None) -> list[Path]:
    crossmatch_dir = crossmatch_dir or config.gaia_crossmatch_dir
    output_dir     = config.sdss_spectra_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table        = _load_tables(crossmatch_dir, "chunk_*.ecsv")
    all_requests = _unique_by_obj(map(_to_request, table))

    pending = list(filter(partial(_is_pending, output_dir), all_requests))
    cached  = [_spectrum_path(output_dir, r) for r in all_requests if not _is_pending(output_dir, r)]

    print(f"[sdss_spectra] {len(pending)} espectro(s) por descargar, {len(cached)} ya en caché")

    fetch = partial(_fetch_and_save, output_dir, config.sdss_data_release)

    with ThreadPoolExecutor(max_workers=config.sdss_spectra_max_workers) as pool:
        results = list(pool.map(fetch, pending))

    return cached + [p for p in results if p is not None]
