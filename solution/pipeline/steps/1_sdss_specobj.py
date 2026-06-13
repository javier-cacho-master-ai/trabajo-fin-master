"""
Catálogo espectroscópico SDSS – Paso 1.

Descarga los metadatos de todos los espectros primarios del catálogo SDSS SpecObj
en lotes por rango de número de placa.  Partir del catálogo espectroscópico (en lugar
del cruce fotométrico de Gaia) garantiza que cada objeto tiene un espectro SDSS real,
eliminando el 98-99 % de pérdida del flujo inverso anterior.

Cada lote se escribe en su propio fichero ECSV de inmediato; una ejecución fallida
reanuda desde el primer lote ausente.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from astropy.table import Table, vstack
from astroquery.sdss import SDSS

from pipeline._retry import retry
from pipeline.config import PipelineConfig
from services.querying import sanitize_adql

_QUERIES_DIR = Path(__file__).parent.parent / "queries"
_PAGE_SIZE   = 250   # SDSS SQL endpoint hard limit per request


# ---------------------------------------------------------------------------
# Puras – sin E/S
# ---------------------------------------------------------------------------


def _batch_ranges(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(s, min(s + size, end)) for s in range(start, end, size)]


def _batch_path(output_dir: Path, batch: tuple[int, int]) -> Path:
    s, e = batch
    return output_dir / f"batch_{s:05d}_{e:05d}.ecsv"


def _is_pending(output_dir: Path, batch: tuple[int, int]) -> bool:
    return not _batch_path(output_dir, batch).exists()


# ---------------------------------------------------------------------------
# Con efectos – E/S en el límite
# ---------------------------------------------------------------------------


@retry()
def _fetch_page(query: str, data_release: int) -> Table:
    result = SDSS.query_sql(query, data_release=data_release)
    return result if result is not None else Table()


def _iter_pages(
    query_template: str
  , data_release: int
  , s: int
  , e: int
  , min_id: int = 0
) -> Iterator[Table]:
    page = _fetch_page(
        query_template.format(plate_start=s, plate_end=e, min_id=min_id)
      , data_release
    )
    match len(page):
        case 0:
            return
        case full if full == _PAGE_SIZE:
            yield page
            yield from _iter_pages(query_template, data_release, s, e, int(page["specObjID"][-1]))  # type: ignore[index]
        case _:
            yield page


def _fetch_batch(
    query_template: str
  , data_release: int
  , batch: tuple[int, int]
) -> Table:
    s, e  = batch
    pages = list(_iter_pages(query_template, data_release, s, e))
    return vstack(pages) if pages else Table()


def _fetch_and_save(
    output_dir: Path
  , query_template: str
  , data_release: int
  , batch: tuple[int, int]
) -> Path:
    table = _fetch_batch(query_template, data_release, batch)
    path  = _batch_path(output_dir, batch)
    table.write(str(path), format="ascii.ecsv", overwrite=True)
    print(f"[sdss_specobj] placas {batch[0]}-{batch[1]}: {len(table):>6,} espectros  →  {path.name}")
    return path


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def is_complete(config: PipelineConfig) -> bool:
    """Devuelve True cuando todos los lotes de placas esperados están en disco."""
    batches = _batch_ranges(config.sdss_plate_start, config.sdss_plate_end, config.sdss_plate_batch_size)
    return all(_batch_path(config.sdss_specobj_dir, b).exists() for b in batches)


def run(config: PipelineConfig) -> list[Path]:
    output_dir = config.sdss_specobj_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    query_template = sanitize_adql(
        (_QUERIES_DIR / "sdss_specobj_by_plate.adql").read_text(encoding="utf-8")
    )

    all_batches = _batch_ranges(config.sdss_plate_start, config.sdss_plate_end, config.sdss_plate_batch_size)
    pending     = list(filter(partial(_is_pending, output_dir), all_batches))
    cached      = [_batch_path(output_dir, b) for b in all_batches if not _is_pending(output_dir, b)]

    print(f"[sdss_specobj] {len(pending)} lote(s) por obtener, {len(cached)} ya en caché")

    fetch = partial(_fetch_and_save, output_dir, query_template, config.sdss_data_release)

    with ThreadPoolExecutor(max_workers=config.sdss_max_workers) as pool:
        fetched = list(pool.map(fetch, pending))

    return cached + fetched
