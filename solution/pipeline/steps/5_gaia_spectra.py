"""
Extracción de espectros calibrados de Gaia – Paso 5.

Lee la tabla Gaia depurada del paso 4, divide los identificadores de fuente en
lotes y llama a gaiaxpy.calibrate() sobre cada lote en paralelo.

El resultado de cada lote se persiste de inmediato en disco como un fichero VOTable
(identificadores de fuente + matrices de flujo e incertidumbre), de modo que un
fallo a mitad de la ejecución reanuda desde el primer lote sin caché.  Tras
completar todos los lotes, los ficheros individuales se combinan en la tabla final
«spectra.vot».  La rejilla de longitudes de onda compartida se guarda en «sampling.npy».
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import gaiaxpy
from astropy.table import Table, vstack as table_vstack

from pipeline._retry import retry
from pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Puras – sin E/S
# ---------------------------------------------------------------------------


def _chunks(items: list, size: int) -> list[list]:
    it = iter(items)
    return list(iter(lambda: list(islice(it, size)), []))


def _extract_source_ids(table: Table) -> list[int]:
    return list(table["source_id"].astype(int))


def _batch_path(output_dir: Path, i: int) -> Path:
    return output_dir / f"batch_{i:04d}.vot"


def _sampling_path(output_dir: Path) -> Path:
    return output_dir / "sampling.npy"


def _is_batch_cached(output_dir: Path, indexed: tuple[int, list]) -> bool:
    i, _ = indexed
    return _batch_path(output_dir, i).exists()


def _combine_batches(output_dir: Path, n_batches: int) -> Table:
    return (
        table_vstack(list(map(
            lambda i: Table.read(str(_batch_path(output_dir, i)), format="votable"),
            range(n_batches),
        )))
        if n_batches > 0 else Table()
    )


# ---------------------------------------------------------------------------
# Con efectos – E/S en el límite
# ---------------------------------------------------------------------------


@retry()
def _calibrate_and_cache(
    output_dir: Path
  , indexed: tuple[int, list]
) -> None:
    i, source_ids = indexed
    spectra_df, sampling = gaiaxpy.calibrate(source_ids)

    batch_table = Table({
        "source_id":  spectra_df["source_id"].values.astype(np.int64)
      , "flux":       np.vstack(spectra_df["flux"].values)
      , "flux_error": np.vstack(spectra_df["flux_error"].values)
    })
    batch_table.write(str(_batch_path(output_dir, i)), format="votable", overwrite=True)

    # La rejilla de muestreo es idéntica para todos los lotes; sólo se escribe una vez
    sp = _sampling_path(output_dir)
    if not sp.exists():
        np.save(str(sp), sampling)

    print(f"[gaia_spectra] lote {i:04d}: {len(spectra_df)} fuentes en caché")


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def is_complete(config: PipelineConfig) -> bool:
    """Devuelve True cuando los ficheros de salida combinados están presentes en disco."""
    d = config.gaia_spectra_dir
    return (d / "spectra.vot").exists() and _sampling_path(d).exists()


def run(config: PipelineConfig, pruned_gaia_path: Path | None = None) -> list[Path]:
    pruned_gaia_path = pruned_gaia_path or (config.pruned_dir / "gaia.ecsv")
    output_dir = config.gaia_spectra_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gaia_table = Table.read(str(pruned_gaia_path), format="ascii.ecsv")
    indexed_batches = list(enumerate(_chunks(_extract_source_ids(gaia_table), config.gaia_spectra_batch_size)))

    pending  = list(filter(partial(lambda d, t: not _is_batch_cached(d, t), output_dir), indexed_batches))
    n_cached = len(indexed_batches) - len(pending)

    print(f"[gaia_spectra] {len(indexed_batches)} lote(s): {len(pending)} por obtener, {n_cached} en caché")

    calibrate = partial(_calibrate_and_cache, output_dir)

    with ThreadPoolExecutor(max_workers=config.gaia_spectra_max_workers) as pool:
        list(pool.map(calibrate, pending))

    spectra_path = output_dir / "spectra.vot"
    combined = _combine_batches(output_dir, len(indexed_batches))
    combined.write(str(spectra_path), format="votable", overwrite=True)

    print(f"[gaia_spectra] {len(combined)} espectros  →  {spectra_path.name}")

    return [spectra_path, _sampling_path(output_dir)]
