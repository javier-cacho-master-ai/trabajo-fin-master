"""
Ejecutor de el flujo por línea de comandos.

El flujo es reanudable por omisión: cada paso comprueba si sus salidas
ya están en disco y se omite en caso afirmativo.  Utilice --force para volver
a ejecutar los pasos completados de todas formas.

Uso
---
Ejecutar el flujo completo:
    uv run python -m dataset_extraction_pipeline.run

Ejecutar sólo los pasos 1 y 2:
    uv run python -m dataset_extraction_pipeline.run --steps 1,2

Forzar la re-ejecución del paso 3 aunque sus salidas existan:
    uv run python -m dataset_extraction_pipeline.run --steps 3 --force

Tamaños de lote y paralelismo personalizados:
    uv run python -m dataset_extraction_pipeline.run --sdss-plate-end 2000 --sdss-plate-batch-size 100 --gaia-workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import fields as dc_fields
from functools import partial
from pathlib import Path

import h5py
from astropy.table import Table

from dataset_extraction_pipeline.config import PipelineConfig
from dataset_extraction_pipeline.steps import REGISTRY


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_STANDARD_LOG_ATTRS = frozenset({
    "args"
  , "asctime"
  , "created"
  , "exc_info"
  , "exc_text"
  , "filename"
  , "funcName"
  , "levelname"
  , "levelno"
  , "lineno"
  , "message"
  , "module"
  , "msecs"
  , "msg"
  , "name"
  , "pathname"
  , "process"
  , "processName"
  , "relativeCreated"
  , "stack_info"
  , "taskName"
  , "thread"
  , "threadName"
})


class _JSONFormatter(logging.Formatter):
    """Escribe sólo los campos estructurados (extra=) como JSON Lines — sin msg."""

    def format(self, record: logging.LogRecord) -> str:
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in _STANDARD_LOG_ATTRS and not k.startswith("_")
        }
        obj = {"ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S")} | extra
        return json.dumps(obj, ensure_ascii=False, default=str)


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s"
      , datefmt="%H:%M:%S"
    ))

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JSONFormatter())

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Análisis de argumentos de la línea de comandos
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    # Cargar la config para pasar los valores, en vez de meterlos hardcoded aquí.
    d = PipelineConfig()

    p = argparse.ArgumentParser(
        description="Flujo de extracción de datos astronómicos."
      , formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--steps",    default="1,2,3,4,5,6", help="Números de paso separados por comas a ejecutar")
    p.add_argument("--force",    action="store_true",    help="Vuelve a ejecutar los pasos aunque sus salidas ya estén en disco")
    p.add_argument("--data-dir", default=str(d.data_dir), help="Directorio raíz para todas las salidas de el flujo")

    # Paso 1 – Catálogo espectroscópico SDSS
    p.add_argument("--sdss-plate-start",      type=int, default=d.sdss_plate_start)
    p.add_argument("--sdss-plate-end",        type=int, default=d.sdss_plate_end)
    p.add_argument("--sdss-plate-batch-size", type=int, default=d.sdss_plate_batch_size)
    p.add_argument("--sdss-workers",          type=int, default=d.sdss_max_workers)

    # Paso 2 – Cruce inverso Gaia
    p.add_argument("--sdss-id-batch-size", type=int, default=d.sdss_id_batch_size)
    p.add_argument("--gaia-workers",       type=int, default=d.gaia_max_workers)

    # Paso 3 – Espectros SDSS
    p.add_argument("--sdss-spectra-workers", type=int, default=d.sdss_spectra_max_workers)

    # Paso 5 – Espectros Gaia
    p.add_argument("--gaia-spectra-batch-size", type=int, default=d.gaia_spectra_batch_size)
    p.add_argument("--gaia-spectra-workers",    type=int, default=d.gaia_spectra_max_workers)

    # Paso 6 – Datos de entrenamiento
    p.add_argument(
        "--sdss-wave-start"
      , type=float
      , default=d.sdss_interp_wavelength_start
      , help="Inicio de la rejilla SDSS interpolada en Ångströms"
    )
    p.add_argument(
        "--sdss-wave-end"
      , type=float
      , default=d.sdss_interp_wavelength_end
      , help="Fin de la rejilla SDSS interpolada en Ångströms"
    )
    p.add_argument(
        "--sdss-wave-n"
      , type=int
      , default=d.sdss_interp_n_points
      , help="Número de puntos en la rejilla SDSS interpolada"
    )

    # Condición de parada
    p.add_argument(
        "--target-spectra"
      , type=int
      , default=0
      , help="Seguir ampliando el rango Gaia hasta obtener este número de espectros SDSS (0 = sin límite)"
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Constructores puros de configuración
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        data_dir=Path(args.data_dir)
      , sdss_plate_start=args.sdss_plate_start
      , sdss_plate_end=args.sdss_plate_end
      , sdss_plate_batch_size=args.sdss_plate_batch_size
      , sdss_max_workers=args.sdss_workers
      , sdss_id_batch_size=args.sdss_id_batch_size
      , gaia_max_workers=args.gaia_workers
      , sdss_spectra_max_workers=args.sdss_spectra_workers
      , gaia_spectra_batch_size=args.gaia_spectra_batch_size
      , gaia_spectra_max_workers=args.gaia_spectra_workers
      , sdss_interp_wavelength_start=args.sdss_wave_start
      , sdss_interp_wavelength_end=args.sdss_wave_end
      , sdss_interp_n_points=args.sdss_wave_n
    )


def _steps_to_run(spec: str) -> list[int]:
    return sorted({int(s.strip()) for s in spec.split(",") if s.strip()})


# ---------------------------------------------------------------------------
# Estadísticas de salida (puras – sin E/S excepto lecturas)
# ---------------------------------------------------------------------------


_is_data_line = lambda l: bool(l.strip()) and not l.startswith("#")


def _ecsv_data_rows(path: Path) -> int:
    return max(0, sum(1 for l in path.read_text(encoding="utf-8").splitlines() if _is_data_line(l)) - 1)


def _count_ecsv_rows(directory: Path, pattern: str) -> int:
    return sum(map(_ecsv_data_rows, sorted(directory.glob(pattern))))


def _vot_count(path: Path) -> int:
    return len(Table.read(str(path), format="votable")) if path.exists() else 0


def _h5_count(path: Path, dataset: str) -> int:
    with h5py.File(str(path), "r") as f:
        return int(f[dataset].shape[0])


def _collect_stats(config: PipelineConfig) -> dict:
    h5_path = config.training_data_dir / "training.h5"
    return {
        "sdss_specobj_surveyed":   _count_ecsv_rows(config.sdss_specobj_dir,    "batch_*.ecsv")
      , "gaia_crossmatch_objects": _count_ecsv_rows(config.gaia_crossmatch_dir, "chunk_*.ecsv")
      , "sdss_spectra_downloaded": len(list(config.sdss_spectra_dir.glob("*.fits")))
      , "gaia_spectra_calibrated": _vot_count(config.gaia_spectra_dir / "spectra.vot")
      , "training_pairs":          _h5_count(h5_path, "gaia_source_id") if h5_path.exists() else 0
    }


# ---------------------------------------------------------------------------
# Formateo del resumen (puro – devuelve cadena)
# ---------------------------------------------------------------------------


def _fmt_duration(t: float | None) -> str:
    if t is None:
        return "(omitido)"
    m, s = divmod(t, 60)
    return f"{int(m)}m {s:.0f}s" if m else f"{s:.1f}s"


def _format_summary(step_times: dict[int, float | None], stats: dict) -> str:
    W, SEP    = 60, "-" * 47
    label_map = {k: v[0] for k, v in REGISTRY.items()}

    stat_rows = [
        ("Espectros SDSS catalogados",    stats["sdss_specobj_surveyed"])
      , ("Pares Gaia-SDSS encontrados",   stats["gaia_crossmatch_objects"])
      , ("Espectros SDSS descargados",    stats["sdss_spectra_downloaded"])
      , ("Espectros Gaia calibrados",     stats["gaia_spectra_calibrated"])
      , ("Pares de entrenamiento (HDF5)", stats["training_pairs"])
    ]
    step_rows = [(n, label_map.get(n, f"Paso {n}"), step_times[n]) for n in sorted(step_times)]
    total     = sum(t for t in step_times.values() if t is not None)

    return "\n".join([
        ""
      , "=" * W
      , "  RESUMEN DE LA CANALIZACIÓN"
      , "=" * W
      , f"\n  {'Estadística':<38} {'Valor':>8}"
      , f"  {SEP}"
      , *[f"  {row[0]:<38} {row[1]:>8,}" for row in stat_rows]
      , f"\n  {'Paso':<5} {'Descripción':<30} {'Tiempo':>8}"
      , f"  {SEP}"
      , *[f"  {row[0]:<5} {row[1]:<30} {_fmt_duration(row[2]):>8}" for row in step_rows]
      , f"  {SEP}"
      , f"  {'Total':<36} {_fmt_duration(total):>8}"
      , "=" * W
      , ""
    ])


# ---------------------------------------------------------------------------
# Ejecución de pasos
# ---------------------------------------------------------------------------


def _execute_step(
    logger: logging.Logger
  , config: PipelineConfig
  , force: bool
  , step_num: int
  , label: str
  , fn
  , is_complete_fn
) -> float | None:
    print(f"\n{'=' * 60}\n  Paso {step_num}: {label}\n{'=' * 60}")
    logger.debug("step_start", extra={"event": "step_start", "step": step_num, "label": label})
    return (
        _skip_step(logger, step_num, label)
        if not force and is_complete_fn(config)
        else _run_step(logger, config, step_num, label, fn)
    )


def _skip_step(logger: logging.Logger, step_num: int, label: str) -> None:
    print("  Ya completado; se omite.  Utilice --force para volver a ejecutar.")
    logger.debug("step_skip", extra={"event": "step_skip", "step": step_num, "label": label})


def _run_step(
    logger: logging.Logger
  , config: PipelineConfig
  , step_num: int
  , label: str
  , fn
) -> float:
    t0 = time.perf_counter()
    fn(config)
    duration = time.perf_counter() - t0
    logger.info("step_complete", extra={
        "event":      "step_complete"
      , "step":       step_num
      , "label":      label
      , "duration_s": round(duration, 2)
    })
    return duration


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------


def _gaia_crossmatch_count(config: PipelineConfig) -> int:
    return _count_ecsv_rows(config.gaia_crossmatch_dir, "chunk_*.ecsv")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    from dataclasses import replace as dc_replace

    args   = _parse_args()
    config = _build_config(args)
    config.prepare_dirs()

    log_path = config.data_dir / "pipeline.log"
    logger   = _setup_logging(log_path)

    config_dict = {f.name: str(getattr(config, f.name)) for f in dc_fields(config)}
    print("\n".join(
        ["", "=" * 60, "  PARÁMETROS DE LA CANALIZACIÓN", "=" * 60]
        + [f"  {f.name:<35} {getattr(config, f.name)}" for f in dc_fields(config)]
        + ["=" * 60, f"  Log: {log_path}", ""]
    ))
    logger.info("pipeline_start", extra={"event": "pipeline_start", "config": config_dict})

    step_times: dict[int, float | None] = {}

    steps_requested   = _steps_to_run(args.steps)
    acquisition_steps = [n for n in steps_requested if n in {1, 2, 3}]
    remaining_steps   = [n for n in steps_requested if n not in {1, 2, 3}]

    force = args.force
    while True:
        execute     = partial(_execute_step, logger, config, force)
        step_times |= {n: execute(n, *REGISTRY[n]) for n in acquisition_steps}
        count       = _gaia_crossmatch_count(config)
        if not args.target_spectra or count >= args.target_spectra:
            break
        new_end = config.sdss_plate_end * 2
        print(f"\n  {count} pares Gaia-SDSS — objetivo: {args.target_spectra}. "
              f"Ampliando rango de placas a {new_end:,}...")
        config = dc_replace(config, sdss_plate_end=new_end)
        config.prepare_dirs()
        force  = False

    step_times |= {n: execute(n, *REGISTRY[n]) for n in remaining_steps}

    stats = _collect_stats(config)
    print(_format_summary(step_times, stats))
    logger.info("pipeline_complete", extra={
        "event":            "pipeline_complete"
      , "total_duration_s": round(sum(t for t in step_times.values() if t is not None), 2)
      , "stats":            stats
      , "step_durations":   {str(k): round(v, 2) for k, v in step_times.items() if v is not None}
    })


if __name__ == "__main__":
    main()
