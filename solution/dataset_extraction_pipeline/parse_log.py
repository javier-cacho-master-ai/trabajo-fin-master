"""
Extrae estadísticas de ejecución del fichero pipeline.log (formato JSON Lines).

Uso:
    uv run python -m dataset_extraction_pipeline.parse_log
    uv run python -m dataset_extraction_pipeline.parse_log --log dataset_extraction_pipeline/data/pipeline.log
    uv run python -m dataset_extraction_pipeline.parse_log --run -1   # última ejecución (por omisión)
    uv run python -m dataset_extraction_pipeline.parse_log --run 0    # primera ejecución
"""

from __future__ import annotations

import argparse
import json
from functools import reduce
from pathlib import Path
from typing import Iterator

from dataset_extraction_pipeline.config import PipelineConfig

W    = 60
SEP  = "=" * W
THIN = "-" * 47

STAT_LABELS: dict[str, str] = {
    "sdss_specobj_surveyed":   "Espectros SDSS catalogados"
  , "gaia_crossmatch_objects": "Pares Gaia-SDSS encontrados"
  , "sdss_spectra_downloaded": "Espectros SDSS descargados"
  , "gaia_spectra_calibrated": "Espectros Gaia calibrados"
  , "training_pairs":          "Pares de entrenamiento (HDF5)"
}


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_records(log_path: Path) -> Iterator[dict]:
    for line in log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                pass


def _fold_runs(
    acc: tuple[list[list[dict]], list[dict]]
  , record: dict
) -> tuple[list[list[dict]], list[dict]]:
    """Accumulate records into (completed_runs, current_run) groups."""
    completed, current = acc
    match record.get("event"):
        case "pipeline_start":    return completed, [record]
        case "pipeline_complete": return completed + [current + [record]], []
        case _:                   return completed, current + [record]


def _build_step(record: dict) -> tuple[str, dict]:
    match record["event"]:
        case "step_complete":
            return record["step"], {"label": record.get("label", ""), "duration_s": record.get("duration_s")}
        case "step_skip":
            return record["step"], {"label": record.get("label", ""), "skipped": True}


def _build_run(records: list[dict]) -> dict:
    by_event = {r["event"]: r for r in records}
    steps    = dict(
        _build_step(r)
        for r in records
        if r.get("event") in {"step_complete", "step_skip"}
    )
    return {
        "started_at":       by_event["pipeline_start"]["ts"]
      , "finished_at":      by_event["pipeline_complete"]["ts"]
      , "config":           by_event["pipeline_start"].get("config", {})
      , "steps":            steps
      , "stats":            by_event["pipeline_complete"].get("stats", {})
      , "total_duration_s": by_event["pipeline_complete"].get("total_duration_s")
    }


def _load_runs(log_path: Path) -> list[dict]:
    records      = list(_parse_records(log_path))
    completed, _ = reduce(_fold_runs, records, ([], []))
    return [_build_run(group) for group in completed]


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "(omitido)"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.0f}s" if m else f"{s:.1f}s"


def _fmt_stat(label: str, val: object) -> str:
    return f"  {label:<38} {val:>8,}" if isinstance(val, int) else f"  {label:<38} {val:>8}"


def _fmt_step(num: str, step: dict) -> str:
    return f"  Paso {num}: {step.get('label', f'Paso {num}'):<30} {_fmt_duration(step.get('duration_s')):>8}"


def _print_run(run: dict) -> None:
    config_lines = [f"  {k:<35} {v}" for k, v in run["config"].items()]

    stat_lines = (
        [_fmt_stat(label, run["stats"].get(key, "—")) for key, label in STAT_LABELS.items()]
        if run["stats"] else []
    )

    step_lines = [
        _fmt_step(n, s)
        for n, s in sorted(run["steps"].items(), key=lambda kv: int(kv[0]))
    ]

    print("\n".join([
        f"\n{SEP}"
      , f"  Ejecución: {run['started_at']}  →  {run.get('finished_at', '(incompleta)')}"
      , SEP
      , "\n  PARÁMETROS"
      , f"  {THIN}"
      , *config_lines
      , *(["\n  ESTADÍSTICAS", f"  {THIN}", *stat_lines] if stat_lines else [])
      , *(["\n  TIEMPOS POR PASO", f"  {THIN}", *step_lines, f"  {THIN}",
           f"  {'Total':<36} {_fmt_duration(run['total_duration_s']):>8}"] if step_lines else [])
      , f"{SEP}\n"
    ]))


def _suggest_plate_end(run: dict, target: int) -> None:
    config  = run.get("config", {})
    matched = run.get("stats", {}).get("gaia_crossmatch_objects", 0)
    start   = int(config.get("sdss_plate_start", 266))
    end     = int(config.get("sdss_plate_end",   766))
    scanned = end - start

    if matched <= 0 or scanned <= 0:
        print("  No hay suficientes datos para estimar el rango necesario.")
        return

    density      = matched / scanned
    required_end = start + int(target / density)

    print("\n".join([
        f"\n  ESTIMACIÓN PARA {target:,} PARES GAIA-SDSS"
      , f"  {THIN}"
      , f"  Pares encontrados en esta ejecución   {matched:>8,}"
      , f"  Placas escaneadas                     {scanned:>8,}"
      , f"  Densidad observada (pares/placa)      {density:>8.3f}"
      , f"  --sdss-plate-end necesario            {required_end:>8,}"
      , "\n  Comando sugerido:"
      , f"  uv run python -m dataset_extraction_pipeline.run --force "
        f"--sdss-plate-start {start} "
        f"--sdss-plate-end {required_end} "
        f"--gaia-workers {int(config.get('gaia_max_workers', 4))}"
    ]))


def main() -> None:
    p = argparse.ArgumentParser(description="Analiza el fichero de log de el flujo.")
    p.add_argument("--log",    default=str(PipelineConfig().data_dir / "pipeline.log"), help="Ruta al fichero de log JSON Lines")
    p.add_argument("--run",    type=int, default=-1,        help="Índice de la ejecución a mostrar (0 = primera, -1 = última)")
    p.add_argument("--all",    action="store_true",         help="Mostrar todas las ejecuciones registradas")
    p.add_argument("--target", type=int, default=None,      help="Estimar --sdss-plate-end necesario para obtener este número de pares Gaia-SDSS")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Fichero de log no encontrado: {log_path}")
        return

    runs = _load_runs(log_path)
    if not runs:
        print("No se encontraron ejecuciones completas en el log.")
        return

    targets = runs if args.all else [runs[args.run]]
    for run in targets:
        _print_run(run)
    if args.target and not args.all:
        _suggest_plate_end(runs[args.run], args.target)

    print(f"\nTotal de ejecuciones en el log: {len(runs)}")


if __name__ == "__main__":
    main()
