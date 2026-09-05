from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración inmutable para toda el flujo de extracción."""

    data_dir: Path = Path("data")

    # Paso 1 – Catálogo espectroscópico SDSS
    # Las placas SDSS DR13 van de ~266 a ~9190; lotes de 50 ≈ 20 000 espectros por lote
    sdss_plate_start:      int = 1267 
    sdss_plate_end:        int = 2500   # 500 placas ≈ 200 000 espectros
    sdss_plate_batch_size: int = 50

    # Paso 1 y 2 – Parámetros compartidos SDSS / Gaia
    sdss_id_batch_size: int = 200   # límite de la cláusula IN() de los TAP de SDSS y Gaia
    sdss_max_workers:   int = 1     # el TAP de SDSS acepta mal la concurrencia alta
    sdss_data_release:  int = 13

    # Paso 2 – Cruce inverso Gaia
    gaia_max_workers: int = 4

    # Paso 3 – Descarga de espectros SDSS
    sdss_spectra_max_workers: int = 4

    # Paso 5 – Espectros calibrados de Gaia
    gaia_spectra_batch_size:  int = 50
    gaia_spectra_max_workers: int = 2

    # Paso 6 – Transformación a formato de entrenamiento
    sdss_interp_wavelength_start: float = 3_800.0
    sdss_interp_wavelength_end:   float = 9_200.0
    sdss_interp_n_points:         int   = 3_600

    # ---- directorios de salida derivados (sólo lectura) ----------------------

    @property
    def sdss_specobj_dir(self) -> Path:
        return self.data_dir / "01_sdss_specobj"

    @property
    def gaia_crossmatch_dir(self) -> Path:
        return self.data_dir / "02_gaia_crossmatch"

    @property
    def sdss_spectra_dir(self) -> Path:
        return self.data_dir / "03_sdss_spectra"

    @property
    def pruned_dir(self) -> Path:
        return self.data_dir / "04_pruned"

    @property
    def gaia_spectra_dir(self) -> Path:
        return self.data_dir / "05_gaia_spectra"

    @property
    def training_data_dir(self) -> Path:
        return self.data_dir / "06_training_data"

    # ---- inicialización de directorios ---------------------------------------

    @property
    def _all_output_dirs(self) -> tuple[Path, ...]:
        return (
            self.sdss_specobj_dir
          , self.gaia_crossmatch_dir
          , self.sdss_spectra_dir
          , self.pruned_dir
          , self.gaia_spectra_dir
          , self.training_data_dir
        )

    def prepare_dirs(self) -> None:
        """Crea todos los directorios de salida del flujo (idempotente)."""
        for d in self._all_output_dirs:
            d.mkdir(parents=True, exist_ok=True)
