import os
import sys
import traceback
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Definimos la configuración de iSpec
ISPEC_DIR = "/Users/carlasequero/iSpec"

if ISPEC_DIR not in sys.path:
    sys.path.insert(0, ISPEC_DIR)

import ispec

ISPEC_CONFIG = {

    "code": "spectrum",

    # Resolución aproximada de SDSS
    "resolution": 2000,

    # Rango de las muestras del conjunto de datos
    "wave_min_nm": 500.0,
    "wave_max_nm": 900.0,

    # Parámetros iniciales por defecto si no tenemos los de Gaia
    "default_teff": 5500.0,
    "default_logg": 4.0,
    "default_mh": 0.0,

 
    # Recursos de iSpec
    "atmosphere_dir":
        "/Users/carlasequero/iSpec/input/atmospheres/MARCS.GES/",

    "atomic_linelist_file":
        "/Users/carlasequero/iSpec/input/linelists/transitions/"
        "VALD.300_1100nm/atomic_lines.tsv",

    "solar_abundances_file":
        "/Users/carlasequero/iSpec/input/abundances/"
        "Grevesse.2007/stdatom.dat",

    "isotopes_file":
        "/Users/carlasequero/iSpec/input/isotopes/SPECTRUM.lst",

    "line_regions_file":
        "/Users/carlasequero/iSpec/input/regions/"
        "47000_SPECTRUM/"
        "spectrum_synth_good_for_params_all.txt",


    # Elimina de la lista atómica las líneas teóricamente demasiado débiles.
    "minimum_theoretical_depth": 0.01,

    # Número máximo de iteraciones del ajuste de iSpec
    "max_iterations": 6,
 
}

def load_ispec_resources(config=ISPEC_CONFIG):

    print("Cargando modelos atmosféricos...")
    modeled_layers_pack = ispec.load_modeled_layers_pack(
        config["atmosphere_dir"]
    )

    print("Cargando abundancias solares...")
    solar_abundances = ispec.read_solar_abundances(
        config["solar_abundances_file"]
    )

    print("Cargando isótopos...")
    isotopes = ispec.read_isotope_data(
        config["isotopes_file"]
    )

    print("Cargando lista atómica...")
    atomic_linelist = ispec.read_atomic_linelist(
        config["atomic_linelist_file"],
        wave_base=config["wave_min_nm"],
        wave_top=config["wave_max_nm"]
    )

    # Filtrar líneas extremadamente débiles
    if "theoretical_depth" in atomic_linelist.dtype.names:
        atomic_linelist = atomic_linelist[
            atomic_linelist["theoretical_depth"]
            >= config["minimum_theoretical_depth"]
        ]

    print(f"Líneas atómicas cargadas: {len(atomic_linelist)}")

    print("Cargando regiones espectrales...")
    line_regions = ispec.read_line_regions(
        config["line_regions_file"]
    )

    print(f"Regiones cargadas: {len(line_regions)}")

    print("\nRecursos de iSpec cargados correctamente.")

    return {
        "modeled_layers_pack": modeled_layers_pack,
        "solar_abundances": solar_abundances,
        "isotopes": isotopes,
        "atomic_linelist": atomic_linelist,
        "line_regions": line_regions
    }

def build_ispec_spectrum(
    wavelength_aa,
    flux,
    error=None,
    wave_min_nm=500.0,
    wave_max_nm=900.0
):

    wavelength_aa = np.asarray(wavelength_aa, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # Å -> nm
    wavelength_nm = wavelength_aa / 10.0

    if error is None:
        error = np.zeros_like(flux, dtype=float)
    else:
        error = np.asarray(error, dtype=float)

    # Filtrar valores inválidos
    valid = (
        np.isfinite(wavelength_nm)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (wavelength_nm >= wave_min_nm)
        & (wavelength_nm <= wave_max_nm)
    )

    wavelength_nm = wavelength_nm[valid]
    flux = flux[valid]
    error = error[valid]

    if len(wavelength_nm) == 0:
        raise ValueError("El espectro no contiene puntos válidos.")

    # Ordenar por longitud de onda
    order = np.argsort(wavelength_nm)

    wavelength_nm = wavelength_nm[order]
    flux = flux[order]
    error = error[order]

    spectrum = np.recarray(
        len(wavelength_nm),
        dtype=[
            ("waveobs", float),
            ("flux", float),
            ("err", float)
        ]
    )

    spectrum["waveobs"] = wavelength_nm
    spectrum["flux"] = flux
    spectrum["err"] = error

    return spectrum

def normalize_spectrum_for_ispec(
    spectrum,
    resolution=2000
):

    continuum_model = ispec.fit_continuum(
        spectrum,
        from_resolution=resolution,
        model="Splines",
        order="median+max",
        median_wave_range=1.0,
        max_wave_range=5.0,
        automatic_strong_line_detection=True,
        strong_line_probability=0.5,
        use_errors_for_fitting=False
    )

    normalized_spectrum = ispec.normalize_spectrum(
        spectrum,
        continuum_model,
        consider_continuum_errors=False
    )

    return normalized_spectrum

def get_initial_parameters(
    gaia_row=None,
    config=ISPEC_CONFIG
):

    teff = config["default_teff"]
    logg = config["default_logg"]
    mh = config["default_mh"]

    if gaia_row is not None:

        if (
            "teff_gspphot" in gaia_row.index
            and pd.notna(gaia_row["teff_gspphot"])
        ):
            teff = float(gaia_row["teff_gspphot"])

        if (
            "logg_gspphot" in gaia_row.index
            and pd.notna(gaia_row["logg_gspphot"])
        ):
            logg = float(gaia_row["logg_gspphot"])

        if (
            "mh_gspphot" in gaia_row.index
            and pd.notna(gaia_row["mh_gspphot"])
        ):
            mh = float(gaia_row["mh_gspphot"])

    # Abundancia alfa esperada
    alpha = ispec.determine_abundance_enchancements(
        mh
    )

    try:
        vmic = ispec.estimate_vmic(
            teff,
            logg,
            mh
        )
    except Exception:
        vmic = 1.0

    try:
        vmac = ispec.estimate_vmac(
            teff,
            logg,
            mh
        )
    except Exception:
        vmac = 3.0

    vsini = 2.0

    return {
        "teff": teff,
        "logg": logg,
        "MH": mh,
        "alpha": alpha,
        "vmic": vmic,
        "vmac": vmac,
        "vsini": vsini
    }

def select_valid_regions(
    spectrum,
    line_regions
):

    wave_min = np.min(
        spectrum["waveobs"]
    )

    wave_max = np.max(
        spectrum["waveobs"]
    )

    valid = (
        (line_regions["wave_base"] >= wave_min)
        & (line_regions["wave_top"] <= wave_max)
    )

    regions = line_regions[valid]

    return regions

def analyze_spectrum_with_ispec(
    wavelength_aa,
    flux,
    error=None,
    gaia_row=None,
    resources=None,
    config=ISPEC_CONFIG
):

    if resources is None:
        raise ValueError(
            "Debes cargar primero los recursos de iSpec."
        )

    # ========================================================
    # 1. Crear espectro iSpec
    # ========================================================

    spectrum = build_ispec_spectrum(
        wavelength_aa=wavelength_aa,
        flux=flux,
        error=error,
        wave_min_nm=config["wave_min_nm"],
        wave_max_nm=config["wave_max_nm"]
    )

    if len(spectrum) < 50:
        raise ValueError(
            "El espectro contiene muy pocos puntos."
        )

    # ========================================================
    # 2. SNR
    # ========================================================

    try:
        snr = ispec.estimate_snr(
            spectrum["flux"],
            num_points=10
        )
    except Exception:
        snr = np.nan

    # ========================================================
    # 3. Normalizar
    # ========================================================

    normalized = normalize_spectrum_for_ispec(
        spectrum,
        resolution=config["resolution"]
    )

    # ========================================================
    # 4. Parámetros iniciales
    # ========================================================

    initial = get_initial_parameters(
        gaia_row=gaia_row,
        config=config
    )

    teff = initial["teff"]
    logg = initial["logg"]
    MH = initial["MH"]
    alpha = initial["alpha"]

    vmic = initial["vmic"]
    vmac = initial["vmac"]
    vsini = initial["vsini"]

    # ========================================================
    # 5. Continuo fijo = 1
    # ========================================================

    continuum_model = ispec.fit_continuum(
        normalized,
        fixed_value=1.0,
        model="Fixed value"
    )

    # ========================================================
    # 6. Seleccionar regiones
    # ========================================================

    line_regions = select_valid_regions(
        normalized,
        resources["line_regions"]
    )

    if len(line_regions) == 0:
        raise ValueError(
            "No hay regiones válidas para este espectro."
        )

    # ========================================================
    # 7. Parámetros que iSpec optimizará
    # ========================================================

    free_params = [
        "teff",
        "logg",
        "MH"
    ]

    # ========================================================
    # 8. Ajuste
    # ========================================================

    (
        synthetic_spectrum,
        params,
        errors,
        abundances_found,
        loggf_found,
        status,
        stats
    ) = ispec.model_spectrum(

        normalized,
        continuum_model,

        resources["modeled_layers_pack"],
        resources["atomic_linelist"],
        resources["isotopes"],
        resources["solar_abundances"],

        free_params,

        teff,
        logg,
        MH,
        alpha,

        vmic,
        vmac,
        vsini,

        0.6,

        config["resolution"],

        regions=line_regions,

        code=config["code"],

        max_iterations=config["max_iterations"]
    )

    # ========================================================
    # 9. Output
    # ========================================================

    result = {

        "snr": snr,

        "initial_teff": teff,
        "initial_logg": logg,
        "initial_mh": MH,

        "teff": params.get(
            "teff",
            np.nan
        ),

        "logg": params.get(
            "logg",
            np.nan
        ),

        "mh": params.get(
            "MH",
            np.nan
        ),

        "alpha": params.get(
            "alpha",
            alpha
        ),

        "vmic": params.get(
            "vmic",
            vmic
        ),

        "vmac": params.get(
            "vmac",
            vmac
        ),

        "vsini": params.get(
            "vsini",
            vsini
        ),

        "teff_err": errors.get(
            "teff",
            np.nan
        ),

        "logg_err": errors.get(
            "logg",
            np.nan
        ),

        "mh_err": errors.get(
            "MH",
            np.nan
        ),

        "status": status
    }

    if isinstance(stats, dict):

        for key, value in stats.items():

            if np.isscalar(value):
                result[
                    f"fit_{key}"
                ] = value

    return result

