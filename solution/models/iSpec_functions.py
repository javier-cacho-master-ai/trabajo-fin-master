import os
import sys
import traceback
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ============================================================
# CONFIGURACIÓN DE ISPEC
# ============================================================

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

    # ========================================================
    # Recursos de iSpec
    # ========================================================

    "atmosphere_dir":
        "/Users/carlasequero/iSpec/input/atmospheres/"
        "MARCS.GES/",

    "atomic_linelist_file":
        "/Users/carlasequero/iSpec/input/linelists/transitions/"
        "VALD.300_1100nm/atomic_lines.tsv",

    "solar_abundances_file":
        "/Users/carlasequero/iSpec/input/abundances/"
        "Grevesse.2007/stdatom.dat",

    "isotopes_file":
        "/Users/carlasequero/iSpec/input/isotopes/"
        "SPECTRUM.lst",

    "line_regions_file":
        "/Users/carlasequero/iSpec/input/regions/"
        "47000_SPECTRUM/"
        "spectrum_synth_good_for_params_all.txt",

    # Elimina líneas teóricamente demasiado débiles
    "minimum_theoretical_depth": 0.01,

    # Número máximo de iteraciones
    "max_iterations": 6,
}


# ============================================================
# CARGA DE RECURSOS
# ============================================================

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

    print(
        f"Líneas atómicas cargadas: "
        f"{len(atomic_linelist)}"
    )

    print("Cargando regiones espectrales...")

    line_regions = ispec.read_line_regions(
        config["line_regions_file"]
    )

    print(
        f"Regiones cargadas: "
        f"{len(line_regions)}"
    )

    print(
        "\nRecursos de iSpec cargados correctamente."
    )

    return {
        "modeled_layers_pack": modeled_layers_pack,
        "solar_abundances": solar_abundances,
        "isotopes": isotopes,
        "atomic_linelist": atomic_linelist,
        "line_regions": line_regions
    }


# ============================================================
# CONVERSIÓN AL FORMATO ISPEC
# ============================================================

def build_ispec_spectrum(
    wavelength_aa,
    flux,
    error=None,
    wave_min_nm=500.0,
    wave_max_nm=900.0
):

    wavelength_aa = np.asarray(
        wavelength_aa,
        dtype=float
    )

    flux = np.asarray(
        flux,
        dtype=float
    )

    # Angstrom -> nm
    wavelength_nm = wavelength_aa / 10.0

    # ========================================================
    # Errores
    # ========================================================

    if error is None:

        error = np.zeros_like(
            flux,
            dtype=float
        )

    else:

        error = np.asarray(
            error,
            dtype=float
        )

    # ========================================================
    # Filtrar valores inválidos
    # ========================================================

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

        raise ValueError(
            "El espectro no contiene puntos válidos."
        )

    # ========================================================
    # Ordenar por longitud de onda
    # ========================================================

    order = np.argsort(
        wavelength_nm
    )

    wavelength_nm = wavelength_nm[order]
    flux = flux[order]
    error = error[order]

    # ========================================================
    # Crear estructura de iSpec
    # ========================================================

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


# ============================================================
# NORMALIZACIÓN AL CONTINUO
# ============================================================

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


# ============================================================
# PARÁMETROS INICIALES
# ============================================================

def get_initial_parameters(
    gaia_row=None,
    config=ISPEC_CONFIG
):

    teff = config["default_teff"]
    logg = config["default_logg"]
    mh = config["default_mh"]

    # ========================================================
    # Usar parámetros Gaia si los tenemos
    # ========================================================

    if gaia_row is not None:

        if (
            "teff_gspphot" in gaia_row.index
            and pd.notna(
                gaia_row["teff_gspphot"]
            )
        ):

            teff = float(
                gaia_row["teff_gspphot"]
            )

        if (
            "logg_gspphot" in gaia_row.index
            and pd.notna(
                gaia_row["logg_gspphot"]
            )
        ):

            logg = float(
                gaia_row["logg_gspphot"]
            )

        if (
            "mh_gspphot" in gaia_row.index
            and pd.notna(
                gaia_row["mh_gspphot"]
            )
        ):

            mh = float(
                gaia_row["mh_gspphot"]
            )

    # ========================================================
    # Alpha
    # ========================================================

    alpha = ispec.determine_abundance_enchancements(
        mh
    )

    # ========================================================
    # Microturbulencia
    # ========================================================

    try:

        vmic = ispec.estimate_vmic(
            teff,
            logg,
            mh
        )

    except Exception:

        vmic = 1.0

    # ========================================================
    # Macroturbulencia
    # ========================================================

    try:

        vmac = ispec.estimate_vmac(
            teff,
            logg,
            mh
        )

    except Exception:

        vmac = 3.0

    # ========================================================
    # Rotación inicial
    # ========================================================

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


# ============================================================
# SELECCIÓN DE REGIONES
# ============================================================

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

    return line_regions[valid]


# ============================================================
# ANÁLISIS DE UN ESPECTRO
# ============================================================

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
    # 2. Estimar SNR
    # ========================================================

    try:

        snr = ispec.estimate_snr(
            spectrum["flux"],
            num_points=10
        )

    except Exception:

        snr = np.nan

    # ========================================================
    # 3. Normalizar al continuo
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
    # 5. Seleccionar regiones válidas
    # ========================================================

    line_regions = select_valid_regions(
        normalized,
        resources["line_regions"]
    )

    if len(line_regions) == 0:

        raise ValueError(
            "No hay regiones válidas dentro "
            "del rango del espectro."
        )

    # ========================================================
    # 6. Ajustar las máscaras a las líneas observadas
    # ========================================================

    try:

        line_regions = ispec.adjust_linemasks(
            normalized,
            line_regions,
            max_margin=0.5
        )

    except Exception:

        # Si no se pueden reajustar,
        # conservamos las regiones originales
        pass

    # ========================================================
    # 7. Crear segmentos alrededor de las líneas
    # ========================================================

    segments = ispec.create_segments_around_lines(
        line_regions,
        margin=0.25
    )

    # ========================================================
    # 8. Recortar el espectro a los segmentos
    # ========================================================

    wfilter = ispec.create_wavelength_filter(
        normalized,
        regions=segments
    )

    normalized_selected = normalized[
        wfilter
    ]

    if len(normalized_selected) < 10:

        raise ValueError(
            "Muy pocos puntos después de aplicar "
            "las regiones espectrales."
        )

    # ========================================================
    # 9. Continuo fijo
    # ========================================================

    continuum_model = ispec.fit_continuum(
        normalized_selected,
        fixed_value=1.0,
        model="Fixed value"
    )

    # ========================================================
    # 10. Parámetros libres
    # ========================================================

    free_params = [
        "teff",
        "logg",
        "MH"
    ]

    # No estamos ajustando abundancias individuales todavía
    free_abundances = None

    # Tampoco estamos ajustando log(gf)
    linelist_free_loggf = None

    # ========================================================
    # 11. Otros parámetros iniciales requeridos por iSpec
    # ========================================================

    initial_limb_darkening_coeff = 0.6

    initial_R = config["resolution"]

    # Se supone que el espectro ya está en su sistema
    # de referencia para este análisis.
    initial_vrad = 0.0

    # ========================================================
    # 12. Ajuste con iSpec
    # ========================================================

    (
        obs_spec,
        modeled_synth_spectrum,
        params,
        errors,
        abundances_found,
        loggf_found,
        status,
        stats_linemasks

    ) = ispec.model_spectrum(

        normalized_selected,
        continuum_model,

        resources["modeled_layers_pack"],
        resources["atomic_linelist"],
        resources["isotopes"],
        resources["solar_abundances"],

        # ----------------------------------------------------
        # No ajustamos abundancias ni log(gf)
        # ----------------------------------------------------

        free_abundances,
        linelist_free_loggf,

        # ----------------------------------------------------
        # Parámetros iniciales
        # ----------------------------------------------------

        teff,
        logg,
        MH,
        alpha,

        vmic,
        vmac,
        vsini,

        initial_limb_darkening_coeff,

        initial_R,
        initial_vrad,

        # ----------------------------------------------------
        # Parámetros libres
        # ----------------------------------------------------

        free_params,

        # ----------------------------------------------------
        # Regiones utilizadas
        # ----------------------------------------------------

        segments=segments,
        linemasks=line_regions,

        # ----------------------------------------------------
        # Configuración
        # ----------------------------------------------------

        enhance_abundances=False,

        use_errors=False,

        vmic_from_empirical_relation=False,

        vmac_from_empirical_relation=False,

        max_iterations=config["max_iterations"],

        tmp_dir=None,

        code=config["code"]
    )

    # ========================================================
    # 13. Resultado
    # ========================================================

    result = {

        "snr": snr,

        "n_regions": len(
            line_regions
        ),

        "n_points_fit": len(
            normalized_selected
        ),

        # ----------------------------------------------------
        # Parámetros iniciales
        # ----------------------------------------------------

        "initial_teff": teff,
        "initial_logg": logg,
        "initial_mh": MH,

        # ----------------------------------------------------
        # Parámetros ajustados
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Errores
        # ----------------------------------------------------

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

    # ========================================================
    # 14. Estadísticas del ajuste
    # ========================================================

    if isinstance(
        stats_linemasks,
        dict
    ):

        for key, value in stats_linemasks.items():

            if np.isscalar(value):

                result[
                    f"fit_{key}"
                ] = value

    return result