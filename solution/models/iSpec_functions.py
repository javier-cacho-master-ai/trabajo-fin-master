import os
import sys
import traceback
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Configuramos iSpec
ISPEC_DIR = "/Users/carlasequero/iSpec"

if ISPEC_DIR not in sys.path:
    sys.path.insert(0, ISPEC_DIR)

import ispec


ISPEC_CONFIG = {

    "code": "spectrum",

    # Resolución para trabajar con SDSS
    "resolution": 2000,

    # Rango de los datos
    "wave_min_nm": 500.0,
    "wave_max_nm": 900.0,

    # Parámetros a usar por defecto en caso de no tener Gaia
    "default_teff": 5500.0,
    "default_logg": 4.0,
    "default_mh": 0.0,

    # Definimos los recursos de iSpec
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

    # Elimina líneas teóricamente débiles
    "minimum_theoretical_depth": 0.01,

    # Número máximo de iteraciones para ajustar espectros
    "max_iterations": 6,
}


# Función para cargar recursos
def load_ispec_resources(config=ISPEC_CONFIG):

    modeled_layers_pack = ispec.load_modeled_layers_pack(config["atmosphere_dir"])

    solar_abundances = ispec.read_solar_abundances(config["solar_abundances_file"])

    isotopes = ispec.read_isotope_data(config["isotopes_file"])

    atomic_linelist = ispec.read_atomic_linelist(
        config["atomic_linelist_file"],
        wave_base=config["wave_min_nm"],
        wave_top=config["wave_max_nm"]
    )

    # Filtramos líneas débiles
    if "theoretical_depth" in atomic_linelist.dtype.names:
        atomic_linelist = atomic_linelist[
            atomic_linelist["theoretical_depth"]
            >= config["minimum_theoretical_depth"]
        ]

    line_regions = ispec.read_line_regions(config["line_regions_file"])

    print("\nRecursos de iSpec cargados correctamente.")

    return {
        "modeled_layers_pack": modeled_layers_pack,
        "solar_abundances": solar_abundances,
        "isotopes": isotopes,
        "atomic_linelist": atomic_linelist,
        "line_regions": line_regions
    }


# Convertir espectro a formato aceptado por iSpec

def build_ispec_spectrum(
    wavelength_aa,
    flux,
    error=None,
    wave_min_nm=500.0,
    wave_max_nm=900.0
):

    wavelength_aa = np.asarray(wavelength_aa, dtype=float)

    flux = np.asarray(flux, dtype=float)

    # Angstrom a nm
    wavelength_nm = wavelength_aa / 10.0

    # Si no pasamos error,
    # se definen todos los valores del espectro a 1 para darles el mismo peso en el ajuste
    if error is None:
         error = np.ones_like(flux, dtype=float)

    else:
        error = np.asarray(
            error,
            dtype=float
        )

    # Filtramos por longitudes de onda a analizar
    mask = ((wavelength_nm >= wave_min_nm) & (wavelength_nm <= wave_max_nm)) 

    wavelength_nm = wavelength_nm[mask]
    flux = flux[mask]
    error = error[mask]

    # Ordenamos longitudes de onda
    order = np.argsort(wavelength_nm)

    wavelength_nm = wavelength_nm[order]
    flux = flux[order]
    error = error[order]

    # Definimos estructura para iSpec
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


# Función para normalizar al continuo
def normalize_spectrum_for_ispec(spectrum, resolution=2000):

    continuum_model = ispec.fit_continuum(
        spectrum, # Espectro a normalizar
        from_resolution=resolution,
        model="Splines",
        order="median+max",
        median_wave_range=1.0,
        max_wave_range=5.0,
        automatic_strong_line_detection=True, # Detecta lineas que entorpezcan ajuste
        strong_line_probability=0.5, # Umbral de definición de strong line
        use_errors_for_fitting=False # Errores asociados al flujo no se usan para ajustar al continuo
    )

    normalized_spectrum = ispec.normalize_spectrum(
        spectrum,
        continuum_model,
        consider_continuum_errors=False
    )

    return normalized_spectrum


# Definir parámetros inciales
def get_initial_parameters(gaia_row=None, config=ISPEC_CONFIG):

    teff = config["default_teff"]
    logg = config["default_logg"]
    mh = config["default_mh"]

    # Si tenemos información de Gaia, la usamos
    if gaia_row is not None:

        if ("teff_gspphot" in gaia_row.index and pd.notna(gaia_row["teff_gspphot"])):
            teff = float(gaia_row["teff_gspphot"])

        if ("logg_gspphot" in gaia_row.index and pd.notna(gaia_row["logg_gspphot"])):
            logg = float(gaia_row["logg_gspphot"])

        if ("mh_gspphot" in gaia_row.index and pd.notna(gaia_row["mh_gspphot"])):
            mh = float(gaia_row["mh_gspphot"])

    # Definimos alpha a partir de la metalicidad
    alpha = ispec.determine_abundance_enchancements(mh)

    # Estimamos la microturbulencia
    try:
        vmic = ispec.estimate_vmic(teff, logg, mh)

    except Exception:
        vmic = 1.0

    # Estimamos macroturbulencia
    try:

        vmac = ispec.estimate_vmac(teff, logg, mh)

    except Exception:
        vmac = 3.0

    # Definimos rotación inicial
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


# Escoger regiones a usar
def select_valid_regions( spectrum, line_regions):

    wave_min = np.min(spectrum["waveobs"])

    wave_max = np.max(spectrum["waveobs"])

    valid = (
        (line_regions["wave_base"] >= wave_min)
        & (line_regions["wave_top"] <= wave_max)
    )

    return line_regions[valid]


# Sacamos variables del espectro
def analyze_spectrum_with_ispec(
    wavelength_aa,
    flux,
    error=None,
    gaia_row=None,
    resources=None,
    config=ISPEC_CONFIG
):

    try:
        # Creamos espectro formato iSpec
        spectrum = build_ispec_spectrum(
            wavelength_aa=wavelength_aa,
            flux=flux,
            error=error,
            wave_min_nm=config["wave_min_nm"],
            wave_max_nm=config["wave_max_nm"]
        )
    except Exception as e:
        print(f"Error contruyendo espectro en formato iSpec: {e}")
        return None

    # Normalizamos al continuo
    try:
        normalized = normalize_spectrum_for_ispec(
            spectrum,
            resolution=config["resolution"]
        )
    except Exception as e:
            print("Error transformando al continuo")
            return None    

    # Definimos parámetros iniciales
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

    # Cargamos regiones válidas para iSpec
    line_regions = select_valid_regions( normalized, resources["line_regions"])

    # Definimos segmentos alrededor de la región
    segments = ispec.create_segments_around_lines( line_regions, margin=0.25)

    # Nos quedamos con las regiones importantes
    wavefilter = ispec.create_wavelength_filter(normalized, regions=segments)

    normalized_filtered = normalized[wavefilter]

    # Definimos el continuo con valor de referencia 1
    try:
        continuum_model = ispec.fit_continuum(
            normalized_filtered,
            fixed_value=1.0,
            model="Fixed value"
        )
    except Exception as e:
        print("Error en modelo continuo")
        return None
    
    # Parámetros a definir
    to_adjust_params = ["teff", "logg", "MH"]

    # No ajustamos las siguientes variables
    free_abundances = None
    linelist_free_loggf = None

    # Definimos variables faltantes
    initial_limb_darkening_coeff = 0.6 # el borde emite menos intensidad que el centro
    initial_R = config["resolution"]
    initial_vrad = 0.0 # no hay desplazamiento

    # Ajustamos las variables con iSpec
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

        normalized_filtered,
        continuum_model,
        resources["modeled_layers_pack"],
        resources["atomic_linelist"],
        resources["isotopes"],
        resources["solar_abundances"],
        free_abundances,
        linelist_free_loggf,
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
        to_adjust_params,
        segments=segments,
        linemasks=line_regions,
        enhance_abundances=False,
        use_errors=True,
        vmic_from_empirical_relation=True,
        vmac_from_empirical_relation=True,
        max_iterations=config["max_iterations"],
        verbose=0,
        tmp_dir=None,
        code=config["code"]
    )

    result = {
        "initial_teff": teff,
        "initial_logg": logg,
        "initial_mh": MH,
        "teff": params.get("teff", np.nan),
        "logg": params.get("logg", np.nan),
        "mh": params.get("MH", np.nan),

        "teff_err": errors.get("teff", np.nan),
        "logg_err": errors.get("logg", np.nan),
        "mh_err": errors.get("MH", np.nan),

        "status": status
    }

    return result

# Seleccionar los datos de Gaia segun el ID
def get_gaia_row_by_id(
    gaia_df,
    source_id,
    id_column="source_id"
):

    # Buscar source_id
    match = gaia_df[
        gaia_df[id_column] == source_id
    ]

    if len(match) == 0:
        raise ValueError(
            f"No se encuentra {source_id}"
        )

    # Devolvemos la fila de datos
    row = match.iloc[0]

    return row

# Analizar el espectro real y predicho
def analyze_sample_real_vs_pred(
    y_real,
    y_pred,
    gaia_ids,
    gaia_df,
    wavelength_aa,
    resources,
    sample_size=10,
    random_seed=23,
    id_column="source_id",
    min_teff=2500,
    max_teff=8000
):
    progress_bar = tqdm(
        total=sample_size,
        desc="Espectros analizados",
        unit="cuerpos estelares"
    )
    rng = np.random.default_rng(random_seed)

    random_indexes = rng.permutation(
        len(y_real)
    )

    results = []

    for i in random_indexes:
        # Una vez tenemos tantos resultados como el tamaño pedido, paramos
        if len(results) >= sample_size:
            break

        gaia_id = gaia_ids[i]

        try:
            gaia_row = get_gaia_row_by_id(
                gaia_df=gaia_df,
                source_id=gaia_id,
                id_column=id_column
            )
        except Exception as e:
            print(f"Índice {i} no encontrado gaia_id={gaia_id}")
            continue

        teff_gaia = gaia_row["teff_gspphot"]
        logg_gaia = gaia_row["logg_gspphot"]
        mh_gaia = gaia_row["mh_gspphot"]

        gaia_parameters = np.asarray([teff_gaia, logg_gaia, mh_gaia], dtype=float)

        # Aseguramos que no haya valores no definidos
        if not np.all(np.isfinite(gaia_parameters)):
            continue

        # Comprobar que la temperatura esté en el intervalo válido
        if not (min_teff <= teff_gaia <= max_teff):
            continue

        try:
            real_estimation = analyze_spectrum_with_ispec(
                wavelength_aa=wavelength_aa,
                flux=y_real[i],
                gaia_row=gaia_row,
                resources=resources
            )
        except Exception as e:
            print("Error analizando espectro real")
            continue

        try:
            pred_estimation = analyze_spectrum_with_ispec(
                wavelength_aa=wavelength_aa,
                flux=y_pred[i],
                gaia_row=gaia_row,
                resources=resources
            )
        except Exception as e:
            print("Error analizando espectro predicho")
            continue


        delta_teff = pred_estimation["teff"] - real_estimation["teff"]

        delta_logg = pred_estimation["logg"] - real_estimation["logg"]

        delta_mh = pred_estimation["mh"] - real_estimation["mh"]

        row_result = {

            "test_index": i,
            "source_id": gaia_id,

            # Gaia
            "gaia_teff": teff_gaia,
            "gaia_logg": logg_gaia,
            "gaia_mh": mh_gaia,

            # iSpec SDSS real
            "real_teff": real_estimation["teff"],
            "real_logg": real_estimation["logg"],
            "real_mh": real_estimation["mh"],

            # iSpec SDSS predicho
            "pred_teff": pred_estimation["teff"],
            "pred_logg": pred_estimation["logg"],
            "pred_mh": pred_estimation["mh"],

            # Diferencias predicho - real
            "delta_teff": delta_teff,
            "delta_logg": delta_logg,
            "delta_mh": delta_mh,
        }

        progress_bar.update(1)
        results.append(row_result)

    # Definimos como Dataframe
    results_df = pd.DataFrame(results)
    progress_bar.close()

    return results_df

# Funcion para calcular error y graficar resultados
def analyze_ispec_errors(results_df):

    df = results_df.copy()

    # Errores absolutos
    df["abs_error_teff"] = np.abs(results_df["delta_teff"])
    df["abs_error_logg"] = np.abs(results_df["delta_logg"])
    df["abs_error_mh"] = np.abs(results_df["delta_mh"])

    # Errores relativos
    df["rel_error_teff"] = ( df["abs_error_teff"] / np.abs(df["real_teff"]) ) * 100
    df["rel_error_logg"] = ( df["abs_error_logg"] / np.abs(df["real_logg"]) ) * 100


    # Graficamos histogramas por variable
    plt.hist(
        df["rel_error_teff"].dropna(),
        bins=30
    )

    plt.axvline(
        df["rel_error_teff"].median(),
        linestyle="--",
        label=(
            "Mediana = "
            f"{df['rel_error_teff'].median():.2f}%"
        )
    )

    plt.xlabel("Error relativo en Teff [%]")
    plt.ylabel("Número de observaciones")
    plt.title("Error relativo de Teff")
    plt.legend()

    plt.show()

    plt.hist(
        df["rel_error_logg"].dropna(),
        bins=30
    )

    plt.axvline(
        df["rel_error_logg"].median(),
        linestyle="--",
        label=(
            "Mediana = "
            f"{df['rel_error_logg'].median():.2f}%"
        )
    )

    plt.xlabel("Error relativo en log(g) [%]")
    plt.ylabel("Número de observaciones")
    plt.title("Error relativo de log(g)")
    plt.legend()

    plt.show()

    plt.hist(
        df["abs_error_mh"].dropna(),
        bins=30
    )

    plt.axvline(
        df["abs_error_mh"].median(),
        linestyle="--",
        label=(
            "Mediana = "
            f"{df['abs_error_mh'].median():.3f} dex"
        )
    )

    plt.xlabel("Error absoluto M/H")
    plt.ylabel("Número de observaciones")
    plt.title("Error absoluto de M/H")
    plt.legend()

    plt.show()

    return df