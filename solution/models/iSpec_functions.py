import hashlib
import json
import os
import sys
import traceback
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from astropy.table import Table

# Configuramos iSpec con una ruta relativa a este fichero
ISPEC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "lib", "iSpec")
)

# La tabla de Gaia, también relativa a este fichero, para que no dependa del
# directorio desde el que arranque el kernel
GAIA_DATA_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "gaia_data.ecsv")
)

# Parámetros ya ajustados sobre los espectros reales de SDSS. El ajuste de un
# espectro real no depende del modelo, así que es el mismo para todos: guardarlo
# ahorra la mitad de los ajustes de iSpec a partir del primer modelo analizado.
REAL_PARAMS_CACHE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "ispec_real_params.csv")
)

# Parámetros ya ajustados sobre los espectros predichos. Este ajuste sí depende
# del modelo, pero no de la ejecución: guardarlo evita repetirlo al rehacer un
# análisis y deja que un análisis interrumpido continúe donde lo dejó.
PRED_PARAMS_CACHE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "ispec_pred_params.csv")
)

# Lo que devuelve 'analyze_spectrum_with_ispec', que es lo que se guarda
_RESULT_COLUMNS = [
    "initial_teff", "initial_logg", "initial_mh",
    "teff", "logg", "mh",
    "teff_err", "logg_err", "mh_err",
    "status"
]

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
    "atmosphere_dir": os.path.join(
        ISPEC_DIR, "input", "atmospheres",
        "ATLAS9.Castelli/"
        # "MARCS.GES/"
    ),

    "atomic_linelist_file": os.path.join(
        ISPEC_DIR, "input", "linelists", "transitions",
        "VALD.300_1100nm", "atomic_lines.tsv"
    ),

    "solar_abundances_file": os.path.join(
        ISPEC_DIR, "input", "abundances",
        "Grevesse.2007", "stdatom.dat"
    ),

    "isotopes_file": os.path.join(
        ISPEC_DIR, "input", "isotopes", "SPECTRUM.lst"
    ),

    "line_regions_file": os.path.join(
        ISPEC_DIR, "input", "regions",
        "47000_SPECTRUM",
        "spectrum_synth_good_for_params_all.txt"
    ),

    # Elimina líneas teóricamente débiles
    "minimum_theoretical_depth": 0.01,

    # Número máximo de iteraciones para ajustar espectros
    "max_iterations": 6,
}


# Lo ya leído de disco, para no releerlo en cada llamada: los recursos de iSpec
# suman unos 195 MB entre la malla de atmósferas y la lista de líneas, y la
# tabla de Gaia ocupa 27 MB. Las claves distinguen configuraciones y ficheros
# distintos, de modo que pedir otros sí los lee.
_ispec_resources_cache = {}
_gaia_table_cache = {}


# Función para cargar recursos
def load_ispec_resources(config=ISPEC_CONFIG):

    cache_key = json.dumps(config, sort_keys=True)

    if cache_key in _ispec_resources_cache:
        return _ispec_resources_cache[cache_key]

    # Sin el sintetizador compilado, ispec.model_spectrum genera espectros
    # sintéticos de ceros y el ajuste devuelve los parámetros iniciales de Gaia
    # intactos: el error real - predicho saldría exactamente 0 para todo objeto
    if config["code"] == "spectrum" and not ispec.is_spectrum_support_enabled():
        raise RuntimeError(
            "El sintetizador SPECTRUM de iSpec no está compilado "
            "(falta ispec/synthesizer.so). Ejecuta 'make spectrum' en "
            "lib/iSpec con el entorno del proyecto activo."
        )

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

    resources = {
        "modeled_layers_pack": modeled_layers_pack,
        "solar_abundances": solar_abundances,
        "isotopes": isotopes,
        "atomic_linelist": atomic_linelist,
        "line_regions": line_regions
    }

    _ispec_resources_cache[cache_key] = resources

    return resources


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

    # Si no pasamos error, se definen todos los valores del espectro a 1 para darles el mismo peso en el ajuste
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

    # Si la síntesis ha fallado, el espectro modelado queda a cero y los
    # parámetros devueltos son los iniciales sin ajustar: no es un resultado
    if (
        modeled_synth_spectrum is None
        or not np.any(modeled_synth_spectrum["flux"])
    ):
        print("Error: la síntesis ha devuelto un espectro nulo")
        return None

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
# Función para cargar la tabla de Gaia guardada en local
def load_gaia_table(path=GAIA_DATA_FILE):

    path = os.fspath(path)

    if path in _gaia_table_cache:
        return _gaia_table_cache[path]

    gaia_table = Table.read(path, format="ascii.ecsv")

    # El resto de funciones trabajan con el DataFrame, no con la tabla
    gaia_df = gaia_table.to_pandas()

    print(f"Tabla de Gaia cargada: {len(gaia_df)} objetos.")

    # El DataFrame se comparte entre quienes lo piden, así que se consulta pero
    # no se modifica en el sitio: quien necesite cambiarlo, que use '.copy()'
    _gaia_table_cache[path] = gaia_df

    return gaia_df


# Función para buscar una fila de Gaia por su identificador
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

# Clave que identifica un ajuste: cambia si cambia el espectro, la malla de
# longitudes de onda, los parámetros de Gaia que lo inicializan o la
# configuración de iSpec, de modo que un dato distinto nunca reutiliza el ajuste
def _spectrum_cache_key(wavelength_aa, flux, gaia_parameters, config):

    digest = hashlib.sha256()

    for array in (wavelength_aa, flux, gaia_parameters):
        digest.update(np.ascontiguousarray(array, dtype=float).tobytes())

    digest.update(json.dumps(config, sort_keys=True).encode())

    return digest.hexdigest()[:16]


# Función para leer los ajustes ya calculados, reales o predichos
def load_params_cache(path=REAL_PARAMS_CACHE_FILE):

    if not os.path.exists(path):
        return {}

    # 'source_id' se lee como entero: son identificadores de Gaia de hasta 19
    # dígitos, que en el float64 por defecto de pandas perderían precisión
    table = pd.read_csv(path, dtype={"source_id": "int64"})

    return {
        row["cache_key"]: {column: row[column] for column in _RESULT_COLUMNS}
        for _, row in table.iterrows()
    }


# Función para añadir un ajuste a la caché, según se calcula
def append_params(path, source_id, cache_key, result):

    row = {
        "source_id": source_id,
        "cache_key": cache_key,
        **{column: result[column] for column in _RESULT_COLUMNS}
    }

    # Se escribe fila a fila y no al final: un análisis completo son horas, y
    # así una interrupción conserva todo lo ajustado hasta ese momento
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False
    )


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
    max_teff=8000,
    real_cache_path=REAL_PARAMS_CACHE_FILE,
    pred_cache_path=PRED_PARAMS_CACHE_FILE,
    config=ISPEC_CONFIG
):

    # Ajustes de espectros reales de análisis anteriores, de este modelo o de
    # cualquier otro: el espectro real es el mismo para todos
    real_params_cache = (
        load_params_cache(real_cache_path) if real_cache_path else {}
    )

    # Ajustes de espectros predichos de análisis anteriores. Solo los reutiliza
    # el mismo modelo: la clave incluye el flujo, que cambia con la predicción
    pred_params_cache = (
        load_params_cache(pred_cache_path) if pred_cache_path else {}
    )

    reused_real = 0
    reused_pred = 0
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
        #if not (min_teff <= teff_gaia <= max_teff):
        #   continue

        # El ajuste del espectro real no depende del modelo, así que se
        # reutiliza el que ya se calculó en este u otro análisis
        cache_key = _spectrum_cache_key(
            wavelength_aa, y_real[i], gaia_parameters, config
        )

        real_estimation = real_params_cache.get(cache_key)

        if real_estimation is not None:
            reused_real += 1
        else:
            try:
                real_estimation = analyze_spectrum_with_ispec(
                    wavelength_aa=wavelength_aa,
                    flux=y_real[i],
                    gaia_row=gaia_row,
                    resources=resources,
                    config=config
                )
            except Exception as error:
                print(f"Error analizando el espectro real de {gaia_id}: {error}")
                continue

            # El ajuste devuelve None cuando no converge, sin lanzar excepción
            if real_estimation is None:
                print("Sin ajuste para el espectro real")
                continue

            real_params_cache[cache_key] = real_estimation

            if real_cache_path:
                append_params(
                    real_cache_path, gaia_id, cache_key, real_estimation
                )

        # El ajuste del espectro predicho depende del modelo, pero no de la
        # ejecución: si ya se calculó para esta misma predicción, se reutiliza
        pred_cache_key = _spectrum_cache_key(
            wavelength_aa, y_pred[i], gaia_parameters, config
        )

        pred_estimation = pred_params_cache.get(pred_cache_key)

        if pred_estimation is not None:
            reused_pred += 1
        else:
            try:
                pred_estimation = analyze_spectrum_with_ispec(
                    wavelength_aa=wavelength_aa,
                    flux=y_pred[i],
                    gaia_row=gaia_row,
                    resources=resources,
                    config=config
                )
            except Exception as error:
                print(
                    f"Error analizando el espectro predicho de {gaia_id}: {error}"
                )
                continue

            if pred_estimation is None:
                print("Sin ajuste para el espectro predicho")
                continue

            pred_params_cache[pred_cache_key] = pred_estimation

            if pred_cache_path:
                append_params(
                    pred_cache_path, gaia_id, pred_cache_key, pred_estimation
                )


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

    print(
        f"Espectros analizados: {len(results_df)} "
        f"| ajustes reutilizados: {reused_real} reales, {reused_pred} predichos"
    )

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