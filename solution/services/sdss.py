from specutils import Spectrum
import astropy.units as u

def parse_to_si(wavelength_array, flux_array) -> Spectrum:
    native_wave = (10 ** wavelength_array * u.AA)
    native_flux = (flux_array * 1e-17) * u.Unit('erg / (cm2 s AA)')
    
    return Spectrum(
        spectral_axis=native_wave.to(u.nm), 
        flux=native_flux.to(u.Unit('W / (m2 nm)'))
    )