import time 
import functools

hplanck = 6.62606868E-34  # Js  Plank consant
c = 299792458  # m/s   Speed of light
Ang = 1e-10  # m Angstrom
echarge = 1.6021733E-19  # C  electron charge
emass = 9.109e-31  # kg Electron rest mass
r0 = 2.8179403227e-15  # m classical electron radius = e^2/(4pi*e0*me*c^2)

def time_it(func):
    """Decorator to measure execution time of a function."""
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Compute execution time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds")
        return result  # Return original function result
    return wrapper

def energy2wavelength_a(energy_kev: float) -> float:
    """
    Converts energy in keV to wavelength in A
    wavelength_a = energy2wave(energy_kev)
    lambda [A] = h*c/E = 12.3984 / E [keV]
    """
    
    # Electron Volts:
    E = 1000 * energy_kev * echarge

    # SI: E = hc/lambda
    lam = hplanck * c / E # in meters
    wavelength_a = lam / Ang # in angstroms

    return wavelength_a

def wavelength_a2energy(wavelength):
    """
    Converts wavelength in A to energy in keV
     Energy [keV] = h*c/L = 12.3984 / lambda [A]
    """

    # SI: E = hc/lambda
    lam = wavelength * Ang
    E = hplanck * c / lam

    # Electron Volts:
    Energy = E / echarge
    return Energy / 1000.0