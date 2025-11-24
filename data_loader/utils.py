import time 
import re
from functools import wraps


####################
# Conversions #
####################

hplanck = 6.62606868E-34  # Js  Plank consant
c = 299792458  # m/s   Speed of light
Ang = 1e-10  # m Angstrom
echarge = 1.6021733E-19  # C  electron charge
emass = 9.109e-31  # kg Electron rest mass
r0 = 2.8179403227e-15  # m classical electron radius = e^2/(4pi*e0*me*c^2)

UNIT_MAP = {
    "Angs": 1e-1,
    "pm": 1e-3,
    "nm": 1,
    "um": 1e3,
    "mm": 1e6,
    "cm": 1e7,
    "m": 1e9,
    "sec": 1.0,
    "s": 1.0,
    "millisecond": 1e-3,
    "ms": 1e-3,
    "percent": 1e-2,
    "ndeg": 1e-9,
    "deg": 1.0,
}

def time_it(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)  # Preserve function metadata
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

# ----------------------
# Helper functions
# ----------------------
def clean_key(key):
    """
    Converts strings like:
    'SCAN-X [Position] (float64,um)' -> 'scan_x'
    'Start point' -> 'start_point'
    """
    key = key.lower()
    key = re.sub(r'\[.*?\]|\(.*?\)', '', key)  # remove bracketed units
    key = re.sub(r'[^a-z0-9]+', '_', key)      # replace non-alphanum with _
    key = key.strip('_')
    return key


def inject_attrs(obj, d, scan):
    """
    Flatten dictionary entries as attributes on the object.
    """
    scan = clean_key(scan)
    for k, v in d.items():
        cleaned_k = clean_key(k)
        attr = '_'.join([scan,cleaned_k])
        if 'exposure_time' in attr and not hasattr(obj, attr):
            attr = 'exposure_time'
            setattr(obj, attr, v)

        if ('step' in attr or 'point' in attr) and 'size' not in attr and not hasattr(obj, attr):
            setattr(obj, attr, v)


def log_roi_params(func):
    """
    Decorator for class methods that logs function calls and parameters per ROI.
    Assumes all methods have `roi_name` as the first positional argument after `self`
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get roi_name (first arg after self)
        mod_kwargs = kwargs.copy()
        
        try:
            roi = args[0]
        except:
            roi = kwargs['roi']
            mod_kwargs.pop("roi")
            assert len(args) == 0 

        entry = {
            "function": func.__name__,
            "args": args[1:],  
            "kwargs": mod_kwargs
        }

        roi.log_params.append(entry)
    
        return func(*args, **kwargs)

    return wrapper


def convert_value_with_unit(val):

    if isinstance(val, (int, float)):
        return val
        
    val_str = str(val).strip()

    m = re.match(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z]+)?$", val_str)
    if not m:
        return val  
        
    num, unit = m.groups()
    num = float(num)
    if unit is None:
        return num
    return num * UNIT_MAP.get(unit, 1.0)
    

    
def parse_log(log_str):

    lines = log_str.strip().split("\n")
    i, n = 0, len(lines)

    metadata = {}
    scans = {}
    attributes = {}

    # -----------------------------------
    # 1. PARSE METADATA (top section)
    # -----------------------------------
    while i < n and not lines[i].startswith('---'):
        if ":" in lines[i]:
            key, val = lines[i].split(":", 1)
            metadata[key.strip()] = val.strip()
        i += 1

    # Skip separator
    while i < n and lines[i].startswith('---'):
        i += 1

    # -----------------------------------
    # 2. PARSE SCAN BLOCKS (SCAN-X, SCAN-Y)
    # -----------------------------------
    scan_index = 0

    while i < n:
        line = lines[i]

        # Stop at attributes section
        if line.startswith("Session logged attributes"):
            break

        # Detect a scan block
        if line.startswith("Type:"):
            scan_dict = {}
            scan_name = None

            while i < n and not lines[i].startswith('---'):
                line = lines[i]

                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()

                    # Convert Python list literal
                    if val.startswith("[") and val.endswith("]"):
                        try:
                            val = eval(val)
                        except:
                            pass
                    else:
                        # Try to convert numeric types with units
                        val = convert_value_with_unit(val)

                    scan_dict[key] = val

                    # Use Device as scan name
                    if key == "Device":
                        scan_name = val

                i += 1

            # Auto-name if no device
            if scan_name is None:
                scan_name = f"scan_{scan_index}"

            scans[scan_name] = scan_dict
            scan_index += 1

        # Skip separator lines
        while i < n and lines[i].startswith('---'):
            i += 1

    # -----------------------------------
    # 3. PARSE SESSION LOGGED ATTRIBUTES
    # -----------------------------------
    if i < n and lines[i].startswith("Session logged attributes"):
        i += 1

        # Column names
        columns = [c.strip() for c in lines[i].split(";")]
        i += 1

        # Values (numeric conversion)
        raw_values = [v.strip() for v in lines[i].split(";")]
        values = []

        for v in raw_values:
            try:
                if "." in v or "e" in v.lower():
                    values.append(float(v))
                else:
                    values.append(int(v))
            except:
                values.append(v)

        # Create a dictionary mapping
        attributes = dict(zip(columns, values))

    return metadata, scans, attributes