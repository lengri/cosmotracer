import numpy as np
import pyLSD as lsd 
import os 

try:
    from utm2ll import convert_xy_to_latlon
except ImportError:
    from .utm2ll import convert_xy_to_latlon

def calculate_xyz_scaling_factors(
    x : np.ndarray,
    y : np.ndarray,
    z : np.ndarray,
    epsg : int,
    nuclide : str,
    verbose : bool = False,
    scaling_params = {
        "stdatm": False,
        "age": 0,
        "w": -1
    }
):
    """
    This function expects inputs in easting, northing, and elevation [m].
    All arrays should be 1d (just like they are stored in landlab)
    """
    
    _nuclide_dict = {
        "He": 3,
        "Be": 10,
        "C": 14,
        "Al": 26
    }

    # convert x and y to lat and lon
    lat, lon = convert_xy_to_latlon(x, y, epsg)
    
    # save the scaling parameters here...
    scaling_output = {
        "sp": np.zeros(x.shape),
        "eth": np.zeros(x.shape),
        "th": np.zeros(x.shape),
        "totmu": np.zeros(x.shape),
        "nmu": np.zeros(x.shape),
        "pmu": np.zeros(x.shape)
    }
    
    n = len(lat)
    for i, (la, lo, zz) in enumerate(zip(lat, lon, z)):
        if verbose: print(f"Calculating {nuclide} production scaling: {i/n*100:.0f} %", end="\r")
        out = lsd.apply_LSD_scaling_routine(
            lat=la,
            lon=lo,
            alt=zz,
            nuclide=_nuclide_dict[nuclide],
            **scaling_params
        )
        
        scaling_output["sp"][i] = out[nuclide]
        scaling_output["eth"][i] = out["eth"]
        scaling_output["th"][i] = out["th"]
        scaling_output["totmu"][i] = out["muTotal"]
        scaling_output["nmu"][i] = out["mn"]
        scaling_output["pmu"][i] = out["mp"]
        
    if verbose: print(f"Calculating {nuclide} production scaling: 100 %") 
       
    return scaling_output
        
    