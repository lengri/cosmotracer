
import numpy as np
import pyLSD as lsd

from cosmotracer.utils import (
    SQLiteScalingCache, 
    convert_xy_to_latlon
)

def calculate_xyz_scaling_factors(
    x : np.ndarray,
    y : np.ndarray,
    z : np.ndarray,
    epsg : int,
    nuclide : str,
    verbose : bool = False,
    scaling_params : dict = {
        "stdatm": False,
        "age": 0,
        "w": -1
    },
    allow_cache : bool = True,
    round_level : int = 1
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
    
    # save the scaling parameters here as a 2d array:
    # It has n rows (for each datapoint) and 
    # 6 columns: 0 - sp, 1 - eth, 2 - th, 3 totmu,
    # 4 - nmu, 5 - pmu
    #
    scaling_output = np.zeros((len(x), 6))
    n = len(lat)
    
    # Connect to cache. Newly calculated values are added to cache
    # once the with... is exited
    with SQLiteScalingCache(
        allow_cache=allow_cache,
        round_level=round_level,
        nuclide_key=nuclide
    ) as cache:
    
        for i, (la, lo, zz) in enumerate(zip(lat, lon, z)):
            
            if verbose: print(f"Calculating {nuclide} production scaling: {i/n*100:.0f} %", end="\r")
            
            # check if the cache output is none...
            value_list = cache.get_cache_value(la, lo, zz)
            
            if value_list is None:
            
                out = lsd.apply_LSD_scaling_routine(
                    lat=la,
                    lon=lo,
                    alt=zz,
                    nuclide=_nuclide_dict[nuclide],
                    **scaling_params
                )
            
                scaling_output[i,0] = out[nuclide][0]
                scaling_output[i,1] = out["eth"][0]
                scaling_output[i,2] = out["th"][0]
                scaling_output[i,3] = out["muTotal"][0]
                scaling_output[i,4] = out["mn"][0]
                scaling_output[i,5] = out["mp"][0]
                
                # save value to cache
                cache.set_cache_value(
                    la, lo, zz, list(scaling_output[i,:])
                )
                
            else:
                
                scaling_output[i,:] = np.array(value_list)
        
    if verbose: print(f"Calculating {nuclide} production scaling: 100 %") 
    
    return scaling_output

if __name__ == "__main__": 
    
    import time
    
    XLLCENTER =  644730.0
    YLLCENTER = 2924450.0
    np.random.seed(1)
    # calculate random arrays for x, y, z
    args = dict(
        x = np.linspace(XLLCENTER, XLLCENTER+100000, 5),
        y = np.linspace(YLLCENTER, YLLCENTER+100000, 5),
        z = np.random.uniform(0, 7000, size=5),
        epsg=32611,
        nuclide="He",
        allow_cache=True,
        round_level=1
    )
    
    # Test with SQlite cache
    start1 = time.time()
    outputSQ = calculate_xyz_scaling_factors(
        **args
    )
    end1 = time.time()
    
    
    
    