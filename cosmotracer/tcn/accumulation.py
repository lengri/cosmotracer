import numpy as np
from pyLSD import apply_LSD_scaling_routine
import time 
import ujson, os

def _get_cached_scaling_factor(
    cache : dict,
    lat : float,
    lon : float,
    alt : float,
    nuclide : float
):
    key = str((round(lat, 5), round(lon, 5), round(alt, 2), nuclide))
    if key in cache:
        return cache[key], key 
    else:
        return None, key

def _save_scaling_cache(
    cachedir : str,
    cache : dict
):
    with open(cachedir, "w", encoding="utf-8") as f:
        ujson.dump(cache, f)

def _load_scaling_cache(
    cachedir : dict,
):
    if os.path.exists(cachedir):
        with open(cachedir, "r") as f:
            try:
                cache = ujson.load(f)
            except EOFError:
                cache = {}
                print("WARNING: Pickle cache corrupted")
    else:
        cache = {}
    
    return cache    

def calculate_steady_state_erosion(
    concentration : np.ndarray | float,
    bulk_density : np.ndarray | float = 2.7,
    production_rate : np.ndarray | float = 1.,
    attenuation_length : np.ndarray | float = 160.,
    halflife : np.ndarray | float = np.inf
):
    e = (attenuation_length / bulk_density) * (production_rate / concentration - np.log(2)/halflife) / 1e2
    return e

def calculate_steady_state_concentration(
    erosion_rate : float,
    bulk_density : float = 2.7,
    production_rate : float = 1.,
    attenuation_length : float = 160.,
    halflife : float = np.inf
):
    lambd = np.log(2) / halflife
    conc = production_rate / (lambd + bulk_density*1e2*erosion_rate/attenuation_length)
    return conc

def calculate_transient_concentration(
    exhumation_rates : np.ndarray,
    dt : float,
    surface_elevations : np.ndarray,
    depth_integration : float = 2.,
    production_rate : float = 1.,
    halflife : float = np.inf,
    bulk_density : float = 2.7,
    attenuation_length : float = 160,
    nuclide : int = 3,
    # production_pathway : str = "sp", # TODO: Implement this!!!
    latitudes : np.ndarray | None = None,
    longitudes : np.ndarray | None = None,
    inheritance_concentration : np.ndarray | None = None,
    throw_integration_error : bool  = False,
    write_scaling_cache : bool = False
):
    """
    This function approximates nuclide concentration of  m samples rising towards 
    the surface at variable velocities (exhumation rates). 
    
    Parameters:
    -----------
        exhumation_rates : np.ndarray
            (n, m) array of exhumation rates for samples at n time steps. Must be in m/yr.
            This function assumes that the array is sorted from young to old along its first axis.
            I.e., exhumation_rates[0,:] are the exumation rates present as the sample reaches 
            the surface.
        dt : float
            The time step size in years.
        surface_elevations : np.ndarray
            (n, m) array of m surface elevation values (in meters) at n time steps.
            This input is used to scale changes in production rate due to surface uplift through time.
            Assumes that elevations are sorted from modern to old along the first axis.
        depth_integration : float
            Determines over which depth the accumulation of the target nuclide will be calculated.
            The actual total depth value will depend on the step size and exhumation rate, but the used
            integrated exhumation rate will be >= depth_integration.
            The number of steps needed to guarantee an integration over a depth >= depth_integration
            is the same for all samples and determined by the sample with the overall lowest exhumation rate. 
            It might be computationally more efficient to split samples with very high and very low exhumation 
            rates that require drastically different number of steps.
            If the depth_integration value cannot be reached for at least one sample, the array of exhumation_rates
            is expanded step by step backwards in time by assuming (1) constant exhumation in the past, (2) constant
            time step size, (3) constant surface_elevations. A warning is displayed if this is the case.
        production_rate : float
            The SLHL production rate (at/g/yr). Default is 1 at/g/yr for an arbitrary nuclide.
        halflife : float
            The halflife time in years. Default is np.inf (i.e., a stable isotope).
        bulk_density : float
            The average/bulk density of the overlying rock column. Default is 2.7.
        attenuation_length : float
            The attenuation length for the production pathway. Default is 160 g/cm^2.
        nuclide : int
            The mass number of the isotope. Needed to calculate the scaling factors according to Lifton et al. (2014).
            Default is 3 (3He).
        latitudes : np.ndarray
            Optional input for latitudes (in decimal degrees). If None, defaults to 90.0°.
        longitudes : np.ndarray
            Optional input for longitudes (in decimal degrees). If None, defaults to 0.0°.
        inheritance_concentration: np.ndarray
            Initial inherited concentration of the target nuclide in at/g. If None, defaults to 
            0 at/g.
        throw_integration_error : bool = False
            Whether to throw an error if exhumation histories have to be extrapolated to reach
            the desired depth_integration.
    """
    
    ### A CURIOUS NOTE: This algorithm has the interesting effect that larger dt can be better at 
    ### accurately calculating concentrations for fast erosion rates and shallow integration depths.
    ### This is because the depth integration criterion is fullfiled by rounding up the considered 
    ### column depth to the nearest value larger than the depth integration threshold.
    ### This has the effect of using much larger column depths for very long time steps. If the erosion
    ### is fast, concentration loss by radioactive decay can be almost completely ignored and thus,
    ### the benefit of using larger total depths outweights the inaccuracy in calculated radioactive decay.
    
    # Load or create the scaling cache, if desired
    # NOTE: We still create a cache to avoid repetition in calculating
    # scaling factors. The difference is in writing it to a file or not...
    load_cache0 = time.time()
    if write_scaling_cache:
        cachedir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "scaling_cache.json"
        )
        cache = _load_scaling_cache(cachedir)
    else:
        cache = {}
    load_cache1 = time.time()
    
    
    # Some shorthand notations
    exh = exhumation_rates
    z = surface_elevations
    lats = latitudes
    lons = longitudes
    prod = production_rate
    lambd = np.log(2) / halflife
    c0 = inheritance_concentration
    mu = bulk_density / attenuation_length
    
    _z2iso = {
        3 : "He",
        10 : "Be",
        26 : "Al",
        14 : "C"
    }
    
    n, m = exh.shape
    
    # Defaults for latitude, longitude
    if lats is None:
        lats = np.ones(m)*90.
    if lons is None:
        lons = np.zeros(m)
        
    # Defualts for inheritance
    if c0 is None:
        c0 = np.zeros(m)
        
    # Convert the exhumation rates to a total exhumation per time step array
    exh_per_t = exh*dt
    
    # Convert the exhumation each time step to column depth ("below surface")
    # where the newest entry is 0 (coldepth[0,:])
    coldep = np.zeros((n+1,m))
    coldep[1:,:] = np.cumsum(exh_per_t, axis=0)

    # check if we have reached depth_integration.
    min_dep_at_t = np.min(coldep, axis=1)

    where_integrated = np.where(min_dep_at_t>=depth_integration)[0]   
    shape_time_start = time.time()
    
    #id_print = np.argmin(coldep[-1,:])
    #print("coldep", coldep[:,id_print])
    #print("exh per dt", exh_per_t[:,id_print])
    #print("exh rate", exh[:,id_print])
    #print("dt", dt)
    #print("id", id_print)
    
    # print("Ensuring depth integration")
    if len(where_integrated) == 0: 

        # we haven't reached the depth integration for at least one sample
        # keep adding new rows to our z, exh, exh_per_t, exh_tot, t arrays
        # until we've reached depth_integrated for all samples.
        if throw_integration_error:
            raise Exception("Exhumation history does not allow integration over desired depth_integration")

        reached_integration = False
        
        while not reached_integration:
            exh = np.vstack([exh, exh[-1,:]])
            exh_per_t = np.vstack([exh_per_t, exh_per_t[-1,:]])
            coldep = np.vstack([coldep, coldep[-1,:]+exh_per_t[-1,:]])
            z = np.vstack([z, z[-1,:]])
            
            if coldep[-1,:].min() >= depth_integration:
                reached_integration = True
        
        # we know that we'll use all available rows to integrate the concentrations
        # since we've needed to add some.
        n, m = exh.shape 
        
    else:
        # we can cut down on the shape of exhumation, because we don't need the rows 
        # below where we've reached depth integration.
        n_cut = where_integrated[0]
        if n_cut <= n: # Don't do anything if n_cut happens to equal n + 1.
            exh = exh[:n_cut,:]
            exh_per_t = exh_per_t[:n_cut,:]
            coldep = coldep[:n_cut+1,:]
            z = z[:n_cut,:]
            
            n, m = exh.shape               
    shape_time_end = time.time()
    # Calculate the scaling factors here. A small optimisation is that existing pairs of
    # are not calculated again.
    scaling_factors = np.zeros(z.shape)
    
    # print("Scaling factors")
    scaling_time_start = time.time()
    n_scaling_factors = z.shape[0] * z.shape[1]
    n_cache_factors_used = 0
    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            sf, key = _get_cached_scaling_factor(
                cache, lats[j], lons[j], z[i,j], nuclide
            )
            if sf is None:
                out = apply_LSD_scaling_routine(
                    lat=lats[j], lon=lons[j], alt=z[i,j], nuclide=nuclide
                )
                sf = out[_z2iso[nuclide]][0]
                cache[key] = sf
                
            else:
                n_cache_factors_used += 1
            
            scaling_factors[i,j] = sf
                
    # Write the new cache to disc, if desired
    write_cache0 = time.time()
    # write cache if wanted and if at least some scaling factors are new...
    if write_scaling_cache and (n_cache_factors_used != n_scaling_factors):
        _save_scaling_cache(
            cachedir, cache
        )
    write_cache1 = time.time()
        
    # duplicate the modern entry for scaling factor so it matches the shape of coldepth
    scaling_factors = np.vstack([scaling_factors[0,:], scaling_factors])
    scaling_time_end = time.time()
    # Next up: start from the bottom / largest depths and integrate the concentration.
    conc_out = c0
    
    # Since density and att have units of grams and cm, we convert the coldep into cm as well.
    coldep *= 100
    integrate_time_start = time.time()
    # print("Integrating concentration")
    
    for i in list(range(1, n + 1))[::-1]:
        P0 = prod*scaling_factors[i,:]
        P1 = prod*scaling_factors[i-1,:]
        
        C0 = coldep[i,:]
        C1 = coldep[i-1,:]
        
        a_p = (P1 - P0) / dt 
        a_c = (C1 - C0) / dt
        b_p = P0
        b_c = C0
        
        integral = lambda x : (-np.exp(-mu * (a_c * x + b_c)) * (a_c * b_p * mu + a_p * (a_c * mu * x + 1))) / (a_c**2 * mu**2)
        production_in_dt = integral(dt) - integral(0)
        conc_out += production_in_dt
        conc_out -= dt*lambd*conc_out # decay fraction
    integrate_time_end = time.time()
    
    print("Load cache time:", load_cache1-load_cache0)
    print("Cutting time:", shape_time_end-shape_time_start)
    print("Scaling time:", scaling_time_end-scaling_time_start)
    print(f"\tScaling cache used: {n_cache_factors_used}/{n_scaling_factors}")
    print("Integration time:", integrate_time_end-integrate_time_start)
    print("Write cache time:", write_cache1-write_cache0)
    print("Minmax concs:", conc_out.min(), conc_out.max())
    
    return conc_out, scaling_factors[0,:]*prod
        
if __name__ == "__main__":
    
    pass
                
            
            
                

