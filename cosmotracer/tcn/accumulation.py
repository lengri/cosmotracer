import numpy as np
import os, logging, time
logger = logging.getLogger(__name__)
from cosmotracer.tcn import calculate_xyz_scaling_factors

class IntegrationError(Exception):
    """
    Throw this error in case depth cannot be properly integrated
    for transient concentrations.
    """
    pass

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
    nuclide : str = "He",
    production_pathway : str = "sp",
    northings : np.ndarray | None = None,
    eastings : np.ndarray | None = None,
    epsg : int | None = None,
    inheritance_concentration : np.ndarray | None = None,
    throw_integration_error : bool  = False,
    allow_cache : bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function approximates nuclide concentration of  m samples rising towards 
    the surface at variable velocities (exhumation rates). 
    
    Parameters:
    -----------
        exhumation_rates : np.ndarray
            (n, m) array of exhumation rates for samples at n time steps. Must be in m/yr.
            This function assumes that the array is sorted from old to young along its first axis.
            I.e., exhumation_rates[n,:] are the exumation rates present as the sample reaches 
            the surface.
        dt : float
            The time step size in years.
        surface_elevations : np.ndarray
            (n, m) array of m surface elevation values (in meters) at n time steps.
            This input is used to scale changes in production rate due to surface uplift through time.
            Assumes that elevations are sorted from old to young along the first axis. (Analogous to exhumation_rates)
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
        nuclide : str
            The isotope name. Needed to calculate the scaling factors according to Lifton et al. (2014).
            Default is "He" (3He).
        production_pathway : str
            Determines which lat, lon, z, scaling factors are used. Available pathways are sp, eth, th, totmu, nmu, pmu.
            Note that the production_rate needs to be scaled to the relevant pathway separately!
        northing : np.ndarray
            Optional input for easting (in m). If None, defaults to 0.
        easting : np.ndarray
            Optional input for northing (in m). If None, defaults to 0.
        epsg : int
            The epsg code. Assumes that it is a UTM coordinate system.
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
    
    # Some shorthand notations
    exh = np.flipud(exhumation_rates) # legacy: we have to reverse the first axis to accumulate concentrations...
    z = np.flipud(surface_elevations) # legacy: we have to reverse the first axis to accumulate concentrations...
    prod = production_rate
    lambd = np.log(2) / halflife
    c0 = inheritance_concentration
    mu = bulk_density / attenuation_length
    
    n, m = exh.shape
    
    pathway2id = {
        "sp": 0,
        "eth": 1,
        "th": 2,
        "totmu": 3,
        "nmu": 4,
        "pmu": 5
    }
    # Defaults for latitude, longitude
    if northings is None:
        northings = np.zeros(m)
    if eastings is None:
        eastings = np.ones(m)*500_000 # avoid bounds errors for utm
        
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
    
    # print("Ensuring depth integration")
    if len(where_integrated) == 0: 
        logger.info("Depth integration not reached, adding slices until total depth > depth_integration")
        # we haven't reached the depth integration for at least one sample
        # keep adding new rows to our z, exh, exh_per_t, exh_tot, t arrays
        # until we've reached depth_integrated for all samples.
        if throw_integration_error:
            logger.error(
                f"Function interrupted because `throw_integration_error` is {throw_integration_error}:\n"
                f"\tColumn depth for at least one sample is: {coldep[-1,:].min():.2f}\n"
                f"\tColumn depth is smaller than required `depth_integration` {depth_integration:.2f}\n"
            
            )
            raise IntegrationError("Exhumation history does not allow integration over desired depth_integration")

        reached_integration = False
        
        while not reached_integration:
            exh = np.vstack([exh, exh[-1,:]])
            exh_per_t = np.vstack([exh_per_t, exh_per_t[-1,:]])
            coldep = np.vstack([coldep, coldep[-1,:]+exh_per_t[-1,:]])
            z = np.vstack([z, z[-1,:]])
            
            if coldep[-1,:].min() >= depth_integration:
                reached_integration = True

            logger.info(
                "Added slice to exhumation history, dimensions are:\n"
                f"\texh: {exh.shape}\n"
                f"\texh_per_t: {exh_per_t.shape}\n"
                f"\tcoldep: {coldep.shape}\n"
                f"\tz: {z.shape}"
            )
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
            
        logger.info(
            "Cut exhumation history to new dimensions:\n"
            f"\texh: {exh.shape}\n"
            f"\texh_per_t: {exh_per_t.shape}\n"
            f"\tcoldep: {coldep.shape}\n"
            f"\tz: {z.shape}"
        )   
                       
    shape_time_end = time.time()
    
    # Calculate the scaling factors here. We make use of cached values 
    # if it is allowed.
    scaling_factors = np.zeros(z.shape)
    
    # print("Scaling factors")
    scaling_time_start = time.time()
    
    # loop over each row in z and calculat the scaling factors
    for i in range(0, z.shape[0]):
        
        out = calculate_xyz_scaling_factors(
            x=eastings,
            y=northings,
            z=z[i,:],
            epsg=epsg,
            nuclide=nuclide,
            verbose=False,
            allow_cache=allow_cache
        )
        scaling_factors[i,:] = out[:,pathway2id[production_pathway]]
        logger.debug(f"Calculated scaling factors for step {i} of {z.shape[0]}")
    
    logger.info("Successfully calculated scaling factors")
        
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
        
        logger.debug(
            f"Calculated concentrations for step {i}:\n"
            f"\tProduction rates: P0={np.round(P0,2)}, P1={np.round(P1,2)}\n"
            f"\tColumn depths: C0={np.round(C0,2)}, C1={np.round(C1,2)}\n"
            f"\tProduction in step: {np.round(production_in_dt,2)}\n"
            f"\tDecay fraction: {np.round(dt*lambd*conc_out,2)}\n"
        )
        
    logger.info("Successfully integrated concentrations")
    integrate_time_end = time.time()
    
    logger.info(
        f"Cutting time:, {shape_time_end-shape_time_start:.3f}\n"
        f"Scaling time:, {scaling_time_end-scaling_time_start:.3f}\n"
        f"Integration time:, {integrate_time_end-integrate_time_start:.3f}\n"
        f"Minmax concs:, {conc_out.min():.2f}, {conc_out.max():.2f}"
    )
    
    return conc_out, scaling_factors[0,:]*prod
        
if __name__ == "__main__":
    
    # test the transient concentration accumulation...
     exhum = np.ones((100, 2))*1e-3
     z = np.ones(exhum.shape)+500
     PSLHL = 116.
     northing = np.linspace(3924450.0, 3924450.0+50000, exhum.shape[1])
     easting = np.linspace(344730.0, 344730.0+50000, exhum.shape[1])
     
     calculate_transient_concentration(
         exhumation_rates=exhum,
         dt=1000,
         surface_elevations=z,
         production_rate=PSLHL,
         northings=northing,
         eastings=easting,
         epsg=32611,
         allow_cache=True,
         depth_integration=10.
     )
     
     sf = calculate_xyz_scaling_factors(
         x=easting,
         y=northing,
         z=z[-1,:],
         epsg=32611,
         nuclide="He",
         allow_cache=True
     )[:,0]
     
     ss_conc = calculate_steady_state_concentration(
         erosion_rate=exhum[0,:],
         production_rate=116.*sf
     )
     print(ss_conc)
     
     
     print(sf)
                
            
            
                

