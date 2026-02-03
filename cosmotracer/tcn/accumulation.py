import logging
import time

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)
from cosmotracer.tcn import calculate_xyz_scaling_factors


class DepthIntegrationError(Exception):
    """
    Throw this error in case depth cannot be properly integrated
    for transient concentrations.
    """
    pass


def calculate_depth_interval_concentration(
    z0,
    z1,
    exhumation_rate,
    production_rate=1.0,
    attenuation_length=160.0,      # g/cm²
    bulk_density=2.7,              # g/cm³
    halflife=np.inf,
    initial_concentration=0.0
):
    """
    Numerically stable calculation of concentration as a sample is exhumed
    from depth z0 to z1 at constant exhumation_rate.

    Returns concentration at z1.
    """
    # ensure arrays for vectorized ops
    z0 = np.asarray(z0)
    z1 = np.asarray(z1)
    e = np.asarray(exhumation_rate)
    p0 = production_rate

    # units: match the original units used in your code
    rho = bulk_density * 1e2**3   # g/m^3  (1 g/cm^3 = 1e6 g/m^3)
    att = attenuation_length * 1e2**2  # g/m^2 (1 g/cm^2 = 1e4 g/m^2)
    mu = rho / att

    lam = 0.0 if np.isinf(halflife) else np.log(2.0) / halflife

    # total travel time
    t1 = (z0 - z1) / e
    
    # if the nuclide is stable, we need a special treatment to avoid nans
    if lam == 0.0:
        C = initial_concentration + (p0/(mu*e))*(np.exp(-mu*z1)-np.exp(-mu*z0))
        return C

    # decayed initial concentration
    C0 = initial_concentration * np.exp(-lam * t1)

    denom = lam + mu * e

    # numerically stable evaluation:
    # C = C0 + (p0/denom) * ( exp(-mu*z1) - exp(-(lam/e + mu)*z0 + (lam/e)*z1) )
    # This form avoids exp(+large) entirely; both exponent arguments are negative for positive parameters.
    term1 = np.exp(-mu * z1)
    term2 = np.exp(-(lam / e + mu) * z0 + (lam / e) * z1)

    C = C0 + (p0 / denom) * (term1 - term2)
    return C

def _calculate_depth_interval_concentration(
    z0: float|np.ndarray, # column depth in m
    z1: float|np.ndarray, # column depth, we assume that z1 < z0 (the sample moves towards the surface)
    exhumation_rate: float|np.ndarray, # this value is greater than zero
    production_rate: float|np.ndarray = 1.,
    attenuation_length: float = 160.,
    bulk_density: float = 2.7,
    halflife: float = np.inf,
    initial_concentration: float|np.ndarray = 0.
) -> float|np.ndarray:
    """
    Calculate the concentration of a sample as it is exhumed towards the surface from z0 to z1
    at rate exhumation_rate. Assumes that the surface production rate at a single point does not
    vary with time.
    
    LEGACY. TODO: REMOVE IN FUTURE VERSION
    
    Parameters:
    -----------
        z0: float|np.ndarray
            Initial depth. Either a single float > 0. or an array of shape (n,).
        z1 float|np.ndarray
            Final depth. Either a single float >= 0. or an array of shape (n,).
        exhumation_rate: float|np.ndarray
            Rate at which the sample moves towards the surface, or rate at which sample moves
            from z0 to z0. Either a single float or an array of shape (n,).
        production rate: float|np.ndarray
            Surface production rate at each point in at/g/yr. Single float 
            or array of shape (n,). Default is 1 at/g/yr
        attenuation_length: float
            The attenuation length in g/cm^2. Default is 160 g/cm^2
        bulk_density: float
            The bulk density in g/cm^3. Default is 2.7 g/cm^3
        halflife: float
            The halflife of the CRN in question. Default assumes a stable nuclide.
        initial_concentration: float|np.ndarray
            The concentration that the sample has at z0. Default is 0 at/g. Must be single float
            or array of shape (n.).
    
    Returns:
    --------
        concentrations: float|np.ndarray
            A single float or an array of concentrations.
    """
    
    # Introduce some shorthand and convert units
    rho = bulk_density * 1e2**3 # g/m^3
    att = attenuation_length * 1e2**2  # g/m^2
    p0 = production_rate 
    e = exhumation_rate
    decay = np.log(2) / halflife
    mu = rho/att    
    
    # t0 = 0
    t1 = (z0-z1) / e
    
    # decay of initial concentration
    c0 = initial_concentration*np.exp(-decay*t1)
    frac = p0*np.exp(-decay*t1) / (decay+mu*e)
    
    # main term
    concentration = frac*(np.exp(t1*(decay+mu*e)-mu*z0) - np.exp(-z0*mu)) + c0 
    
    return concentration

def calculate_steady_state_erosion_multiple_pathways(
    concentration: float,
    bulk_density: float = 2.7,
    production_rates: np.ndarray = np.array([1.]),
    attenuation_lengths: np.ndarray = np.array([160.]),
    halflife: float = np.inf,
    max_erosion_rate: float = 1.,
    solver_tolerance: float = 1e-7
):
    """
    This function calculates the steady state erosion rate inferred from a concentration
    assumed to have contributions from n production components, such as spallation, muon etc.
    
    If more than 1 component is supplied, the equation is solved numerically. For 1 component only,
    an analytical solution is derived using calculate_steady_state_erosion(). 
    """
    
    # check: if we only have one prod rate, pass to calculate_steady_state_erosion() 
    if len(production_rates) != len(attenuation_lengths):
        raise Exception("production_rates does not match shape of attenuation_lengths")
    if len(production_rates) == 1:
        e = calculate_steady_state_erosion(
            concentration=concentration,
            bulk_density=bulk_density,
            production_rate=production_rates[0],
            attenuation_length=attenuation_lengths[0],
            halflife=halflife
        )
        return e 
    else:
        # we have to use scipy optimize to find a solution here!
        def _objfun(e, cin, rho, prods, atts, lambd):
            c = np.sum(prods/(lambd+rho*1e2*e/atts))
            return (c - cin)**2
        
        out = minimize_scalar(
            fun=_objfun,
            args=(
                concentration,
                bulk_density,
                production_rates,
                attenuation_lengths,
                np.log(2)/halflife
            ),
            bounds=(0, max_erosion_rate),
            method="Bounded",
            options={"xatol": solver_tolerance}
        )
        
        return out.x
    
def calculate_steady_state_erosion(
    concentration : np.ndarray|float,
    bulk_density : np.ndarray|float = 2.7,
    production_rate : np.ndarray|float = 1.,
    attenuation_length : np.ndarray|float = 160.,
    halflife : np.ndarray|float = np.inf
):
    """Returns erosion rate in m/yr"""
    e = (attenuation_length / bulk_density / 1e2) * (production_rate / concentration - np.log(2)/halflife)
    return e

def calculate_steady_state_concentration(
    exhumation_rate : float,
    bulk_density : float = 2.7,
    production_rate : float = 1.,
    attenuation_length : float = 160.,
    halflife : float = np.inf
):
    """
    Calculate the surface concetration for a sample exhumed at constant rate exhumation_rate.
    
    Parameters:
    -----------
        exhumation_rate : float
            Erosion rate in m/yr
        etc
    """
    lambd = np.log(2) / halflife
    conc = production_rate / (lambd + bulk_density*1e2*exhumation_rate/attenuation_length)
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
    northings : np.ndarray|None = None,
    eastings : np.ndarray|None = None,
    epsg : int|None = None,
    inheritance_concentration : np.ndarray|None = None,
    throw_integration_error : bool  = False,
    allow_cache : bool = False,
    depth_approximation: bool = False,
    _easy_production: bool = False
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
    ### This results in using much larger column depths for very long time steps. If the erosion
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
            raise DepthIntegrationError("Exhumation history does not allow integration over desired depth_integration")

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
    
    if _easy_production:
        scaling_factors = np.ones(z.shape)
    else:
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
    
    # to get even more accurate, calculate the inherited concentration that samples
    # might have as they arrive at their respective max modelled depths.
    # Use the depth interval function for this and assume constant exh and
    # surface production
    Psurf_init = prod*scaling_factors[-1,:] # upside down!!
    exh_init = exh[-1,:]
    zeta_init = coldep[-1,:]
    
    if depth_approximation:
        conc_out = calculate_depth_interval_concentration(
            z0=np.full_like(zeta_init, np.inf),
            z1=zeta_init,
            exhumation_rate=exh_init,
            production_rate=Psurf_init,
            attenuation_length=attenuation_length,
            bulk_density=bulk_density,
            halflife=halflife
        ) + c0 # just add inheritance, its a bit dirty but better than nothing
    else:
        conc_out = c0
    # Next up: start from the bottom / largest depths and integrate the concentration.
    # conc_out = c0
    
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
     
     cinit = calculate_depth_interval_concentration(
         z0=np.inf, z1=0, exhumation_rate=1e-3,
     )
     print(cinit)
     
     ctrans, prod = calculate_transient_concentration(
         exhumation_rates=exhum,
         dt=1000,
         surface_elevations=z,
         production_rate=PSLHL,
         northings=northing,
         eastings=easting,
         epsg=32611,
         allow_cache=True,
         depth_integration=1.
     )

     css = calculate_steady_state_concentration(
         exhumation_rate=exhum[0,:],
         production_rate=prod
     )
     print(cinit, ctrans, css)
                
            
            
                

