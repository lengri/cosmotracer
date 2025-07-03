import numpy as np

HE_MAGNETITE_PARAMETERS = dict(
    
    # Spallation parameters
    sp_attenuation=160,               # g/cm^2
    sp_attenuation_uncert_abs=10,     # g/cm^2
    sp_attenuation_uncert_rel=0.0625, # -
    P_sp_SLHL=116,                    # at/g/yr (Hofmann et al. 2021)
    P_sp_SLHL_uncert_abs=13,          # at/g/yr (Hofmann et al. 2021)
    P_sp_SLHL_uncert_rel=0.112,       # -
    
    # Muogenic parameters
    mu_attenuation=8780,              # g/cm^2
    mu_attenuation_uncert_abs=np.nan, # g/cm^2
    mu_attenuation_uncert_abs=np.nan, # g/cm^2
    mu_P_SLHL = np.nan,               # g/cm^2
    
    # Isotope parameters
    halflife=np.inf,                  # yrs
    
    # Mineral parameters
    density_magnetite=5.15            # g/cm^3
)