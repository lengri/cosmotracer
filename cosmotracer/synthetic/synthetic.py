import numpy as np
import riserfit as rf 
import pyLSD as lsd
from numba import njit
from riserfit import (
    DistributionFromInterpolator,
    distribution_from_sample
)
import scipy as sp

from landlab.components import (
    FlowAccumulator,
    StreamPowerEroder
)
from landlab import RasterModelGrid

import os, sys, platformdirs
from pathlib import Path

# For saving files
from cosmotracer.utils.filing import ModelCache
from cosmotracer.tcn import calculate_xyz_scaling_factors
from cosmotracer.tcn.accumulation import calculate_steady_state_concentration

# Cache dir and file name. Change only if you know what you're doing...
    
class CosmoLEM(RasterModelGrid):
    
    """
    A class for modelling landscape evolution using the Stream-Power-Law
    and tracking nuclide concentrations from that landscape.
    
    Based on landlab's RasterModelGrid
    """
    
    def __init__(
        self,
        z_init = None,
        n_sp : float = 1.,
        m_sp : float = 0.5,
        shape : tuple = (100, 100),
        dx : float = 10.,
        xy_of_lower_left : bool = (0., 0.),
        allow_cache : bool = False,
        fail_on_overwrite : bool = True,
        identifier : int = 0,
        field_name : str = "topographic__elevation"
    ):
        
        # Constants
        self.n_sp = n_sp
        self.m_sp = m_sp
        self._z = z_init
        self.xy_spacing = dx
        self.grid_shape = shape
        self.xy_ll = xy_of_lower_left
        self.identifier = identifier

        # Dict for step run info
        self.step_info = {
            "T_total": 0.,
            "dt": None,
            "U": None,
            "K_sp": None
        }
        
        # Set up the cache
        self._cache = ModelCache(
            allow_cache=allow_cache,
            fail_on_overwrite=fail_on_overwrite
        )
        
        # The initiated grid will always be a east-facing basin
        # unless the user supplies some array to z_init.
        super().__init__(
            shape=shape,
            xy_spacing=dx,
            xy_of_lower_left=xy_of_lower_left
        )
        self._init_grid(field_name=field_name)
        
        # create the cachekey
        self._update_cachekey()
        
        # Flow accumulator for running the model
        # DepressionFinderAndRouter will only be
        # relevant to spin up the model.
        self.fa = FlowAccumulator(
            self, 
            flow_director="FlowDirectorD8",
            depression_finder="DepressionFinderAndRouter"
        )
        self.fa.run_one_step()
        
        # initialise fields for cosmogenic nuclide stuff
        
        # Lat, lon, elev scaling factors
        self.add_zeros("tcn__scaling_sp")
        self.add_zeros("tcn__scaling_eth")
        self.add_zeros("tcn__scaling_th")
        self.add_zeros("tcn__scaling_totmu")
        self.add_zeros("tcn__scaling_nmu")
        self.add_zeros("tcn__scaling_pmu")
        self.add_zeros("tcn__concentration")
        
        # Topo shielding
        self.add_ones("tcn__topographic_shielding")
        
        # Concentrations
        self.add_zeros("tcn__nuclide_concentration")
        
    def run_one_step(
        self,
        U,
        K_sp,
        dt
    ):
        
        self.step_info["dt"] = dt
        self.step_info["U"] = U 
        self.step_info["T_total"] += dt
        self.step_info["K_sp"] = K_sp

        # We have to create a new SPL in case
        # K is different...
        self.spl = StreamPowerEroder(
            self, K_sp=K_sp, m_sp=self.m_sp, n_sp=self.n_sp
        )

        z_old = self._z.copy()
        
        self._z[self.core_nodes] += U*dt
        self.fa.run_one_step()
        self.spl.run_one_step(dt=dt)
        
        exhum = (U*dt - (self._z - z_old)) / dt 
        self.step_info["exhumation_rate"] = exhum
        
    def save_grid(
        self, 
        filepath = None,
        field_name = "topographic__elevation"
    ):
        
        if filepath is not None:
            np.savetxt(
                filepath, self.at_node[field_name]
            )
        
        self._update_cachekey()
        self._cache.save_file(
            filekey=self.cachekey,
            array=self.at_node[field_name]
        )
    
    def load_grid(
        self, 
        path = None,
        T_total = None,
        U : float = None,
        K_sp : float = None,
        dt : float = None
    ):
        
        # If user wants to load grid from txt file, do it here
        if path is not None:
            self._z = np.loadtxt(
                path
            )
        # If not: Lookup cache for existing entries
        else:
            self._update_cachekey(
                T_total=T_total, 
                U_step=U, 
                K_step=K_sp,
                dt_step=dt
            )
            self._z = self._cache.load_file(
                filekey=self.cachekey
            )
        
        # If we haven't assigned any array to self._z, something has gone wrong.
        if self._z is None:
            raise Exception("Could not load any model; self._z is None")
        
        self.at_node["topographic__elevation"] = self._z
        
        # we need to create a new FA
        self.fa = FlowAccumulator(
            self, 
            flow_director="FlowDirectorD8",
            depression_finder="DepressionFinderAndRouter"
        )
        self.fa.run_one_step()        
    
    def _init_grid(
        self,
        field_name="topographic__elevation"
    ):
        
        if self._z is None:
                
            # If we can't find a cached version,
            # create a new grid here. However, we throw an error if T_total is not zero.
            
            if self.step_info["T_total"] > 0.:
                raise Exception("Could not initialise grid: T_total > 0 and no cached run found")
        
            ns_grad=0.5
            ew_grad=0.1
            noise_mag=500
            # np.random.seed(1)

            shape = self.shape

            if self._z is None:
                z = np.zeros(shape)
                for i in range(1, shape[1]-1):
                    z[1:-1,i] += np.abs(ns_grad*np.linspace(-self.dx*shape[0]/2,self.dx*shape[0]/2,shape[0]-2))+ew_grad*i*self.dx

                noise = np.random.random((shape[0]-2, shape[1]-2))*noise_mag # crank up the noise            
                z[1:-1,1:-1] += noise

                z = z.flatten()
                
            # save elevations
            self._z = z 
        
        # self.set_closed_boundaries_at_grid_edges(True,True,False,True)
        self.add_zeros(
            field_name, at="node", units="m"
        )
        self.at_node[field_name] = self._z        

    def _update_cachekey(
        self, 
        T_total = None,
        U_step = None,
        K_step = None,
        dt_step = None
    ):
        
        # Use info in step_info or the directly supplied values
        t_use = T_total if T_total is not None else self.step_info["T_total"]
        u_use = U_step if U_step is not None else self.step_info["U"]
        k_use = K_step if K_step is not None else self.step_info["K_sp"]
        dt_use = dt_step if dt_step is not None else self.step_info["dt"]        
            
        cache_components = [
            f"dim{self.shape[0]}{self.shape[1]}",
            f"T{t_use}",
            f"dt{dt_use}",
            f"U{u_use}",
            f"K_sp{k_use}",
            f"n_sp{self.n_sp}",
            f"m_sp{self.m_sp}",
            f"key{self.identifier}"
        ]
        self.cachekey = "_".join(cache_components) + ".h5"  
    
    def calculate_TCN_topo_scaling(
        self
    ):
        pass 
    
    def calculate_TCN_xyz_scaling(
        self, epsg, nuclide, opt_args : dict = {}
    ):
        scaling = calculate_xyz_scaling_factors(
            x=self.x_of_node[self.core_nodes],
            y=self.y_of_node[self.core_nodes],
            z=self.at_node["topographic__elevation"][self.core_nodes],
            epsg=epsg,
            nuclide=nuclide,
            **opt_args
        )
        
        self.at_node["tcn__scaling_sp"][self.core_nodes] = scaling["sp"]
        self.at_node["tcn__scaling_eth"][self.core_nodes] = scaling["eth"]
        self.at_node["tcn__scaling_th"][self.core_nodes] = scaling["th"]
        self.at_node["tcn__scaling_totmu"][self.core_nodes] = scaling["totmu"]
        self.at_node["tcn__scaling_nmu"][self.core_nodes] = scaling["nmu"]
        self.at_node["tcn__scaling_pmu"][self.core_nodes] = scaling["pmu"]
    
    def calculate_TCN_steady_state_concentration(
        self,
        bulk_density : float = 2.7,
        production_rate_SLHL : float = 1.,
        attenuation_length : float = 160.,
        halflife : float = np.inf
    ):
        
        # NOTE: Assuming steady state erosion, this will cause minor errors in 
        # transient basins. Maybe I'll implement an external function for
        # transient calculations, but that would require storing each exhumation
        # and elevation time steps relevant for the transient concentration
        # calculations.
        
        # NOTE: For now we just use the spallogenic component for the total concentration.
        # in the future, it should be a sum over the individual production histories 
        # (sp, mu, eth, th).
        
        xyz_scaling = production_rate_SLHL*self.at_node["tcn__scaling_sp"][self.core_nodes]
        topo_shielding = self.at_node["tcn__topographic_shielding"][self.core_nodes]
        prod = xyz_scaling*topo_shielding*production_rate_SLHL
        self.at_node["tcn__nuclide_concentration"][self.core_nodes] = calculate_steady_state_concentration(
            erosion_rate = self.step_info["exhumation_rate"][self.core_nodes],
            bulk_density=bulk_density,
            production_rate=prod,
            attenuation_length=attenuation_length,
            halflife=halflife
        )

class GradientFactory():
    def __init__(self, shape):
        self.shape = shape
        
    def create_uniform(
        self,
        value
    ):
        return np.full(self.shape, value)

    def create_linear(
        self,
        value_pair,
    ):
        
        y0 = value_pair[0]
        y1 = value_pair[1]
        
        m = self.shape[1]
        e = (y1-y0)/(m-1)*np.arange(0,m,1)+y0
        
        return np.full(self.shape, e)

    def create_step(
        self,
        value_pair
    ):
        y0 = value_pair[0]
        y1 = value_pair[1]
        
        e = np.ones(self.shape)
        e[:,:int(self.shape[1]/2)] = y0
        e[:,int(self.shape[1]/2):] = y1 
        
        return e
    
def _is_numeric(value):
    if type(value) == np.ndarray: return False 
    try:
        float(value)
        return True
    except:
        return False


@njit
def gaussian_kernel_numba(x, mu, sigma):
    frac = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    exponent = -((x - mu)**2) / (2.0 * sigma**2)
    return frac * np.exp(exponent)

@njit
def kde_gaussian(x, xi, hi, wi):
    
    tot_sum = np.zeros(x.shape[0])
    for i in range(xi.shape[0]):
        mu = xi[i]
        h = hi[i]
        w = wi[i]
        tot_sum += w * gaussian_kernel_numba(x, mu, h)
    area = np.trapz(tot_sum, x)
    kde = tot_sum / area
    return kde
    
class SyntheticDistribution():
    
    def __init__(self, xi, dx, h_rule = lambda x : x*0.05):
        self.dx = dx
        self.h_rule = h_rule
        hi = h_rule(xi)
        self.min = max(xi.min()-hi.max()*5, 0)
        self.max = xi.max()+hi.max()*5
        self.x = np.arange(self.min, self.max+dx, dx)

        self.y = kde_gaussian(
            self.x,
            xi=xi,
            hi=hi,
            wi=np.ones(xi.shape)
        )
        
        self.pdf = sp.interpolate.interp1d(
            self.x, self.y, bounds_error=False,
            fill_value=(0, 0)
        )
    
    def draw_sample(self, size):
        
        self.sample = np.random.choice(
            self.x,
            size=size,
            replace=True,
            p = self.y/np.sum(self.y) 
        )
        return self.sample

    def calculate_sample_pdf(self):
        hi = self.h_rule(self.sample)
        self.sample_y = kde_gaussian(
            self.x,
            xi=self.sample,
            hi=hi,
            wi=np.ones(self.sample.shape)
        )
        self.sample_pdf = sp.interpolate.interp1d(
            self.x, self.sample_y, bounds_error=False,
            fill_value=(0, 0)
        )
    
    def calculate_sample_misfit(self):
        # calculate the sample kde
        self.mf = np.trapz(
            np.abs(self.y-self.sample_y),
            x=self.x
        )
        return self.mf
    
    def repeat_n_draws(self, n, size):
        
        self.repeat_mfs = np.zeros(n)
        
        for i in range(0, n):
            self.draw_sample(size=size)
            self.calculate_sample_pdf()
            self.repeat_mfs[i] = self.calculate_sample_misfit()
        
        return self.repeat_mfs
        

if __name__ == "__main__":

    vals = np.random.normal(1000, 50, 5000)
    
    dist = distribution_from_sample(
        vals, 10
    )
    
    sd = SyntheticDistribution(vals, 10)
    sd.draw_sample(200)
    sd.repeat_n_draws(n=10, size=200)
    print(sd.repeat_mfs)
    
    import matplotlib.pyplot as plt
    plt.plot(sd.x, sd.y)
    plt.plot(sd.x, sd.sample_y)
    plt.plot(dist.x, dist.density)
    plt.show()
