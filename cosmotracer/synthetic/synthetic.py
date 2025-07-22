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
    A class for modelling landscape evolution using the Stream Power Law.
    It inherits functionalities from landlab's RasterModelGrid and makes use of
    StreamPowerEroder to compute fluvial incision.
    
    To study cosmogenic nuclide signals of landscapes, CosmoLEM also contains
    multiple functions related to calculating TCN production rates,
    scaling factors, and concentrations.

    Initial topographies can be loaded from preexisting txt files, or from a
    Cache automatically handled by cosmotracer.
    
    Writen by Lennart Grimm (2025)
    """
    
    def __init__(
        self,
        z_init : None | np.ndarray = None,
        n_sp : float = 1.,
        m_sp : float = 0.5,
        shape : tuple = (100, 100),
        xy_spacing : float = 10.,
        xy_of_lower_left : bool = (0., 0.),
        allow_cache : bool = False,
        fail_on_overwrite : bool = True,
        identifier : int | str = 0
    ) -> None:
        
        """
        Sets up a CosmoLEM instance.
        
        Parameters:
        -----------
            z_init : None | np.ndarray
                Initial topography. If None, z_init is generated as a noisy DEM
                with a left-facing outlet. Must be a 1d array compatible with 
                landlab's RasterModelGrid. After intialisation, it can be accessed
                via instance.at_node["topographic__elevation"].
            n_sp : float
                The n exponent of the stream power law. Default is 1.
            m_sp : float
                The m exponent of the stream power law. Default is 0.5.
            shape : tuple
                The shape of the instance. Default is (100, 100).
            xy_spacing : float
                The spacing of cells (m) in the instance. Default is 100 m.
            xy_of_lower_left : tuple
                Easting and Northing (x, y) coordinate tuple of the bottom left
                cell of the instance. This is important for calculating scaling
                factors for cosmogenic nuclide prodution rates. Default is (0., 0.).
            allow_cache : bool
                Option to create (or use an existing) cache to save time slices of the 
                LEM. If set to True, the topography can be saved to cache by typing 
                self.save_grid(). This creates a unique .h5 file that includes the model
                runtime in the file name in your Cache directory. Default is set to False.
            fail_on_overwrite : bool
                Additional caching option to raise an Exception when an existing Cache file would
                be overwritten by a new call of self.save_grid(). Default is True.
            identifier : int | str
                An additional identifier that can be added to the file name of the cached file.

        """ 
        
        # Constants
        self.n_sp = n_sp
        self.m_sp = m_sp
        self._z = z_init
        self.xy_spacing = xy_spacing
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
            xy_spacing=xy_spacing,
            xy_of_lower_left=xy_of_lower_left
        )
        self._init_grid(field_name="topographic__elevation")
        
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
        U : float,
        K_sp : float,
        dt : float
    ) -> None:
        
        """
        Advances the LEM by one step. Information about this step is stored in a small dict,
        self.step_info. This dict stores the time step size ("dt"), the imposed uplift rate ("U"),
        the total elapsed model run time ("T_total"), the imposed erodibility ("K_sp"), and the
        resulting exhumation rate calculated during the time step ("exhumation_rate").
        
        Parameters:
        -----------
            U : float
                The imposed uplift rate (m/yr).
            K_sp : float
                The imposed erodibility (units depend on m, n).
            dt : float
                The time step size (yr).
        
        """
        
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
        filepath : str = None,
        field_name : str = "topographic__elevation"
    ):
        """
        Save the current state of the grid/field to a file. If filepath is None,
        the grid will be saved to the Cache directory. Note that the file is only saved
        if allow_cache is set to True.
        
        Parameters:
        -----------
            filepath : str
                If not None, the field defined by field_name
                will be saved to a .txt file using np.savetxt.
            field_name : str
                Determines which field to save to file. Default is "topographic__elevation".
                
        """
        
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
        filepath : str | None = None,
        T_total : float | None = None,
        U : float | None = None,
        K_sp : float | None = None,
        dt : float | None = None,
        field_name : str = "topographic__elevation"
    ):
        """
        Load and add a field to the instance from a file or the Cache. If T_total, U, K_sp, and dt
        are left as their default values, this function will look into the self.step_info dict
        to set these values.
        
        Parameters:
        -----------
            filepath: str | None
                Path to the raw .txt file containing the field information. This has to be a 1d array
                compatible with RasterModelGrids linear indexing. If set to the default None,
                this function will load a file from Cache instead.
            T_total : float | None
                The total runtime of the model as stated in the Cache file name.
            U : float | None
                The imposed uplift rate of the model as stated in the Cache file name.
            K_sp : float | None
                The imposed erodibility of the model as stated in the Cache file name.
            dt : float | None
                The imposed time step of the model as stated in the Cache file name.
            field_name : str
                The field name corresponding to the data in the loaded file. Default is 
                "topographic__elevation".
        """
        
        # If user wants to load grid from txt file, do it here
        if filepath is not None:
            self._z = np.loadtxt(
                filepath
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
        
        self.at_node[field_name] = self._z
        
        # we need to create a new FA
        if field_name == "topographic__elevation":
            self.fa = FlowAccumulator(
                self, 
                flow_director="FlowDirectorD8",
                depression_finder="DepressionFinderAndRouter"
            )
            self.fa.run_one_step()        
    
    def _init_grid(
        self,
        field_name : str = "topographic__elevation"
    ):
        """
        Initiate the a new grid, either by loading from Cache or by creating a new random initial topography.
        
        Parameters:
        -----------
            field_name : str
                The name of the field to be initialised.
        """
        
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
        T_total : float | None = None,
        U_step : float | None = None,
        K_step : float | None = None,
        dt_step : float | None = None
    ):
        """
        Update the cachekey, i.e. the name of the file that is to be loaded from
        or saved to cache.
        
        Parameters:
        -----------
            T_total : float | None 
                Total model runtime.
            U_step : float | None 
                Imposed uplift rate.
            K_step : float | None 
                Imposed erodibility.
            dt_step : float | None 
                Imposed time step size.
                
        """
        
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
        """
        Calculate scaling factors related to topographic shielding effects.
        
        TODO: Implement scheme.
        """
        pass 
    
    def calculate_TCN_xyz_scaling(
        self, 
        epsg : int, 
        nuclide : str, 
        opt_args : dict = {}
    ):
        """
        Calculate elevation, latitude, and longitude scaling factors. 
        The resulting factors are stored in new at_node fields named
        "tcn__scaling_{pathway}", where pathway is "sp" for spallogenic,
        "eth" for epi-thermal, "th" for thermal, "totmu" for total muon,
        "nmu" for negative muon, and "pmu" for positive muon production
        pathways.
        
        Parameters:
        -----------
            epsg : int
                The epsg code for the coordinate system. Used to interpret the
                self.x_of_node, self.y_of_node values.
            nuclide : str
                The TCN of interest. Must be "He", "Be", "C", or "Al".
            opt_args : dict
                Dict of optional arguments passed on to 
                cosmotracer.calculate_xyz_scaling_factors().
        """
        scaling = calculate_xyz_scaling_factors(
            x=self.x_of_node[self.core_nodes],
            y=self.y_of_node[self.core_nodes],
            z=self.at_node["topographic__elevation"][self.core_nodes],
            epsg=epsg,
            nuclide=nuclide,
            **opt_args
        )
        
        self.at_node["tcn__scaling_sp"][self.core_nodes] = scaling[:,0]
        self.at_node["tcn__scaling_eth"][self.core_nodes] = scaling[:,1]
        self.at_node["tcn__scaling_th"][self.core_nodes] = scaling[:,2]
        self.at_node["tcn__scaling_totmu"][self.core_nodes] = scaling[:,3]
        self.at_node["tcn__scaling_nmu"][self.core_nodes] = scaling[:,4]
        self.at_node["tcn__scaling_pmu"][self.core_nodes] = scaling[:,5]
    
    def calculate_TCN_steady_state_concentration(
        self,
        bulk_density : float = 2.7,
        production_rate_SLHL : float = 1.,
        attenuation_length : float = 160.,
        halflife : float = np.inf
    ):
        """
        Calculates the expected TCN concentration at each cell assuming steady-state
        erosion rates over the depth integration time scale. Erosion rate information
        is taken from self.step_info("exhumation_rate").
        If you are interested in calculating concentrations for transient (changing)
        erosion rates, work with cosmotracer.tcn.calculate_transient_concentration().
        
        Parameters:
        -----------
            bulk_density : float
                Average density of the rock/soil column (g/cm^3). Default is 2.7 g/cm^3.
            production_rate_SLHL : float
                The TCN production rate at sea-level high-latitude (at/g/yr). Default is 1 at/g/yr.
            attenuation_length : float
                The attenuation length for the production pathway (g/cm^2). Default is 160 g/cm^2
            halflife : float
                The TCN half life time (yr). Default assumes a stable nuclide (np.inf).
                
        """
        
        # NOTE: Assuming steady state erosion, this will cause minor errors in 
        # transient basins. Maybe I'll implement an external function for
        # transient calculations, but that would require storing each exhumation
        # and elevation time steps relevant for the transient concentration
        # calculations.
        
        # NOTE: For now we just use the spallogenic component for the total concentration.
        # in the future, it should be a sum over the individual production histories 
        # (sp, mu, eth, th).
        
        xyz_scaling = self.at_node["tcn__scaling_sp"][self.core_nodes]
        topo_shielding = self.at_node["tcn__topographic_shielding"][self.core_nodes]
        prod = xyz_scaling*topo_shielding*production_rate_SLHL
        
        self.at_node["tcn__nuclide_concentration"][self.core_nodes] = calculate_steady_state_concentration(
            erosion_rate = self.step_info["exhumation_rate"][self.core_nodes],
            bulk_density=bulk_density,
            production_rate=prod,
            attenuation_length=attenuation_length,
            halflife=halflife
        )
    
    def track_n_points(
        self, 
        n : int = 100,
        seed : int = 1,
        weights : np.ndarray | None = None
    ):
        """
        Chooses n core nodes id for easy access and tracking "random samples"
        generated from the eroding landscape.
        
        Parameters:
        -----------
            n : int = 100
                The sample size. Default is 100.
            seed : int = 1
                The numpy random seed.
            weights : np.ndarray | None
                The weights associated with each possible id. Must have the same
                shape as self.core_nodes. Default is an unweighted sample.
                
        """
        # set the seed
        np.random.seed(seed)
        
        # deal with weights
        if weights is None:
            weights = np.ones(len(self.core_nodes))/len(self.core_nodes)
        else:
            # ensure that weights are normalised
            weights /= np.sum(weights)
            
        # draw a random sample
        self.tracked_nodes = np.random.choice(
            self.core_nodes, 
            replace=False,
            size=n,
            p=weights
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
