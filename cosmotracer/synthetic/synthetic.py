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
from landlab import NodeStatus

import logging
logger = logging.getLogger(__name__)
# For saving files
from cosmotracer.utils.filing import (
    ModelCache,
    get_cachedir
)
from cosmotracer.tcn import calculate_xyz_scaling_factors
from cosmotracer.tcn.accumulation import (
    calculate_steady_state_concentration,
    calculate_transient_concentration
)

# Cache dir and file name. Change only if you know what you're doing...
class ParseError(Exception):
    pass

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
        xy_of_lower_left : bool = (500_000, 0.),
        allow_cache : bool = False,
        fail_on_overwrite : bool = True,
        identifier : int | str = 0,
        epsg : int | None = 32611
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
            epsg : int | None
                The EPSG code for the coordinate system. This has to be defined when tracking transient
                concentrations.

        """ 
        
        # Constants
        self.n_sp = n_sp
        self.m_sp = m_sp
        self._z = z_init
        self.xy_spacing = xy_spacing
        self.grid_shape = shape
        self.xy_ll = xy_of_lower_left
        self.identifier = identifier
        self.epsg = epsg

        # Dict for step run info
        self.step_info = {
            'T_total': 0.,
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
        
        # Topo shielding
        self.add_ones("tcn__topographic_shielding")
        
        # Concentrations
        self.add_zeros("tcn__nuclide_concentration_sp")
        self.add_zeros("tcn__nuclide_concentration_eth")
        self.add_zeros("tcn__nuclide_concentration_th")
        self.add_zeros("tcn__nuclide_concentration_totmu")
        self.add_zeros("tcn__nuclide_concentration_nmu")
        self.add_zeros("tcn__nuclide_concentration_pmu")
        
        # The exhumation rates
        self.add_zeros("exhumation_rate")
        
        # Node tracking
        self.tracked_nodes = None
        self.tracked_exhumation = None
        self.tracked_z = None
        self.tracked_transient_concentrations = None
        
        logger.info(
            f"Successfully initialised CosmoLEM instance"
        )
    
    def set_outlet_position(
        self,
        location : str = "W"
    ):
        """
        Set the single open outlet node for the model.
        This can be one of the 4 cardinal directions, or
        NW/NE/SE/SW.
        """    
        
        n, m = self.shape
        # walk through all possible cases
        # NOTE: Remember that the origin is being treated as "lower left",
        # not upper left, as is normally the case when plotting with
        # matplotlib.
        match location:
            case "S":
                rc_id = (0, int(m/2))
            case "SE":
                rc_id = (0, m-1)
            case "E":
                rc_id = (int(n/2),m-1)
            case "NE":
                rc_id = (n-1, m-1)
            case "N":
                rc_id = (n-1, int(m/2))
            case "NW":
                rc_id = (n-1, 0)
            case "W":
                rc_id = (int(n/2), 0)
            case "SW":
                rc_id = (0, 0)
            case _:
                raise ParseError(f"Could not parse location argument {location}")
        
        # convert index to linear index
        outlet_id = np.ravel_multi_index(rc_id, self.shape)
        
        # set all boundary nodes to closed...
        lin_id_S = np.array([
            np.ravel_multi_index((0,j), self.shape) for j in range(0,m)
        ])
        lin_id_E = np.array([
            np.ravel_multi_index((i,m-1), self.shape) for i in range(0,n)
        ])
        lin_id_N = np.array([
            np.ravel_multi_index((n-1,j), self.shape) for j in range(0,m)
        ])
        lin_id_W = np.array([
            np.ravel_multi_index((i,0), self.shape) for i in range(0,n)
        ])
        
        self.status_at_node[lin_id_N] = NodeStatus.CLOSED
        self.status_at_node[lin_id_E] = NodeStatus.CLOSED
        self.status_at_node[lin_id_S] = NodeStatus.CLOSED
        self.status_at_node[lin_id_W] = NodeStatus.CLOSED
        
        self.status_at_node[outlet_id]= NodeStatus.FIXED_VALUE
        
    def run_one_step(
        self,
        U : float,
        K_sp : float,
        dt : float
    ) -> None:
        
        """
        Advances the LEM by one step. Information about this step is stored in a small dict,
        self.step_info. This dict stores the time step size ("dt"), the imposed uplift rate ("U"),
        the total elapsed model run time ('T_total'), and the imposed erodibility ("K_sp"). 
        The resulting exhumation rate calculated during the time step is stored in 
        self.at_node["exhumation_rate"].
        
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
        self.step_info['T_total'] += dt
        self.step_info["K_sp"] = K_sp

        # We have to create a new SPL in case
        # K is different...
        self.spl = StreamPowerEroder(
            self, K_sp=K_sp, m_sp=self.m_sp, n_sp=self.n_sp
        )

        z_old = self._z.copy()
        self._z[self.core_nodes] += U*dt
        self.fa.run_one_step() # update flow routing
        self.spl.run_one_step(dt=dt) # erode landscape
        
        exhum = (U*dt - (self._z - z_old)) / dt 
        self.at_node["exhumation_rate"] = exhum
        
        logger.info(
            f"Evolved landscape for one step:\n"
            f"\t{dt=}\n"
            f"\t{U=}\n"
            f"\t{K_sp=}\n"
            f"\tn_sp={self.n_sp}\n"
            f"\tm_sp={self.m_sp}\n"
            f"\tT_total={self.step_info['T_total']}"
        )
        
    def save_grid(
        self, 
        filepath : str = None,
        field_name : str = "topographic__elevation"
    ):
        """
        Save the current state of the grid/field to a file. If filepath is None,
        the grid will be saved to the Cache directory. Note that the file is only saved
        if allow_cache is set to True.
        
        In case of saving to cache, the file name will be self-recording, i.e. it will contain
        the grid dimension, K, U, n, m, and the total runtime. See `self.cachekey` for the full
        file name.
        
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
            logger.info(f"Saved field {field_name} to {filepath} at runtime {self.step_info['T_total']}")
        
        self._update_cachekey()
        
        self._cache.save_file(
            filekey=self.cachekey,
            array=self.at_node[field_name]
        )
        logger.info(
            f"Saved cache file {self.cachekey} in {get_cachedir()} at runtime {self.step_info['T_total']}"
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
        
        To check existing files in the cache, use `cosmotracer.utils.get_cachedir()`.
        
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
            logger.info(f"Loaded file from {filepath}")
        # If not: Lookup cache for existing entries
        else:
            self._update_cachekey(
                T_total=T_total, 
                U_step=U, 
                K_step=K_sp,
                dt_step=dt
            )
            logger.info(f"Updated cachekey to {self.cachekey}")
            self._z = self._cache.load_file(
                filekey=self.cachekey
            )
            logger.info("Attempted to load file from cache")
        
        # If we haven't assigned any array to self._z, something has gone wrong.
        if self._z is None:
            logger.error(
                f"No data for {field_name} could be loaded\n"
                f"If data was loaded from cache, no file with name {self.cachekey} was found."
            )
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
            logger.info(
                "Ran FlowAccumulator for one step with FlowDirectorD8 and DepressionFinderAndRouter"
            )
        else:
            logger.info("Did not run FlowAccumulator because field_name != 'topographic__elevation'")
                    
    
    def _init_grid(
        self,
        field_name : str = "topographic__elevation",
        seed : int | None = None
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
            
            if self.step_info['T_total'] > 0.:
                raise Exception("Could not initialise grid: T_total > 0 and no cached run found")

            if seed is not None:
                np.random.seed(seed)
                
            init_values = np.zeros(self.number_of_nodes)
            init_values[self.core_nodes] = np.random.uniform(0, 1, size=len(self.core_nodes))
            # save field values
            self._z = init_values
            
            logger.info(f"Constructed noisy initial topography {field_name}")      
             
        # self.set_closed_boundaries_at_grid_edges(True,True,False,True)
        self.add_zeros(
            field_name, at="node"
        )
        self.at_node[field_name] = self._z 
        
        logger.info(f"Set field values for field {field_name}")

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
        t_use = T_total if T_total is not None else self.step_info['T_total']
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
        
        logger.info(f"Updated cachekey to {self.cachekey} at runtime {self.step_info['T_total']}")
    
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
        nuclide : str = "He", 
        opt_args : dict = {}
    ):
        """
        Calculate elevation, latitude, and longitude scaling factors. 
        The resulting factors are stored in new at_node fields named
        `tcn__scaling_{pathway}`, where pathway is "sp" for spallogenic,
        "eth" for epi-thermal, "th" for thermal, "totmu" for total muon,
        "nmu" for negative muon, and "pmu" for positive muon production
        pathways.
        
        Parameters:
        -----------
            epsg : int
                The epsg code for the coordinate system. Used to interpret the
                self.x_of_node, self.y_of_node values.
            nuclide : str
                The TCN of interest. Must be "He", "Be", "C", or "Al". Default is "He".
            opt_args : dict
                Dict of optional arguments passed on to 
                cosmotracer.calculate_xyz_scaling_factors().
        """
        scaling = calculate_xyz_scaling_factors(
            x=self.x_of_node[self.core_nodes],
            y=self.y_of_node[self.core_nodes],
            z=self.at_node["topographic__elevation"][self.core_nodes],
            epsg=self.epsg,
            nuclide=nuclide,
            **opt_args
        )
        
        self.at_node["tcn__scaling_sp"][self.core_nodes] = scaling[:,0]
        self.at_node["tcn__scaling_eth"][self.core_nodes] = scaling[:,1]
        self.at_node["tcn__scaling_th"][self.core_nodes] = scaling[:,2]
        self.at_node["tcn__scaling_totmu"][self.core_nodes] = scaling[:,3]
        self.at_node["tcn__scaling_nmu"][self.core_nodes] = scaling[:,4]
        self.at_node["tcn__scaling_pmu"][self.core_nodes] = scaling[:,5]
        
        logger.info(f"Calculated scaling factors at runtime {self.step_info['T_total']}")
    
    def calculate_TCN_steady_state_concentration(
        self,
        bulk_density : float = 2.7,
        production_rate_SLHL : float = 1.,
        attenuation_length : float = 160.,
        halflife : float = np.inf,
        production_pathway : str = "sp"
    ):
        """
        Calculates the expected TCN concentration at each cell assuming steady-state
        erosion rates over the depth integration time scale. Erosion rate information
        is taken from self.at_node["exhumation_rate"].
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
            production_pathway : str
                The production pathway corresponding to either "sp" for spallogenic,
                "eth" for epi-thermal, "th" for thermal, "totmu" for total muon,
                "nmu" for negative muon, and "pmu" for positive muon production
                pathways. The results are always saved to the field
                "tcn__{production_pathway}_nuclide_concentration"
                
        """
        
        # NOTE: Assuming steady state erosion, this will cause minor errors in 
        # transient basins. Maybe I'll implement an external function for
        # transient calculations, but that would require storing each exhumation
        # and elevation time steps relevant for the transient concentration
        # calculations.
        
        # NOTE: For now we just use the spallogenic component for the total concentration.
        # in the future, it should be a sum over the individual production histories 
        # (sp, mu, eth, th).
        
        xyz_scaling = self.at_node[f"tcn__scaling_{production_pathway}"][self.core_nodes]
        topo_shielding = self.at_node["tcn__topographic_shielding"][self.core_nodes]
        prod = xyz_scaling*topo_shielding*production_rate_SLHL
        
        self.at_node[f"tcn__nuclide_concentration_{production_pathway}"][self.core_nodes] = calculate_steady_state_concentration(
            erosion_rate = self.at_node["exhumation_rate"][self.core_nodes],
            bulk_density=bulk_density,
            production_rate=prod,
            attenuation_length=attenuation_length,
            halflife=halflife
        )
        
        logger.info(f"Calculated {production_pathway} nuclide productions at runtime {self.step_info['T_total']}")

    def _track_exhumation_z(self):
        
        # NOTE: Splitting this off into its own function allows us to 
        # track exhumation and z even if we are not really using it.
        # we can then call calculate_TCN_transient_concentration after
        # user-defined time steps and only calculate these concentrations
        # every now and then (whenever relevant).
        
        # By default, calculate_TCN_transient_concentration calls _track_exhumation_z
        # anyways...
        # we need to have defined self.tracked_nodes
        if self.tracked_nodes is None:
            raise Exception("self.tracked_nodes must be a 1d array of ids.")
        
        # check if the arrays already exist. If not, create them now.
        if self.tracked_exhumation is None or self.tracked_z is None:
            self.tracked_exhumation = self.at_node["exhumation_rate"][self.tracked_nodes].reshape(1,len(self.tracked_nodes))
            self.tracked_z = self.at_node["topographic__elevation"][self.tracked_nodes].reshape(1,len(self.tracked_nodes))
    
        else:
            # NOTE: The stacking is such that the newest step is always at position [n] 
            # of the 2d array. This matches the order expected by 
            # calculate_transient_concentration.
            self.tracked_exhumation = np.vstack(
                (
                    self.tracked_exhumation,
                    self.at_node["exhumation_rate"][self.tracked_nodes]
                    
                )
            )
            self.tracked_z = np.vstack(
                (
                    self.tracked_z,
                    self.at_node["topographic__elevation"][self.tracked_nodes]
                    
                )
            )
            
    def calculate_TCN_transient_concentration(
        self,
        bulk_density : float = 2.7,
        production_rate_SLHL : float = 1.,
        attenuation_length : float = 160.,
        halflife : float = np.inf,
        depth_integration : float = 1.,
        nuclide : str = "He"
    ):
        """
        Calculates the transient nuclide concentrations for model-tracked nodes 
        during the simulation.

        This method computes the nuclide concentrations for nodes that have been
        registered using `self.track_nodes()`. At each simulation time step, it 
        calculates exhumation and elevation histories for the tracked nodes and 
        computes the corresponding transient nuclide concentrations using 
        `cosmotracer.tcn.calculate_transient_concentration()`.
        
        Coordinates (eastings, northings) and EPSG code are pulled from the model 
        state (`self.x_of_node`, `self.y_of_node`, `self.epsg`).

        Parameters:
        -----------
            bulk_density : float
                The average density of the overlying rock column in g/cm^3.
                Default is 2.7.
            production_rate_SLHL : float
                The sea-level high-latitude (SLHL) surface production rate in at/g/yr.
                Default is 1.
            attenuation_length : float
                The attenuation length in g/cm^2. Default is 160.
            halflife : float
                The half-life of the nuclide in years. Use np.inf for stable isotopes.
                Default is np.inf.
            depth_integration : float
                Target integration depth (in meters) over which nuclide accumulation 
                will be calculated. Determines how much of the exhumation history is 
                used in the calculation. Default is 1.
            nuclide : str
                The isotope for which the concentration is being calculated.
                Default is "He" (³He). Different isotopes have different scaling factors.
        """
        # we create/stack arrays with surface elevations and 
        # erosion rate sof  the points of interest.
        self._track_exhumation_z()
        
        logger.info(
            f"Tracked node exhumation and elevation for tracked nodes at runtime {self.step_info['T_total']}"
        )
        # with this data, we can calculate transient concentrations for the current time-step.
        tracked_concs, _ = calculate_transient_concentration(
            exhumation_rates=self.tracked_exhumation,
            dt=self.step_info["dt"],
            surface_elevations=self.tracked_z,
            depth_integration=depth_integration,
            production_rate=production_rate_SLHL,
            halflife=halflife,
            bulk_density=bulk_density,
            attenuation_length=attenuation_length,
            nuclide=nuclide,
            northings=self.y_of_node[self.tracked_nodes],
            eastings=self.x_of_node[self.tracked_nodes],
            epsg=self.epsg,
            allow_cache=True
        )
        logger.info(
            f"Calculated transient concentrations at runtime {self.step_info['T_total']}"
        )
        
        # Create a new tracked_transient_concentrations stack, if this is the first step
        if self.tracked_transient_concentrations is None:
            self.tracked_transient_concentrations = tracked_concs.reshape(1,len(self.tracked_nodes))
        # Or add it to the existing stack
        else:
            self.tracked_transient_concentrations = np.vstack(
                (
                    self.tracked_transient_concentrations,
                    tracked_concs
                    
                )
            )
    
    def track_nodes(
        self, 
        n : int = 100,
        seed : int = 1,
        weights : np.ndarray | None = None,
        valid_ids : np.ndarray | None = None
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
                shape as self.core_nodes or the array supplied to valid_ids. 
                Default is an unweighted sample.
            valid_ids : np._ndarray | None
                An array of valid ids to sample from. If left as None, it is set
                to self.core_nodes.
                
        """
        # set the seed
        np.random.seed(seed)
            
        # deal with the valid ids
        if valid_ids is None:
            ids = self.core_nodes
        else:
            ids = valid_ids
            
        # deal with weights
        if weights is None:
            weights = np.ones(len(ids))/len(ids)
        else:
            # ensure that weights are normalised
            weights /= np.sum(weights)
            
        # draw a random sample
        self.tracked_nodes = np.random.choice(
            ids, 
            replace=False,
            size=n,
            p=weights
        )
        
        logger.info(
            f"Started tracking nodes, {n=}"
        )
    
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
