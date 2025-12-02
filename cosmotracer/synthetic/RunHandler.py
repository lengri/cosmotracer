import os
import time
from datetime import datetime, timedelta

import h5py
import numpy as np
from landlab.components import ChiFinder

from .synthetic import CosmoLEM


class ModelStartException(Exception):
    pass


class RunHandler:
    
    """
    Handles saving each time step of aa model run to hdf5, can
    also restart a model at the appropriate time and fill-in 
    saved exhumations etc. For accurate transient CTN concentration 
    calculations.
    """
    
    def __init__(
        self,
        shape: tuple,
        T_spinup: float,
        dt: float,
        dx: float,
        U: float,
        K_sp: float,
        n_sp: float,
        m_sp: float,
        sp_crit: float,
        K_diff: float,
        outlet: str,
        identifier: int|str = 0,
        xy_of_lower_left: tuple = (500_000, 0.)
    ):

        self.model = CosmoLEM(
            n_sp=n_sp,
            m_sp=m_sp,
            shape=shape,
            xy_spacing=dx,
            xy_of_lower_left=xy_of_lower_left,
            identifier=identifier,
            allow_cache=True
        )
        self.model.set_outlet_position(outlet)

        self.model.load_grid(
            T_total=T_spinup,
            U=U,
            K_sp=K_sp,
            dt=dt,
            K_diff=K_diff,
            sp_crit=sp_crit
        )
        
        self.K_sp_spinup = K_sp
        self.K_diff_spinup = K_diff
        self.sp_crit = sp_crit
        self.U_spinup = U
        self.dt_spinup = dt
        self.T_spinup = T_spinup
        
        # by default, track all core nodes...
        self.model.tracked_nodes = self.model.core_nodes
        
    def init_run(
        self,
        U: np.ndarray|float,
        K_sp: np.ndarray|float,
        dt: float,
        P_SLHL_dict: dict,
        att_dict: dict,
        dint_dict: dict,
        halflife: float,
        nuclide: str,
        savedir: str,
        tracked_nodes: np.ndarray|None = None
    ):
        # just save these parameters as attributes
        # main part: Checks for existing hdf5 files and
        # creates one if not there yet.
        # ALSO: will re-load last time step in existing file
        # in case last model run was terminated early!
        
        # do a few checks to see that U, K_sp have the correct format.
        u_is_array = hasattr(U, '__iter__')
        k_is_array = hasattr(K_sp, '__iter__')
        
        if (u_is_array and k_is_array) and (len(U) != len(K_sp)):
            raise ModelStartException(
                f"Incompatible U, K_sp formats with "
                f"{len(U)=} and {len(K_sp)=}"
            )
        # if both are not iterable, abort
        if not u_is_array and not k_is_array:
            raise ModelStartException(
                "Cannot determine model runtime if "
                "U, K_sp are not array-like!"
            )
        
        # cast both to np.ndarray if necessary
        if not u_is_array:
            U = np.ones(len(K_sp))*U
        if not k_is_array:
            K_sp = np.ones(len(U))*K_sp

        # set tracked nodes, or use the default (all core nodes)
        # these ids are in reference to total (flattened) grid size!
        if tracked_nodes is not None: self.model.tracked_nodes = tracked_nodes
        # quickly convert them to the ids corresponding to only core_nodes too:
        # Tracked nodes transformed to core_id arrays (the 2d arrays in this h5)
        id_tracked2core = []
        for id in self.model.tracked_nodes:
            for i, idc in enumerate(self.model.core_nodes):
                if id==idc:
                    id_tracked2core.append(i)
        
        self.tracked_nodes_coreref = id_tracked2core
        
        n_steps = len(U)
        T_max = n_steps*dt
        
        self.dt = dt
        self.T_max = T_max
        self.Uarray = U
        self.Karray = K_sp
        self.Pdict = P_SLHL_dict
        self.t12 = halflife
        self.nuc = nuclide
        self.attdict = att_dict
        self.dintdict = dint_dict
        
        # Create hdf5 key
        hkey = (
            f"model_U{self.U_spinup}_Ksp{self.K_sp_spinup}"
            f"_Kdiff{self.K_diff_spinup}_n{self.model.n_sp}"
            f"_m{self.model.m_sp}_dt{dt}_Tmax{T_max}.h5"
        )
        self.filepath = os.path.join(savedir, hkey)
        
        # start at the first array entry, 
        # this value will be changed if we load a model.
        self.i_start = 0 
        self.model_is_complete = False 
        
        if not os.path.exists(self.filepath):
            
            print(f"Creating file {self.filepath}")
            
            with h5py.File(self.filepath, "w") as f:
                
                # write attributes
                f.attrs["model_shape"] = self.model.shape
                f.attrs["outlet"] = self.model.outlet_position
                f.attrs["n_sp"] = self.model.n_sp 
                f.attrs["m_sp"] = self.model.m_sp
                f.attrs["U_spinup"] = self.U_spinup 
                f.attrs["K_sp_spinup"] = self.K_sp_spinup
                f.attrs["K_diff_spinup"] = self.K_diff_spinup
                f.attrs["sp_crit"] = self.sp_crit
                f.attrs["dt_spinup"] = self.dt_spinup
                f.attrs["T_spinup"] = self.T_spinup
                f.attrs["dx"] = self.model.xy_spacing
                f.attrs["epsg"] = self.model.epsg
                f.attrs["system_datetime_creation"] = str(datetime.now())
                f.attrs["system_datetime_laststep"] = str(datetime.now())
                f.attrs["model_U_laststep"] = 0.
                f.attrs["model_Ksp_laststep"] = 0.
                f.attrs["model_T_total"] = 0.
                
                self._create_1d_dataset(f, "model_t", n_steps)
                self._create_1d_dataset(f, "model_U", n_steps, data=self.Uarray)
                self._create_1d_dataset(f, "model_K", n_steps, data=self.Karray)
                for pathway in self.Pdict.keys(): 
                    self._create_1d_dataset(f, f"model_Peff_{pathway}", n_steps)
                
                # these are ids over entire model domain, 
                # should filter out to core_nodes?
                f.create_dataset(
                    "model_tracked_node_id", 
                    data=self.model.tracked_nodes, 
                    maxshape=self.model.tracked_nodes.shape, 
                    dtype="int64"
                )
                            
                f.create_dataset(
                    "model_tracked_node_id_corearray", 
                    data=id_tracked2core, 
                    maxshape=(len(id_tracked2core),), 
                    dtype="int64"
                )

                # We need to track all elevation values, 
                # otherwise we can't "restart" the model
                f.create_dataset(
                    "model_z", 
                    (1, self.model.number_of_nodes), 
                    maxshape=(1, self.model.number_of_nodes), 
                    dtype="float64"
                )
                f.create_dataset(
                    "model_exhum", 
                    (0, len(self.model.core_nodes)), 
                    maxshape=(n_steps, len(self.model.core_nodes)), 
                    dtype="float64"
                )
                
                # Also track chi to display upstream distance of nodes.
                f.create_dataset(
                    "model_chi",
                    (1, len(self.model.tracked_nodes)), 
                    maxshape=(1, len(self.model.tracked_nodes)), 
                    dtype="float64"
                )
                
                # These datasets are tracked throughout the entire model run
                for pathway in self.Pdict.keys(): 
                    self._create_2d_dataset(f, f"model_trans_conc_{pathway}", n_steps)
                    self._create_2d_dataset(f, f"model_tcn_scaling_{pathway}", n_steps)
                self._create_2d_dataset(f, "model_tracked_z", n_steps)
        
        else:
            # look if the file has information in it!
            with h5py.File(self.filepath, "r") as f:
                
                model_t = f["model_t"][:]

                if len(model_t) > 0:
                    
                    t_max = model_t[-1]

                    self.model.step_info["T_total"] = t_max
                    load_model = True if t_max % dt == 0 else False
                    
                    # also print a warning in case tmax is not compatible with dx
                    if t_max % dt != 0:
                        print(
                            f"WARNING: {t_max=} found in file {self.filepath} "
                            f"not compatible with {dt=}. "
                            "Unable to load past model run from file."
                        )
                    
                    # additional check: Is the model run actually complete? In that case,
                    # run_model should just immediately return None
                    if t_max == self.T_max:
                        self.model_is_complete = True
                else:
                    load_model = False
                
                # if tmax > 0 and compatible: load info...
                if load_model and not self.model_is_complete:
                    
                    # Define T_start!
                    # start at the index one after the last computed index (no -1 needed!)
                    self.i_start = len(model_t) 

                    # need to set tracked_exhumation, tracked_z, tracked_transient_concentrations
                    # model_exhum contains all exhumation rates of core nodes, need to filter to 
                    # tracked nodes only!
                    self.model.tracked_exhumation = f["model_exhum"][:,self.tracked_nodes_coreref]
                    self.model.tracked_z = f["model_tracked_z"][:,:]
                    for pathway in self.Pdict.keys():
                        setattr(
                            self.model, 
                            f"tracked_transient_concentration_{pathway}",
                            f[f"model_trans_conc_{pathway}"][:,:]
                        )
                    
                    # also set the elevation, make sure that these two are synced
                    self.model._z = f["model_z"][:]
                    self.model.at_node["topographic__elevation"] = self.model._z
                    
                    # No need to load the past U values.
                    
                    # assert that the supplied values in Uarray and Karray
                    u_equal = f.attrs["model_Ksp_laststep"] != self.Karray[self.i_start]
                    if u_equal:
                        raise ModelStartException(
                            f"Determined U starting point {self.Uarray[self.i_start]=} " 
                            f"does not equal {f.attrs['model_U_laststep']}"
                        )
                    
                    k_equal = f.attrs["model_U_laststep"] != self.Uarray[self.i_start]
                    if k_equal:
                        raise ModelStartException(
                            f"Determined K starting point {self.Karray[self.i_start]=} "
                            f"does not equal {f.attrs['model_K_laststep']}"
                        )

    def _create_1d_dataset(
        self,
        f: h5py.File,
        dataset_name,
        n_steps,
        data = None
    ):
        if data is None:
            f.create_dataset(
                dataset_name, 
                (0,), 
                maxshape=(n_steps,), 
                dtype="float64"
            )
        else:
            f
            f.create_dataset(
                dataset_name, 
                (n_steps,),
                data=data, 
                maxshape=(n_steps,), 
                dtype="float64"
            )
            
    def _create_2d_dataset(
        self,
        f: h5py.File,
        dataset_name,
        n_steps
    ):
        f.create_dataset(
            dataset_name,
            (0, len(self.model.tracked_nodes)),
            maxshape=(n_steps, len(self.model.tracked_nodes)),
            dtype="float64"
        )
    
    def run_model(
        self,
    ):
        
        if self.model_is_complete:
            return None
        
        start_time = time.time()
        
        for k, u in zip(self.Karray[self.i_start:], self.Uarray[self.i_start:]):
            
            current_time = time.time()
            
            print(
                f"Running model with parameters Ksp={k}, U={u}, Kdiff={self.K_diff_spinup}. "
                f"Runtime: {self.model.step_info["T_total"]}/{self.T_max}, "
                f"zmax: {self.model.at_node["topographic__elevation"].max():.2f} "
                f"Runtime: {str(timedelta(seconds=int(current_time-start_time)))}",
                end="\r"
            )

            self.model.run_one_step(
                U=u,
                K_sp=k,
                dt=self.dt,
                sp_crit=self.sp_crit,
                K_diff=self.K_diff_spinup
            )
            
            # calculate scaling factors for current elevations (for all core nodes!)
            self.model.calculate_TCN_xyz_scaling(
                nuclide=self.nuc
            )
            
            # for each pathway, calculate the transient concentration!
            for pathway in self.Pdict.keys():
            # we-re calculate some scaling factors, but that should be fine.
                self.model.calculate_TCN_transient_concentration(
                    depth_integration=self.dintdict[pathway],
                    nuclide=self.nuc,
                    production_rate_SLHL=self.Pdict[pathway],
                    halflife=self.t12,
                    production_pathway=pathway,
                    attenuation_length=self.attdict[pathway]
                )
        
            self._save_step_to_hdf5(
                K_step=k,
                U_step=u,
                T_step=self.model.step_info["T_total"]
            )
    
    def _save_step_to_hdf5(self, K_step, U_step, T_step):
        
        with h5py.File(self.filepath, "a") as f:
            
            f.attrs["system_datetime_laststep"] = str(datetime.now())
            f.attrs["model_Ksp_laststep"] = K_step 
            f.attrs["model_U_laststep"] = U_step
            f.attrs["model_T_total"] = T_step
            
            f["model_t"].resize(f["model_t"].shape[0]+1, axis=0)
            f["model_t"][-1] = self.model.step_info["T_total"]

            for pathway, pslhl in self.Pdict.items():
                
                # save concentrations
                f[f"model_trans_conc_{pathway}"] \
                    .resize(f[f"model_trans_conc_{pathway}"].shape[0]+1, axis=0)
                f[f"model_trans_conc_{pathway}"][-1,:] = getattr(
                    self.model,
                    f"tracked_transient_concentration_{pathway}"
                )[-1,:]
                
                # save scaling factors
                f[f"model_tcn_scaling_{pathway}"] \
                    .resize(f[f"model_tcn_scaling_{pathway}"].shape[0]+1, axis=0)
                f[f"model_tcn_scaling_{pathway}"][-1,:] = \
                    self.model.at_node[f"tcn__scaling_{pathway}"][self.model.tracked_nodes]
                    
                # Save an effective production rate
                f[f"model_Peff_{pathway}"] \
                    .resize(f[f"model_Peff_{pathway}"].shape[0]+1, axis=0)
                f[f"model_Peff_{pathway}"][-1] = \
                    np.mean(self.model.at_node[f"tcn__scaling_{pathway}"][self.model.core_nodes])*pslhl
            
            # always save all elevations to restart model
            f["model_z"][:] = self.model.at_node["topographic__elevation"] 
            
            # quickly calculate chi values:
            chi = ChiFinder(
                grid=self.model,
                reference_area=1.,
                reference_concavity=self.model.m_sp/self.model.n_sp,
                min_drainage_area=0.,
                clobber=True
            )
            chi.calculate_chi()
            f["model_chi"][:] = self.model.at_node["channel__chi_index"][self.model.tracked_nodes]

            f["model_tracked_z"].resize(f["model_tracked_z"].shape[0]+1, axis=0)
            f["model_tracked_z"][-1,:] = self.model.at_node["topographic__elevation"][self.model.tracked_nodes]
            f["model_exhum"].resize(f["model_exhum"].shape[0]+1, axis=0)
            f["model_exhum"][-1,:] = self.model.at_node["exhumation_rate"][self.model.core_nodes]
        
