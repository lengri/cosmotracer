import h5py
import os
import numpy as np

from .synthetic import CosmoLEM

import time
from datetime import timedelta

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
            dt=dt
        )
        # by default, track all core nodes...
        self.model.tracked_nodes = self.model.core_nodes
        
    def init_run(
        self,
        U: float,
        K_sp: float,
        dt: float,
        T_max: float,
        P_SLHL: float,
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
        
        # set tracked nodes, or use the default (all core nodes)
        if tracked_nodes is not None: self.model.tracked_nodes = tracked_nodes
        
        self.dt = dt
        self.T_max = T_max
        self.U = U
        self.K = K_sp
        self.P = P_SLHL
        self.t12 = halflife
        self.nuc = nuclide
        n_steps = int(np.ceil(T_max/dt))
        
        # Create hdf5 key
        hkey = f"model_U{U}_K{K_sp}_n{self.model.n_sp}_m{self.model.m_sp}_dt{dt}_Tmax{T_max}.hdf5"
        self.filepath = os.path.join(savedir, hkey)
        
        if not os.path.exists(self.filepath):
            print(f"Creating file {self.filepath}")
            with h5py.File(self.filepath, "w") as f:
                
                f.create_dataset(
                    "model_t", 
                    (0,), 
                    maxshape=(n_steps,), 
                    dtype="float64"
                )
                f.create_dataset(
                    "model_tracked_node_id", 
                    data=self.model.tracked_nodes, 
                    maxshape=self.model.tracked_nodes.shape, 
                    dtype="float64"
                )

                # We need to track all elevation values, otherwise we can't "restart" the model
                f.create_dataset(
                    "model_z", 
                    (1, self.model.number_of_nodes), 
                    maxshape=(1, self.model.number_of_nodes), 
                    dtype="float64"
                )
                self._create_grid_dataset(f, "model_trans_conc", n_steps)
                self._create_grid_dataset(f, "model_tcn_scaling", n_steps)
                self._create_grid_dataset(f, "model_exhum", n_steps)
                self._create_grid_dataset(f, "model_tracked_z", n_steps)
        
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
                            f"WARNING: {t_max=} found in file {self.filepath} not compatible with {dt=}."
                            "Unable to load past model run from file."
                        )
                else:
                    load_model = False
                
                # if tmax > 0 and compatible: load info...
                if load_model:

                    # need to set tracked_exhumation, tracked_z, tracked_transient_concentrations
                    self.model.tracked_exhumation = f["model_exhum"][:,:]
                    self.model.tracked_z = f["model_tracked_z"][:,:]
                    self.model.tracked_transient_concentration = f["model_trans_conc"][:,:]
                    
                    # also set the elevation, make sure that these two are synced
                    self.model._z = f["model_z"][:]
                    self.model.at_node["topographic__elevation"] = self.model._z

    def _create_grid_dataset(
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
        
        start_time = time.time()
        
        while self.model.step_info["T_total"] < self.T_max:
            
            current_time = time.time()
            
            print(
                f"Running model with parameters Ksp={self.K}, U={self.U}. "
                f"Runtime: {self.model.step_info["T_total"]}/{self.T_max}, "
                f"zmax: {self.model.at_node["topographic__elevation"].max()} "
                f"Runtime: {str(timedelta(seconds=int(current_time-start_time)))}",
                end="\r"
            )
            
            self.model.run_one_spl_step(
                U=self.U,
                K_sp=self.K,
                dt=self.dt
            )

            self.model.calculate_TCN_transient_concentration(
                depth_integration=4.,
                nuclide=self.nuc,
                production_rate_SLHL=self.P,
                halflife=self.t12
            )
        
            self._save_step_to_hdf5()
    
    def _save_step_to_hdf5(self):
        
        with h5py.File(self.filepath, "a") as f:
            
            f["model_t"].resize(f["model_t"].shape[0]+1, axis=0)
            f["model_t"][-1] = self.model.step_info["T_total"]

            f["model_trans_conc"].resize(f["model_trans_conc"].shape[0]+1, axis=0)
            f["model_trans_conc"][-1,:] = self.model.tracked_transient_concentration[-1,:]

            f["model_tcn_scaling"].resize(f["model_tcn_scaling"].shape[0]+1, axis=0)
            f["model_tcn_scaling"][-1,:] = self.model.at_node["tcn__scaling_sp"][self.model.tracked_nodes]

            f["model_z"][:] = self.model.at_node["topographic__elevation"] # always save all elevations to restart model

            f["model_tracked_z"].resize(f["model_tracked_z"].shape[0]+1, axis=0)
            f["model_tracked_z"][-1,:] = self.model.at_node["topographic__elevation"][self.model.tracked_nodes]
            f["model_exhum"].resize(f["model_exhum"].shape[0]+1, axis=0)
            f["model_exhum"][-1,:] = self.model.at_node["exhumation_rate"][self.model.tracked_nodes]
        
