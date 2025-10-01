"""
This class containts the main class of the cosmotracer
package. It makes heavy use of landlab's RasterModelGrid,
but has many additional capabilities.
"""

from cosmotracer.synthetic.synthetic import CosmoLEM
from cosmotracer.utils.filing import (
    export_watershed_to_gpkg
)
from cosmotracer.utils.wrappers import (
    cut_grid_to_watershed,
    select_watershed
    
)
from landlab.io import esri_ascii
from landlab import RasterModelGrid, NodeStatus
from landlab.components import (
    FlowAccumulator,
    LakeMapperBarnes
)
import numpy as np


import os, logging
logger = logging.getLogger(__name__)


class Basin(CosmoLEM):
    
    def __init__(
        self,
        filepath : str,
        epsg : int,
        is_catchment_dem : bool = False,
        nodata : float = -999999.
    ):
        
        self.epsg = epsg
        self.source = os.path.dirname(filepath)
        self.nodata = nodata

        # load the input ascii file. If it is a catchment_dem,
        # we can feed it directly into the super CosmoLEM. If it isn't,
        # we let the user select the outlet and cut it now before 
        # initialising the super CosmoLEM.
        
        with open(filepath) as fp:
            grid = esri_ascii.load(
                stream=fp, 
                name="topographic__elevation"
            )
            z = grid.at_node["topographic__elevation"]
            logger.info(f"Loaded DEM from {filepath}")
            
        if not is_catchment_dem:
            
            # we need to calculate flow routing here
            fa = FlowAccumulator(
                grid,
                flow_director="FlowDirectorD8"
            )
            fa.run_one_step()
            logger.info("Ran simple FlowAccumulator (no Depression routing)")
            
            # and since this is a raw DEM, we need to fill depressions
            lmb = LakeMapperBarnes(
                grid,
                method="D8",
                fill_flat=False,
                track_lakes=True,
                redirect_flow_steepest_descent=True,
                reaccumulate_flow=True,
                ignore_overfill=True
            )
            logger.debug(
                "Created LakeMapperBarnes with method='D8', fill_flat=False,"
                "track_lakes=True, redirect_flow_steepest_descent=True,"
                "reaccumulate_flow=True, ignore_overfill=True"
            )
            lmb.run_one_step()
            logger.info("Successfully run LakeMapperBarnes")
            
            # now, let the user pick a watershed outlet and cut the 
            # DEM to that watershed.
            logger.info("Asking user to select watershed")
            mask = select_watershed(grid)
            logger.info("User selected watershed")
            
            # set all elevation values outside the watershed mask to nodata
            z[mask==0] = nodata
            
            grid = cut_grid_to_watershed(grid, mask) # this creates a new grid without fa!
            
            logger.info("Successfully cut grid to watershed")
            
        # If the dem is already a processed catchment dem, we don't need to do anything except
        # calculate the flow routing.
        grid.set_nodata_nodes_to_closed(
            grid.at_node["topographic__elevation"], 
            nodata
        )
                
        # Initialise the RasterModelGrid itself
        lin_ind = np.ravel_multi_index((grid.shape[0]-1, 0), grid.shape)
        super().__init__(
            z_init=grid.at_node["topographic__elevation"],
            shape=grid.shape, 
            xy_spacing=(grid.dx, grid.dy), 
            xy_of_lower_left=(
                grid.x_of_node[lin_ind],
                grid.y_of_node[lin_ind]
            ),
            epsg=epsg
        )   
        logger.debug("Initialised cosmotracer.CosmoLEM instance")
        #self.add_zeros("topographic__elevation", at="node")
        #self.at_node["topographic__elevation"] = z[:]
        self.set_nodata_nodes_to_closed(grid.at_node["topographic__elevation"], nodata)
        
        # run the flow accumulator (again) like usual
        fa = FlowAccumulator(
            self,
            flow_director="FlowDirectorD8",
            depression_finder="DepressionFinderAndRouter"
        )
        fa.run_one_step()
        logger.info("Ran proper FlowAccumulator with DepressionFinderAndRouter")
        
        self.FlowAccumulator = fa
    
    def parse_sample_concentrations(
        sample_dict
    ):
        pass
    
    def parse_tcn_constants(
        tcn_dict
    ):
        pass
    
    def save_watershed_gpkg(
        self,
        filepath
    ):
        mask = (self.status_at_node == NodeStatus.CORE).reshape(self.shape).astype(int)
        export_watershed_to_gpkg(
            mask=mask, 
            xy_ll_corner=self.xy_of_lower_left, 
            cellsize=self.dx, 
            filepath=filepath,
            epsg=self.epsg
        )
    
    