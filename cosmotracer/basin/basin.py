"""
This class containts the main class of the cosmotracer
package. It makes heavy use of landlab's RasterModelGrid,
but has many additional capabilities.
"""

from cosmotracer.utils.filing import export_field_to_ascii
from cosmotracer.utils.landlab_convenience import (
    cut_DEM_to_watershed,
    find_watershed
    
)
from landlab.io import esri_ascii
from landlab import RasterModelGrid
import os
from landlab.components import (
    FlowAccumulator,
    LakeMapperBarnes
)
import numpy as np

class Basin(RasterModelGrid):
    
    def __init__(
        self,
        filepath : str,
        epsg : int,
        is_catchment_dem : bool = False,
        nodata=-999999.
    ):
        
        self.epsg = epsg
        self.source = os.path.dirname(filepath)
        self.nodata = nodata

        # load the input ascii file. If it is a catchment_dem,
        # we can feed it directly into the ModelGrid. If it isn't,
        # we let the user select the outlet and cut it now before 
        # initialising the RasterModelGrid.
        
        with open(filepath) as fp:
            mg = esri_ascii.load(
                stream=fp, 
                name="topographic__elevation"
            )
            z = mg.at_node["topographic__elevation"]
        
        if not is_catchment_dem:
            
            # we need to calculate flow routing here
            fa = FlowAccumulator(
                mg,
                flow_director="FlowDirectorD8"
            )
            fa.run_one_step()
            
            # and since this is a raw DEM, we need to fill depressions
            lmb = LakeMapperBarnes(
                mg,
                method="D8",
                fill_flat=False,
                track_lakes=True,
                redirect_flow_steepest_descent=True,
                reaccumulate_flow=True,
                ignore_overfill=True
            )
            lmb.run_one_step()
            
            # now, let the user pick a watershed outlet and cut the 
            # DEM to that watershed.
            mask = find_watershed(mg)
            
            # set all elevation values outside the watershed mask to nodata
            z[mask==0] = nodata
            
            mg, z = cut_DEM_to_watershed(mg, mask) # this creates a new mg without fa!
            
        # If the dem is already a processed basin dem, we don't need to do anything except
        # calculate the flow routing.
        mg.set_nodata_nodes_to_closed(
            z, 
            nodata
        )
                
        # Initialise the RasterModelGrid itself
        lin_ind = np.ravel_multi_index((mg.shape[0]-1, 0), mg.shape)
        super().__init__(
            mg.shape, 
            xy_spacing=(mg.dx, mg.dy), 
            xy_of_lower_left=(
                mg.x_of_node[lin_ind],
                mg.y_of_node[lin_ind]
            )
        )   
        self.add_zeros("topographic__elevation", at="node")
        self.at_node["topographic__elevation"] = z[:]
        self.set_nodata_nodes_to_closed(z, nodata)
        
        # run the flow accumulator (again) like usual
        fa = FlowAccumulator(
            self,
            flow_director="FlowDirectorD8",
            depression_finder="DepressionFinderAndRouter"
        )
        fa.run_one_step()
        
        self.FlowAccumulator = fa
    
    def calculate_basin_ksn(
        min_channel_threshold=1e6,
        method : str = "slope-area",
        mn_ratio : float = 0.5,
    ):
        pass 
    
    def parse_sample_concentrations(
        sample_dict
    ):
        pass
    
    def parse_tcn_constants(
        tcn_dict
    ):
        pass