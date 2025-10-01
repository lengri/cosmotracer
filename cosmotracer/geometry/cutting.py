import numpy as np
from cosmotracer import CosmoLEM

def cut_create_grid(
        instance : CosmoLEM,
        x_of_center : float,
        y_of_center : float,
        radius : float = 5000.
    ):
        
        """
        Returns a new CosmoLEM instance with the topographic__elevation
        field cut to the desired coordinates.
        """
        
        # Filter the nodes by x and y position
        x_ll_exact = x_of_center - radius 
        y_ll_exact = y_of_center - radius
        
        x_ur_exact = x_of_center + radius 
        y_ur_exact = y_of_center + radius 
        
        x_include = np.logical_and(
            instance.x_of_node>x_ll_exact,
            instance.x_of_node<x_ur_exact
        )
        y_include = np.logical_and(
            instance.y_of_node>y_ll_exact,
            instance.y_of_node<y_ur_exact
        )
        id_include = np.logical_and(
            x_include, y_include
        )
        
        # Find shape
        nrows = len(np.unique(instance.y_of_node[id_include]))
        ncols = len(np.unique(instance.x_of_node[id_include]))        
        
        z_out = instance.at_node["topographic__elevation"][id_include]
        # TODO: Fix this for the southern hemisphere?
        xy_ll_node = (instance.x_of_node[id_include].min(), instance.y_of_node[id_include].min())
        dx = instance.dx 
        epsg = instance.epsg
        
        # TODO: We might need to add a shift to xy_ll_node of -dx*0.5
        
        # need to call self.__init__ here?
        grid_out = CosmoLEM(
            z_init=z_out.flatten(),
            n_sp=instance.n_sp,
            m_sp=instance.m_sp,
            shape=(nrows, ncols),
            xy_spacing=dx,
            xy_of_lower_left=xy_ll_node,
            identifier=instance.identifier,
            epsg=epsg
        )
        
        return grid_out