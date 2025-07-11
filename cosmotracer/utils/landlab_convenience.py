from landlab import RasterModelGrid
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from landlab.utils import get_watershed_mask

def map_values_upstream(
    mg : RasterModelGrid,
    node_field_name : str,
    empty_value : float = 0.
):
    """
    Propagate values upstream through a Landlab FlowAccumulator-like object.
    
    Parameters
    ----------

    mg : RasterGrid
    node_field_name : str
        The name of the field defined at nodes
    empty_value : float
        The value at an empty field index.
    
    Returns
    -------
    mg : RasterGrid
        With updated field.
    """

    # get channel nodes and values
    field = mg.at_node[node_field_name]
    
    receivers = mg.at_node["flow__receiver_node"]

    # Get upstream-to-downstream order
    stack = mg.at_node["flow__upstream_node_order"]

    # Propagate values upstream
    for i in stack:
        recv = receivers[i]
        if i != recv:
            if field[i] == empty_value:
                field[i] = field[recv]
        else:
            if not field[recv] == empty_value:
                field[i] = field[recv]

    return mg

def interpolate_field_values(
    grid,
    node_field_name,
    y_coords,
    x_coords
):
    
    field_data = grid.at_node[node_field_name].reshape(grid.shape)

    # Get the grid coordinates
    x = grid.x_of_node.reshape(grid.shape)[0,:]
    y = grid.y_of_node.reshape(grid.shape)[:,0,]

    # Create the interpolator
    interpolator = sp.interpolate.RegularGridInterpolator(
        (y, x), 
        field_data, 
        method='nearest'
    )

    # Interpolate at the specified points
    points = np.column_stack((y_coords, x_coords))
    interpolated_values = interpolator(points)

    return interpolated_values
    
def cut_DEM_to_watershed(mg, watershed_mask):
    """
    Using the watershed mask, search for the bounding box around the
    watershed and use that to create a cut basin DEM with correct georeferencing.
    """
    
    # first. find the watershed extent...
    mask = watershed_mask.reshape(mg.shape)
    
    # we can just add up all rows and check the first and last row not zero
    # and do the same for all columns...
    sum_of_cols = np.sum(mask, axis=0) # this is for jmin and jmax
    sum_of_rows = np.sum(mask, axis=1) # this is for imin, imax
    
    i_is_zero = np.where(sum_of_rows>0)[0][[0,-1]]
    j_is_zero = np.where(sum_of_cols>0)[0][[0,-1]]
    i_min = i_is_zero[0]
    i_max = i_is_zero[1]
    j_min = j_is_zero[0]
    j_max = j_is_zero[1]
    
    # bounding box is described by (i_min, j_min), (i_max+1, j_max+1)
    dx, dy = mg.dx, mg.dy # ascii actually does not take dx and dy separately...
    # xy_ll = mg.xy_of_lower_left
    
    lin_ind = np.ravel_multi_index((i_min, j_min), mg.shape)
    new_x_ll = mg.x_of_node[lin_ind]
    new_y_ll = mg.y_of_node[lin_ind]    
    
    #new_x_ll = xy_ll[0] + j_min*dx
    #new_y_ll = xy_ll[1] + (mg.shape[0]-1-i_max)*dy
    # print(f"New reference point: ({new_x_ll:.0f}, {new_y_ll:.0f})")
    
    zout = mg.at_node["topographic__elevation"].reshape(mg.shape)
    zout = zout[i_min:i_max+1,j_min:j_max] # Adding the +1 creates a buffer row sometimes, why??
    # mg.at_node["topographic__elevation"] = zout
    
    mg_out = RasterModelGrid(
        zout.shape, xy_spacing=(dx, dy), xy_of_lower_left=(new_x_ll, new_y_ll)
    )
    mg_out.add_zeros("topographic__elevation", at="node")
    mg_out.at_node["topographic__elevation"] = zout
    
    return mg_out
    
    
def find_watershed(mg):
    
    A = mg.at_node["drainage_area"]
    indices = []
    
    def _onpick(event):
        mouseevent = event.mouseevent
        x = mouseevent.xdata
        y = mouseevent.ydata
        ind = mouseevent.x
        indices.append((x,y))
        return ind
    
    fig, ax = plt.subplots()
    ax.imshow(
        np.log10(A.reshape(mg.shape)+1), 
        picker=True,
        origin="lower"
    )
    ax.set_title("Please click on the watershed outlet!")
    # image = ax.scatter(random, random, picker=True)
    fig.canvas.mpl_connect("pick_event", _onpick)
    plt.show()
    
    linear_index = np.ravel_multi_index((int(np.round(indices[0][1])), int(np.round(indices[0][0]))), mg.shape)
    mg.set_watershed_boundary_condition_outlet_id(
        linear_index, 
        mg.at_node["topographic__elevation"], 
        -999999.
    )
    # outlet_id = lu.get_watershed_outlet(mg, linear_index)
    # This needs to be the linear index?!
    watershed = get_watershed_mask(mg, linear_index)
    
    return watershed

if __name__ == "__main__":
    
    from landlab.io import esri_ascii 
    from landlab.components import FlowAccumulator
    
    with open(r"C:\Users\Lennart\OneDrive\Desktop\phd\Work\Sites\EasternSierras\Data\FIELDWORK\Basins\DEMs\B01-01.asc") as asc_file:
        mg = esri_ascii.load(
            stream=asc_file,
            at="node",
            name="topographic__elevation"
        )
    
    fa = FlowAccumulator(
        mg,
        flow_director="FlowDirectorD8", 
        depression_finder="DepressionFinderAndRouter"
    )
    fa.run_one_step()
    
    mask = find_watershed(mg)
    mg_out = cut_DEM_to_watershed(mg, mask)
    
    plt.imshow(
        mg_out.at_node["topographic__elevation"].reshape(mg_out.shape),
        vmin=0
    )
    plt.show()