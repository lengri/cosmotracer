"""
This file contains wrappers or extensions of existing landlab functions.
I've written these to avoid unnecessary bloat in other landlab-related
functions.


"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from landlab import NodeStatus, RasterModelGrid
from landlab.utils import get_watershed_mask
from collections import defaultdict

def calculate_node_flow_distances(
    grid: RasterModelGrid,
    sink_node : int
):
    """
    This function returns the flow distance of all nodes to the selected sink/outlet node.
    If there is no flow path from a node to the outlet node, the value is set to np.nan
    
    Requires a to-one FlowAccumulator (D4, D8).
    """

    flow_distances = np.full(
        grid.number_of_nodes, np.nan
    )
    
    lin_ids = np.arange(0, grid.number_of_nodes, 1, dtype=int)
    watershed = get_watershed_mask(grid, sink_node)
    source_nodes = lin_ids[watershed]
    
    # flow_stack = grid.at_node["flow__upstream_node_order"]
    
    # Iterate over all core_nodes, all other nodes can be ignored.
    for source_index in source_nodes:
        
        _, d = calculate_flow_path_to_node(
            grid,
            source_node=source_index,
            sink_node=sink_node
        )
        
        flow_distances[source_index] = d
    
    return flow_distances

def calculate_flow_path_to_node(
    grid : RasterModelGrid,
    source_node : int,
    sink_node : int
) -> tuple[list, float]:
    
    current = source_node 
    distance = 0. 
    node_list = [source_node]
    
    flow_link_lengths = grid.length_of_d8[
        grid.at_node["flow__link_to_receiver_node"]
    ]

    while True:
        
        if current == sink_node:
            return (node_list, distance)

        next_node = grid.at_node["flow__receiver_node"][current]
        
        if current == next_node:
            return ([], np.nan) 
        
        distance += flow_link_lengths[current]
        
        node_list.append(next_node)
        
        current = next_node
        
def map_node_values_upstream(
    grid : RasterModelGrid,
    node_field_name : str,
    empty_value : float = 0.
):
    """
    Propagate node values upstream through a Landlab FlowAccumulator-like object.
    
    Parameters
    ----------

    grid : RasterGrid
    node_field_name : str
        The name of the node-defined field for which values should be propagated.
    empty_value : float
        The value at an empty field index. This should be a value the field of interest
        cannot assume under normal circumstances, like -1 when considering chi indices.
    
    Returns
    -------
    grid : RasterGrid
        With updated field.
    """

    # get channel nodes and values
    field = grid.at_node[node_field_name]
    
    receivers = grid.at_node["flow__receiver_node"]

    # Get upstream-to-downstream order
    stack = grid.at_node["flow__upstream_node_order"]

    # Propagate values upstream
    for i in stack:
        recv = receivers[i]
        if i != recv:
            if field[i] == empty_value:
                field[i] = field[recv]
        else:
            if not field[recv] == empty_value:
                field[i] = field[recv]

    return grid

def interpolate_field_values(
    grid: RasterModelGrid,
    node_field_name: str,
    y_coords: np.ndarray,
    x_coords: np.ndarray
) -> np.ndarray:
    
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
    
def cut_grid_to_watershed(
    grid: RasterModelGrid, 
    watershed_mask: np.ndarray
) -> RasterModelGrid:
    """
    Using the watershed mask, search for the bounding box around the
    watershed and use that to create a cut basin DEM with correct georeferencing.
    """
    
    """z = grid.at_node["topographic__elevation"].reshape(grid.shape)
    extent = [
        grid.x_of_node.min(),
        grid.x_of_node.max(),
        grid.y_of_node.min(),
        grid.y_of_node.max()
    ]"""
    
    # first. find the watershed extent...
    mask = watershed_mask.reshape(grid.shape)

    # we can just add up all rows and check the first and last row not zero
    # and do the same for all columns...
    sum_of_cols = np.sum(mask, axis=0) # this is for jmin and jmax
    sum_of_rows = np.sum(mask, axis=1) # this is for imin, imax
    
    i_is_zero = np.where(sum_of_rows>0)[0][[0,-1]]
    j_is_zero = np.where(sum_of_cols>0)[0][[0,-1]]
    i_min = i_is_zero[0]
    i_max = i_is_zero[1]+1
    j_min = j_is_zero[0]
    j_max = j_is_zero[1]+1    
    
    # bounding box is described by (i_min, j_min), (i_max+1, j_max+1)
    dx, dy = grid.dx, grid.dy # ascii actually does not take dx and dy separately...
    # xy_ll = grid.xy_of_lower_left
    
    if dx != dy:
        raise Exception("Cell side lengths dx, dy must be equal!")
    
    # The 0, 0 position in an ascii file corresponds to the lower left corner.
    # thus, we need to use i_min, j_min as our new ll corner!
    lin_ind = np.ravel_multi_index((i_min, j_min), grid.shape) 
    new_x_ll = grid.x_of_node[lin_ind]
    new_y_ll = grid.y_of_node[lin_ind]   
    
    #new_x_ll = xy_ll[0] + j_min*dx
    #new_y_ll = xy_ll[1] + (grid.shape[0]-1-i_max)*dy
    # print(f"New reference point: ({new_x_ll:.0f}, {new_y_ll:.0f})")
    
    zout = grid.at_node["topographic__elevation"].reshape(grid.shape)
    
    # check that we will not go out of bounds in either direction...
    if i_max >= grid.shape[0] and j_max < grid.shape[1]:
        zout = zout[i_min:,j_min:j_max]
    elif i_max < grid.shape[0] and j_max >= grid.shape[1]:
        zout = zout[i_min:i_max,j_min:]
    elif i_max >= grid.shape[0] and j_max >= grid.shape[1]:
        zout = zout[i_min:,j_min]
    else:
        zout = zout[i_min:i_max,j_min:j_max]
        
    # grid.at_node["topographic__elevation"] = zout

    
    grid_out = RasterModelGrid(
        shape=zout.shape, xy_spacing=(dx, dy), xy_of_lower_left=(new_x_ll, new_y_ll)
    )

    grid_out.add_zeros("topographic__elevation")
    grid_out.at_node["topographic__elevation"] = zout
    
    return grid_out
    
    
def select_watershed(
    grid: RasterModelGrid
) -> np.ndarray:

    A = grid.at_node["drainage_area"]
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
        np.log10(A.reshape(grid.shape)+1), 
        picker=True,
        origin="lower"
    )
    ax.set_title("Please click on the watershed outlet!")
    # image = ax.scatter(random, random, picker=True)
    fig.canvas.mpl_connect("pick_event", _onpick)
    plt.show()
    
    linear_index = np.ravel_multi_index((int(np.round(indices[0][1])), int(np.round(indices[0][0]))), grid.shape)
    grid.set_watershed_boundary_condition_outlet_id(
        linear_index, 
        grid.at_node["topographic__elevation"], 
        -999999.
    )
    # outlet_id = lu.get_watershed_outlet(grid, linear_index)
    # This needs to be the linear index?!
    watershed = get_watershed_mask(grid, linear_index)
    
    return watershed

def set_shoreline_nodes_as_outlets(
    grid: RasterModelGrid,
    water_value: float | int
):
    z = grid.at_node["topographic__elevation"].reshape(grid.shape)

    # Mask of zero elevation cells (sea level)
    zero_mask = (z == water_value)
    
    # Dilate the zero mask to find all neighbors of zero-elevation areas
    neighbor_mask = sp.ndimage.binary_dilation(zero_mask, structure=np.ones((3,3)))
    
    # Outlets are cells that are not 0, but touch 0-elevation cells
    outlets = (~zero_mask) & neighbor_mask

    # Also include the grid edge 
    edge_mask = np.zeros_like(z, dtype=bool)
    edge_mask[0,:] = edge_mask[-1,:] = edge_mask[:,0] = edge_mask[:,-1] = True
    outlets |= edge_mask & (~zero_mask)
    
    outlet_ids = np.arange(0, grid.number_of_nodes, 1)
    outlet_ids = outlet_ids[outlets.flatten()]

    node_status = np.full(grid.status_at_node.shape, NodeStatus.CLOSED)
    node_status[grid.at_node["topographic__elevation"]!=water_value] = NodeStatus.CORE
    node_status[outlet_ids] = NodeStatus.FIXED_VALUE
    grid.status_at_node = node_status
    
    return grid

def _build_channels(segments):
    # adjacency list: start -> list of (start,end) segments
    graph = defaultdict(list)
    starts = set()
    ends = set()
    
    for s, e in segments:
        graph[s].append((s, e))
        starts.add(s)
        ends.add(e)

    # channel heads = starts that never appear as ends
    heads = [s for s in starts if s not in ends]

    channels = []

    def dfs(seg, path):
        start, end = seg
        path.append(seg)

        if end not in graph:
            channels.append(path.copy())
        else:
            for nxt in graph[end]:
                dfs(nxt, path)

        path.pop()

    # run dfs from all heads
    for h in heads:
        for first in graph[h]:
            dfs(first, [])

    return channels
    
def _extend_channel_segments_to_outlet(
    channel_dict: dict
):
    # build channels from source to sink
    ext_channel_dict = {}    
    
    for outlet, outdict in channel_dict.items():
        
        channels = _build_channels(list(outdict.keys()))
        ext_outlet_dict = {}
        # go through each channel and stack the ids distances etc
        for channel in channels:
                            
            ids = outdict[channel[0]]["ids"]
            dists = outdict[channel[0]]["distances"]
            
            for seg in channel[1:]:
                ids = np.concatenate((ids, outdict[seg]["ids"][1:])) # avoid copying the first entry 2x
                dists = np.concatenate((dists, outdict[seg]["distances"][1:]))
                
            # add entry to extended dict
            ext_outlet_dict[(channel[0][0], channel[-1][1])] = {
                "ids": ids,
                "distances": dists
            }
        
        ext_channel_dict[outlet] = ext_outlet_dict
    
    return ext_channel_dict

if __name__ == "__main__":
    
    from landlab.components import FlowAccumulator
    from landlab.io import esri_ascii
    
    segs = [(8, 20), (20, 25), (20, 112), (25, 60), (112, 114), (112, 604)]
    out = _build_channels(segs)
    for c in out:
        print(c)