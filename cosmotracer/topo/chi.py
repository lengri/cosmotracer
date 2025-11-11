from scipy.optimize import minimize
import numpy as np
from landlab import RasterModelGrid
from landlab.components import ChannelProfiler

from cosmotracer.utils.wrappers import (
    _extend_channel_segments_to_outlet, 
    map_node_values_upstream
)

def _chi_objective_function(
    ksn,
    A: np.ndarray,
    z: np.ndarray,
    neighbor_nodes: dict,
    alpha: float = 1,
):
    curv = np.zeros(ksn.shape)
    for i in range(0, len(ksn)):
        curv[i] = len(neighbor_nodes[i])*ksn[i]
        for n in neighbor_nodes[i]:
            curv[i] -= ksn[n]
            
    z -= z[0]
    
    return np.sum((np.matmul(A,ksn)-z)**2)**2+alpha**2*np.sum(curv**2)**2

def build_inversion_equation(
    grid,
    channel_dict,
    every_n
):
    # filter the network to every_n, make sure to keep segment starts and endpoints.
    if every_n > 1: 
        for outlet, outdict in channel_dict.items():
            for segment, segdict in outdict.items():
                ids = np.concatenate(
                    (
                        [0, 1], 
                        np.arange(1+every_n, len(segdict["ids"]), every_n), 
                    )
                )
                if ids[-1] != len(segdict["ids"])-1:
                    ids = np.concatenate((ids, [len(segdict["ids"])-1]))
                channel_dict[outlet][segment]["ids"] = segdict["ids"][ids]
                channel_dict[outlet][segment]["distances"] = segdict["distances"][ids]
    
    _flat_id_list = []
    for outlet, outdict in channel_dict.items():
            for segment, segdict in outdict.items():
                _flat_id_list = _flat_id_list + list(segdict["ids"]) # will contain some double entries
                
    # remove douplicates
    flat_id_list = []
    for id in _flat_id_list:
        if id not in flat_id_list:
            flat_id_list.append(id)
    flat_id_list.pop(0) # remove outlet

    # trace back the path down to the outlet node for each node
    gridid_path_nodes = {}
    Aid_node = {}
    rev_Aid_node = {}
    gridid_node_neighbors = {}
    Aid_node_neighbors = {}
    
    Aid_col = 0
    
    for outlet, outdict in channel_dict.items():
        for segment, segdict in outdict.items():
            for i, id in enumerate(segdict["ids"]):
                if id != outlet and id not in rev_Aid_node.keys():
                    id_path = list(segdict["ids"][:i]) + [id]
                    # go through segments and add them, unil a segment starts with outlet
                    while id_path[0] != outlet:
                        # print(id_path[0])
                        # look for previous segment
                        key = [key for key in outdict.keys() if id_path[0]==key[1]][0]
                        id_path = list(outdict[key]["ids"][:-1]) + id_path

                    # write ID of node in matrix
                    Aid_node[Aid_col] = id
                    rev_Aid_node[id] = Aid_col
                    
                    # id_path.pop(0) # keep outlet node to calculate dchi, for now!
                    gridid_path_nodes[id] = id_path
                    # for the current id, look at upstream and downstream neighors in the
                    # channel dict (only one downstream node, but multiple upstream nodes!)
                    # look at nodes that have the current id as their flow_link
                    
                    # we cannot work with receivers from landlab because we cut out some nodes!
                    # get receiver of current node:
                    if i==0: # go to previous segment
                        key = [key for key in outdict.keys() if id_path[0]==key[1]][0]
                        rec = outdict[key]["ids"][-2]
                        if rec != outlet:
                            rec = [rec]
                        else:
                            rec = []
                    else:
                        rec = segdict["ids"][i-1]
                        if rec != outlet:
                            rec = [rec]
                        else:
                            rec = []
                    
                    # now look for donors
                    if i<len(segdict["ids"])-1: # only one upstream node
                        don = [segdict["ids"][i+1]]
                    else: 
                        # node is at a possible channel junction (or headwater)
                        # check if there are any segments beginning with current node id
                        key = [key for key in outdict.keys() if id==key[0]]
                        don = []
                        for k in key:
                            don.append(outdict[k]["ids"][1])
                    
                    neighbors = rec + don
                    gridid_node_neighbors[id] = neighbors
                    Aid_node_neighbors[Aid_col] = neighbors
                    # print(id, neighbors)
                    # print(segdict["ids"])
                    
                    Aid_col += 1
    
    # convert the ids of Aid_node_neighors to Aid:
    for key, value in Aid_node_neighbors.items():
        Aid_node_neighbors[key] = [rev_Aid_node[v] for v in value]
        
    nA = len(gridid_path_nodes.keys())
    A = np.zeros((nA, nA))
    # print(Aid_node.keys(), rev_Aid_node.values(), len(rev_Aid_node.keys()))
    for i, (key, id_path) in enumerate(gridid_path_nodes.items()):
        dchi = np.diff(grid.at_node["channel__chi_index"][id_path])

        for j, id in enumerate(id_path[1:]):
            jA = rev_Aid_node[id]
            A[i,jA] = dchi[j]     
    
    if "channel__steepness_index" in grid.at_node:
        ksn_init = grid.at_node["channel__steepness_index"][flat_id_list]
    else:
        ksn_init = np.zeros(len(flat_id_list))
        
    z_true = grid.at_node["topographic__elevation"][flat_id_list]
    
    return (A, ksn_init, z_true, Aid_node_neighbors, list(rev_Aid_node.keys()))
    
def ksn_chiinv(
    grid,
    channel_dict,
    every_n: int = 1,
    alpha: float = 10.,
    minimize_args={"method": "Powell"},
    min_channel_threshold: float = 1e6
):

    grid.add_zeros("channel__inv_ksn")
    grid.at_node["channel__inv_ksn"][grid.at_node["drainage_area"]>=min_channel_threshold] = -1
    
    A, ksninit, ztrue, neighbor_dict, inv_ids = build_inversion_equation(
        grid=grid,
        channel_dict=channel_dict,
        every_n=every_n
    )
    
    out = minimize(
        fun=_chi_objective_function, 
        x0=ksninit, 
        args=(A, ztrue, neighbor_dict, alpha),
        bounds=[(0, None) for _ in ksninit],
        **minimize_args
    )
    
    grid.at_node["channel__inv_ksn"][inv_ids] = out.x
    grid = map_node_values_upstream(
        grid=grid,
        node_field_name="channel__inv_ksn",
        empty_value=-1
    )
    # the outlet node has -1 at the moment, just copy the value from above
    for outlet, outdict in channel_dict.items():
        key = [key for key in outdict.keys() if key[0]==outlet][0]
        grid.at_node["channel__inv_ksn"][outlet] = grid.at_node["channel__inv_ksn"][outdict[key]["ids"][1]]
    
    return grid

        
    
    
    