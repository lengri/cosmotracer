
from landlab.components import ChiFinder, ChannelProfiler
from itertools import combinations

import numpy as np
from cosmotracer.topo import ChiQFinder

def _get_channels_upstream(grid, starting_nodes, min_drainage_area):
        """Get channel structure upstream of a list of nodes.
        This function is copied and modified from ChannelProfiler!

        Parameters
        ----------
        i : int, required
            Node id of start of channel segment.

        Returns
        ----------
        channel_segment : list
            Node IDs of the nodes in the current channel segment.
        nodes_to_process, list
            List of nodes to add to the processing queue. These nodes are those
            that drain to the upper end of this channel segment. If
            main_channel_only = False this will be an empty list.
        """
        
        # Store unique nodes only
        channel_nodes = list(starting_nodes)
        
        for ch_outlet in starting_nodes:
            
            upstream_nodes = np.where(grid.at_node["flow__receiver_node"]==ch_outlet)[0]
            upstream_nodes = upstream_nodes[grid.at_node["drainage_area"][upstream_nodes]>=min_drainage_area]
            upstream_nodes = [idu for idu in upstream_nodes if idu not in channel_nodes]
            nodes_to_process = upstream_nodes
            channel_nodes += upstream_nodes
            
            while len(nodes_to_process) > 0:
                
                # Add all channel nodes flowing into nodes_to_process[0],
                # then pop the first node
                
                upstream_nodes = np.where(grid.at_node["flow__receiver_node"]==nodes_to_process[0])[0]
                upstream_nodes = upstream_nodes[grid.at_node["drainage_area"][upstream_nodes]>=min_drainage_area]
                upstream_nodes = [idu for idu in upstream_nodes if idu not in channel_nodes]

                nodes_to_process += upstream_nodes
                channel_nodes += upstream_nodes
                
                nodes_to_process.pop(0)
        
        return channel_nodes

def _get_drainage_area_around_index(
    grid,
    linear_index
):
    rc_id = np.unravel_index(linear_index, grid.shape)
    
    # calculate all surrounding indices, get drainage!
    nodes_around = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            r = rc_id[0]+i
            c = rc_id[1]+j
            try:
                l_id = np.ravel_multi_index((r, c), grid.shape)
                nodes_around.append(
                    (l_id, grid.at_node["drainage_area"][l_id])
                )
            except:
                continue
    
    return nodes_around
    

def _get_tribs_ids(
    grid,
    trunk_nodes,
    min_drainage_area
):
    
    tributary_ids = []
    
    for current_trunk_node in trunk_nodes:
        Ai = _get_drainage_area_around_index(
            grid, 
            current_trunk_node
        )
        for i_around, A_around in Ai:
            if (not i_around in trunk_nodes) and  (i_around not in tributary_ids) and A_around >= min_drainage_area:
                
                # Quickly check that this node doesn't have a receiver that
                # we already identified as a tributary. (Can happen if two
                # streams are running parallel for a few pixels!)
                
                # Actually, we have to check this recursively until we reach the
                # trunk just to be sure!
                
                irec = i_around 
                trib_exists = False
                while irec not in trunk_nodes:
                    irec = grid.at_node["flow__receiver_node"][irec]
                    if irec in tributary_ids:
                        trib_exists = True
                        break
                    
                if not trib_exists:                 
                    tributary_ids.append(i_around)
    
    return tributary_ids
def bootstrap_mn_chi_disorder(
    grid,
    min_drainage_area: float = 1e6,
    use_discharge: bool = False, 
    mn_range: tuple = (0.1, 1.5),
    n_tribs: int = 3, # WARNING: Do not set above ~4!
    max_repeats: int = 100,
    n_mn: int = 20
):
    
    # Follow Mudd et al. (2018) in calculating disorder metrics not just for
    # entire network, but by selecting all combinations of n tributaries, and
    # calculate disorder for those (and the trunk). Reprort IQR!
    
    # Optional: Optimise mn for chiQ, not chi! Requires a node discharge field called "discharge"!
    
    # identify trunk stream
    cp_trunk = ChannelProfiler(
        grid=grid,
        minimum_channel_threshold=min_drainage_area
    )
    cp_trunk.run_one_step()
       
    id_trunk = list(list(cp_trunk.data_structure.values())[0].values())[0]["ids"]
    
    print("Finding tributary combinations...", end="\r")
    # get trib nodes where they join the trunk stream (non-overlapping)
    trib_ids = _get_tribs_ids(grid, id_trunk, min_drainage_area)
    trib_combinations = list(combinations(trib_ids, n_tribs))
    
    print(f"Finding tributary combinations... found {len(trib_combinations)}")
    # if max_repeats < len(trib_combis), draw random combinations now and iterate over them
    if max_repeats < len(trib_combinations):
        id_choose = np.random.choice(len(trib_combinations), size=max_repeats, replace=False)
    else:
        id_choose = np.arange(len(trib_combinations))
        
    # for each combination of tribs, calculate the binary mask of valid pixels
    
    masks = {}
    for i, idc in enumerate(id_choose):
        print(f"Calculating bootstrap masks... {i+1}/{len(id_choose)}", end="\r")
        id_valid = id_trunk 
        current_tribs = trib_combinations[idc]

        # for i_tr in current_tribs:
        
        trib_network_nodes = _get_channels_upstream(
            grid, starting_nodes=current_tribs, min_drainage_area=min_drainage_area
        )

        id_valid = np.concatenate((id_valid, trib_network_nodes))
        
        # id valid is float for some reason, recast to int here
        id_valid = np.array([int(id_v) for id_v in id_valid], dtype=int)
        
        i_mask = np.zeros(grid.number_of_nodes, dtype=bool)

        i_mask[id_valid] = True
        masks[idc] = i_mask 
    
    print(f"Calculating bootstrap masks... {len(id_choose)}/{len(id_choose)}")
    
    mns = np.linspace(mn_range[0], mn_range[1], n_mn)
    disorder_stats = np.zeros((n_mn, len(id_choose)))
    
    for i, mn in enumerate(mns):
        
        # calculate chi or chiQ
        if use_discharge:
            chi = ChiQFinder(
                grid=grid,
                discharge_field="discharge",
                reference_concavity=mn,
                min_drainage_area=min_drainage_area,
                clobber=True
            )
            chi.calculate_chiQ()
        else:
            chi = ChiFinder(
                grid=grid,
                reference_area=1.,
                reference_concavity=mn,
                min_drainage_area=min_drainage_area,
                clobber=True
            )
            chi.calculate_chi()
        
        for j, idc in enumerate(id_choose):
            print(f"Iterating m/n = {mn:.2f}, {j+1}/{len(id_choose)}  ", end="\r")
            # get mask of triburary combination
            id_mask = masks[idc]

            #plt.show()
            # sort by elevation
            id_sorted = np.argsort(
                grid.at_node["topographic__elevation"][id_mask]
            )
            
            chi_network = grid.at_node["channel__chi_index"][id_mask][id_sorted]
            chi_max = chi_network.max()

            # calculate disorder: 
            disorder_stats[i,j] = (1/chi_max) * \
                (np.sum(np.abs(chi_network[1:]-chi_network[:-1]))-chi_max)

    # Calculate summary statistics
    quants = np.quantile(
        disorder_stats, 
        q=[0.25, 0.5, 0.75], 
        axis=1
    )
    
    id_min = np.argmin(quants[1])
    # check in which range q25 < q50_min
    uncert_range_id = np.where(
        quants[0]<=quants[1][id_min]
    )[0]
    
    mn_best = mns[id_min]
    mn_min = mns[uncert_range_id[0]]
    mn_max = mns[uncert_range_id[-1]]
    
    return (mn_best, mn_min, mn_max, mns, quants, disorder_stats)