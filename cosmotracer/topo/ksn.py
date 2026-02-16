import numpy as np
import scipy as sp
from landlab import Component
import warnings

from landlab.components import ChiFinder, ChannelProfiler
from itertools import combinations

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
    mn_range: tuple = (0.1, 1.5),
    n_tribs: int = 3, # WARNING: Do not set above ~4!
    max_repeats: int = 100,
    n_mn: int = 20
):
    
    # Follow Mudd et al. (2018) in calculating disorder metrics not just for
    # entire network, but by selecting all combinations of n tributaries, and
    # calculate disorder for those (and the trunk). Reprort IQR!
    
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

        """cp_trib = ChannelProfiler(
            grid=grid,
            outlet_nodes=list(current_tribs),
            number_of_watersheds=len(current_tribs),
            minimum_channel_threshold=min_drainage_area,
            main_channel_only=False
        )
        cp_trib.run_one_step()
        id_valid_tr = _get_all_ChannelFinder_ids(cp_trib.data_structure)"""
        
        trib_network_nodes = _get_channels_upstream(
            grid, starting_nodes=current_tribs, min_drainage_area=min_drainage_area
        )

        id_valid = np.concatenate((id_valid, trib_network_nodes))
        
        # id valid is float for some reason, recast to int here
        id_valid = np.array([int(id_v) for id_v in id_valid], dtype=int)
        
        i_mask = np.zeros(grid.number_of_nodes, dtype=bool)

        i_mask[id_valid] = True
        masks[idc] = i_mask 
        
        """import matplotlib.pyplot as plt
        plt.imshow(
            i_mask.reshape(grid.shape),
            origin="lower",
            extent=[
                grid.x_of_node.min(), grid.x_of_node.max(),
                grid.y_of_node.min(), grid.y_of_node.max()
            ]
        )
        plt.scatter(
            grid.x_of_node[list(current_tribs)],
            grid.y_of_node[list(current_tribs)],
            c="red"
        )
        plt.show()"""
    
    print(f"Calculating bootstrap masks... {len(id_choose)}/{len(id_choose)}")
    
    mns = np.linspace(mn_range[0], mn_range[1], n_mn)
    disorder_stats = np.zeros((n_mn, len(id_choose)))
    
    for i, mn in enumerate(mns):
        
        # calculate chi
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


def mn_chi_disorder(
    grid,
    min_drainage_area: float = 1e6,
    mn_range: tuple = (0.1, 1.5)
):
    
    is_channel = grid.at_node["drainage_area"] >= min_drainage_area
    
    # we only need to sort the network once (along elevation)
    id_sorted = np.argsort(
        grid.at_node["topographic__elevation"][is_channel]
    )
    
    mns = np.linspace(mn_range[0], mn_range[1], 20)
    disorder = np.zeros(mns.shape)
    
    for i, mn in enumerate(mns):
        chi = ChiFinder(
            grid=grid,
            min_drainage_area=min_drainage_area,
            clobber=True,
            reference_concavity=mn
        )
        chi.calculate_chi()
        
        chi_network = grid.at_node["channel__chi_index"][is_channel][id_sorted]
        chi_max = chi_network.max()
        # calculate disorder: 
        disorder[i] = (1/chi_max) * \
            (np.sum(np.abs(chi_network[1:]-chi_network[:-1]))-chi_max)
    
    return mns, disorder


class SteepnessInverter(Component):
    
    def __init__(
        self, 
        grid, 
        min_drainage_area: float = 1e6,
        regularization_coefficient: float = 1.
    ):
        self._grid = grid 
        self._Acrit = min_drainage_area
        
        # get the valid network
        self._network_ids = self.valid_network_nodes(
            Acrit=self._Acrit
        )
        
        if len(self._network_ids) == 0:
            warnings.warn("No valid channel nodes in network!")
            
        self._n_nodes = len(self._network_ids)
        
        # Keep track of where grid IDs are located in the inversion matrices
        self._grid2mat = {}
        
        self._z = np.zeros(self._n_nodes) # Will store z values of the network
        self._z_bl = np.zeros(self._n_nodes) # Will store the baselevel of each node (outlet)
        self._A = np.zeros((self._n_nodes, self._n_nodes)) # Inversion matrix containing deltaChi
        self._D = np.zeros((self._n_nodes, self._n_nodes)) # Dampening matrix
        self._fill_inversion_matrices()
        
        self._alpha = regularization_coefficient
        
        self._grid.add_zeros("channel__ksn_inv", at="node")
        self._grid.add_zeros("channel__z_inv", at="node")
    
    def _fill_inversion_matrices(self):
        
        boundary_rec_don = [] # Init list, fill with boundary nodes for later
    
        # loop over all ids, store Dchi of downstream nodes in A, and id/rec Dchi in D.
        for i, node in enumerate(self._network_ids):
            print(f"Node {i} of {self._n_nodes}", end="\r")
            
            # We know that the current node is new, add it to the dict!
            self._grid2mat[node] = i 
            
            # Get list of downstream nodes
            rec_ids = self._get_downstream_ids(node)
            
            # fill z vector (adjusted by baselevel elevation), save bl for later
            self._z_bl[i] = self._grid.at_node["topographic__elevation"][rec_ids[-1]]
            self._z[i] = self._grid.at_node["topographic__elevation"][node] - \
                self._z_bl[i]

            # get delta chi for this list of downstream nodes (exclude outlet node)
            dChi = self.delta_chi(rec_ids[:-1])
            
            # place in A, use grid2mat to map the positions.
            rec_ids_mat = [self._grid2mat[j] for j in rec_ids[:-1]]
            self._A[i,rec_ids_mat] = dChi
            
            # Edge case: The current node is sitting 1 above (one of the) outlet(s)!
            # For these, constraint should just be that ksn is consistent with
            # one random donor node?
            
            if len(rec_ids) != 2:
                self._D[i,rec_ids_mat[:2]] = [dChi[0], -dChi[0]]
            else:
                don_id = self._get_donor_ids(node)[0]
                don_id = don_id
                # problem: don_id might not yet exist in grid2mat!
                # save them and fill later after we are done iterating
                boundary_rec_don.append((node, don_id))
        print("\n")
        # Now we can add boundary conditions to the matrix!
        don_ids = [don for _, don in boundary_rec_don]
        boundary_dchi = self.delta_chi(don_ids)
        # Place into D matrix
        for i, donor in enumerate(don_ids):
            # get receiver to check which row we're in
            rec_id = self._get_receiver_id(donor)
            ii = self._grid2mat[rec_id]
            jj = self._grid2mat[donor]
            
            self._D[ii,[ii,jj]] = \
                [boundary_dchi[i], -boundary_dchi[i]]
    
    def invert(self, regularization_coefficient: None|float = None):
        
        if regularization_coefficient is not None:
            self._alpha = regularization_coefficient
            
        K = np.vstack([self._A, self._alpha*self._D])
        rhs = np.hstack([self._z, np.zeros(self._D.shape[0])])

        self._ks = sp.sparse.linalg.lsmr(K, rhs)[0]
        self._grid.at_node["channel__ksn_inv"][self._network_ids] = self._ks
        
        # reconstruct z, recall that we subtracted the baselevel!
        self._grid.at_node["channel__z_inv"][self._network_ids] = \
            self._A @ self._ks + self._z_bl
            
        # Note, alternative way of solving using the normal equations:
        # lhs = self._A.T @ self._A + self._alpha**2 * self._D.T @ self._D
        # rhs = self._A.T @ self._z
        # ks = sp.linalg.solve(lhs, rhs)
        # This is much slower than the iterative solver!
        
        # compute misfit and roughness of solution for L-curve
        self.misfit_roughness()
        
        return self._ks
    
    def misfit_roughness(self):
        
        # Calculates ||A ks - z||2 and ||D ks||2 as misfit and roughness of 
        # the solution (for the given alpha) respectively.
        
        self.misfit = np.linalg.norm(self._A @ self._ks - self._z)
        self.roughness = np.linalg.norm(self._D @ self._ks)
        
        return (self.misfit, self.roughness)
        
    def _get_donor_ids(self, node_id):
        return np.where(self._grid.at_node["flow__receiver_node"]==node_id)[0]
    
    def _get_receiver_id(self, node_id):
        return self._grid.at_node["flow__receiver_node"][node_id]
    
    def _get_downstream_ids(self, node_id):
        # get a list of downstream nodes, excluding the outlet
        rid = self._get_receiver_id(node_id)
        
        if rid == node_id: # Reached boundary node
            warnings.warn("Cannot return downstream path of boundary node!")
            return None
        
        ids = [node_id, rid]
        
        # other special case, one node above outlet
        
        rid = self._get_receiver_id(ids[-1])
        while rid != ids[-1]:
            ids.append(rid)
            rid = self._get_receiver_id(ids[-1])
        
        return ids

    def delta_chi(self, node_id):
        # Calculate Dchi of current node and downstream neighbour
        # NOTE: This also works vectorized!!
        rid = self._get_receiver_id(node_id)
        chi1 = self._grid.at_node["channel__chi_index"][node_id]
        chi0 = self._grid.at_node["channel__chi_index"][rid]
        return chi1-chi0
    
    def valid_network_nodes(self, Acrit):
    # Get IDs of network nodes that fulfill Acrit and are not boundary nodes
    # Further, we filter out channels that only consist of 
    # a boundary node + 1 other node
    
        id_net = np.logical_and(
            self._grid.at_node["drainage_area"]>=Acrit,
            self._grid.status_at_node==0
        )
        id_net = self._grid.nodes.flatten()[id_net]
        
        # sort these nodes according to node_order_upstream!
        id_net_sorted = [
            id_nou for id_nou in self._grid.at_node["flow__upstream_node_order"] if id_nou in id_net
        ]
        
        # Filter out nodes that are next to boundary and that have no donor
        id_net_out = []
        for id in id_net_sorted:
            rec = self._get_receiver_id(id)
            don = self._get_donor_ids(id)
            if self._grid.status_at_node[rec] != 0:
                if len(don) == 0:
                    continue 
            
            id_net_out.append(id)
                
        return id_net_out


def calculate_segmented_ksn(
    id_segments,
    chi_segments,
    z_segments,
    segment_break_ids = [],
    bad_segment_value=-1
):
    """
    Input are nested lists where each entry in id_segments, chi_segments etc. corresponds to
    one segment defined by ids in a global grid (not important here, but we keep the information),
    chi values (the x coordinate), and z values (the y coordinate).
    We also provide further break ids. If one segment contains one or more of these break ids,
    it is split into multiple segments along the break ids.
    
    After having created these new segments, we calculate the average slope of each.
    """
    if len(segment_break_ids) > 0:
        id_segments_use = []
        chi_segments_use = []
        z_segments_use = []
        for id_s, chi_s, z_s in zip(id_segments, chi_segments, z_segments):
            # check if current segment contains break_id
            break_ids = [i for i, _id in enumerate(id_s) if _id in segment_break_ids]
            
            if len(break_ids) == 0:
                id_segments_use.append(id_s)
                chi_segments_use.append(chi_s)
                z_segments_use.append(z_s)
            else:
                # ensure sorted order and include start/end
                break_ids = sorted(break_ids)
                split_points = [0] + [i+1 for i in break_ids] + [len(id_s)]
                
                # slice the segment into subsegments
                for start, end in zip(split_points[:-1], split_points[1:]):
                    id_segments_use.append(id_s[start:end])
                    chi_segments_use.append(chi_s[start:end])
                    z_segments_use.append(z_s[start:end])
    else:
        id_segments_use = id_segments 
        chi_segments_use = chi_segments 
        z_segments_use = z_segments 
    
    slopes = np.zeros(len(id_segments_use))
    for i, (chi, z) in enumerate(zip(chi_segments_use, z_segments_use)):
        if len(chi) > 1:
            slope, _, _, _, _ = sp.stats.linregress(chi, z)
            slopes[i] = slope
        else: 
            slopes[i] = bad_segment_value 
            
    return slopes