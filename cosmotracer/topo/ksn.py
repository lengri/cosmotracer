import numpy as np
import scipy as sp
from landlab import (
    Component,
    NodeStatus
)
import warnings

class SteepnessInverter(Component):
    
    """
    Derive channel-steepness indices of a channel network
    from linear inversion of chi plots.
    """
    
    _name = "SteepnessInverter"
    
    _unit_agnostic = True 
    
    _info = {
        "channel__ks_inv": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "variable",
            "mapping": "node",
            "doc": "Inverted steepness indices"
        },
        "channel__z_inv": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Elevation re-calculated from inverted ks"
        },
         "flow__upstream_node_order": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array containing downstream-to-upstream ordered list of node IDs",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "drainage_area": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m**2",
            "mapping": "node",
            "doc": "Upstream accumulated surface area contributing to the node's discharge",
        }
    }
    
    def __init__(
        self, 
        grid, 
        min_drainage_area: float = 1e6,
        regularization_coefficient: float = 1.
    ):
        self._grid = grid 
        self._Acrit = min_drainage_area
        
        # get the valid network nodes (core nodes, and exclude short channels)
        self._network_ids = self.valid_network_nodes(
            Acrit=self._Acrit
        )
        
        if len(self._network_ids) == 0:
            raise ValueError(
                f"Network of nodes for inversion contains no valid entries! " 
                f"Try reducing {self._Acrit=:.2e}."
            )
            
        self._n_nodes = len(self._network_ids)
        
        # Keep track of where grid IDs are located in the inversion matrices
        self._grid2mat = {node: i for i, node in enumerate(self._network_ids)}
        
        self._z = np.zeros(self._n_nodes) # Will store z values of the network
        self._z_bl = np.zeros(self._n_nodes) # Will store the baselevel of each node (outlet)
        self._fill_inversion_matrices()
        
        self.alpha = regularization_coefficient
        
        # initialise fields to store results
        self._grid.add_zeros("channel__ks_inv", at="node")
        self._grid.add_zeros("channel__z_inv", at="node")
        
    @property
    def alpha(self):
        return self._alpha 
    
    @alpha.setter
    def alpha(self, alpha):
        if alpha < 0:
            raise ValueError("Alpha cannot be negative")
        
        self._alpha = alpha
    
    def _fill_inversion_matrices(self):
        
        # Use sparse matrix representation!
        rows_A, cols_A, vals_A = [], [], []
        rows_D, cols_D, vals_D = [], [], []
    
        # loop over all ids, store Dchi of downstream nodes in A, and id/rec Dchi in D.
        for i, node in enumerate(self._network_ids):
            
            print(f"Filling matrices for node {node} ({i} of {self._n_nodes})     ", end="\r")
            
            #
            # Fill dChi (or A) matrix
            #
            
            # Get list of downstream nodes
            rec_ids = self._get_downstream_ids(node)
            
            # get delta chi for this list of downstream nodes (exclude outlet node)
            dChi = self.delta_chi(rec_ids[:-1])
            
            # place in A, use grid2mat to map the positions.            
            rows_A += [i]*len([self._grid2mat[j] for j in rec_ids[:-1]])
            cols_A += [self._grid2mat[j] for j in rec_ids[:-1]]
            vals_A += list(dChi)
            
            #
            # Fill z vector
            #
            
            # fill z vector (adjusted by baselevel elevation), save bl for later
            self._z_bl[i] = self._grid.at_node["topographic__elevation"][rec_ids[-1]]
            # BL-adjusted elevation values of network
            self._z[i] = self._grid.at_node["topographic__elevation"][node] - \
                self._z_bl[i]
            
            #
            # Fill dampening vector
            #
            
            # Fill is similar to a Graph-Laplacian,
            # gradient-based for boundary nodes.
            
            don_ids = self._get_donor_ids(node)
            # Need to filter donor nodes: 
            # Only use those that are part of the network!
            don_ids = np.array([d for d in don_ids if d in self._network_ids])
            
            rec_id = self._get_receiver_id(node)
            
            # If we are a single node above the boundary, we use
            # upstream ks for gradient-based criterion.
            # (Since we filter out channels of total length 2,
            # there must be at least one donor!)
            if rec_id == self._get_receiver_id(rec_id):
                
                nn = len(don_ids)
                don_dChis = self.delta_chi(don_ids)
                sum_donors = 0
                
                for j, don in enumerate(don_ids):
                    sum_donors += 1/(nn*don_dChis[j])
                    rows_D.append(i)
                    cols_D.append(self._grid2mat[don])
                    vals_D.append(1/(nn*don_dChis[j]))

                rows_D.append(i)
                cols_D.append(self._grid2mat[node])
                vals_D.append(-sum_donors)
            # Other case: There are no donors! In this case, we are
            # at the headwater boundaries. Here we also use a gradient-based
            # criterion with the receiver node
            elif len(don_ids) == 0:
                
                node_dChi = self.delta_chi(node)

                rows_D += [i, i]
                cols_D += [self._grid2mat[node], self._grid2mat[rec_id]]
                vals_D += [1/node_dChi, -1/node_dChi]

            # Final case: we are not at lower or upper boundary!
            # Thus, there has to be one receiver and at least one
            # donor. We can use curvature here.
            else:
                
                nn = len(don_ids)
                node_dChi = self.delta_chi(node)

                # There are some donor nodes! Loop over them and calculate terms for matrix!
                don_dChis = self.delta_chi(don_ids)
                S = (len(don_ids) + 1) / (node_dChi + np.sum(don_dChis))
                sum_donors = 0
                for j, dn in enumerate(don_ids):
                    sum_donors += S/(nn*don_dChis[j]) 
                    rows_D.append(i)
                    cols_D.append(self._grid2mat[dn])
                    vals_D.append(S/(nn*don_dChis[j]))       
                
                # Add entry for node itself and receiver
                rows_D += [i, i]
                cols_D += [self._grid2mat[node], self._grid2mat[rec_id]]   
                vals_D += [-(S/(node_dChi) + sum_donors), S/node_dChi]  
                
        print("\n")
        
        # build up to sparse matrices
        self._A_sp = sp.sparse.csr_matrix(
            (vals_A, (rows_A, cols_A)), 
            shape=(self._n_nodes, self._n_nodes)
        )
        self._D_sp = sp.sparse.csr_matrix(
            (vals_D, (rows_D, cols_D)), 
            shape=(self._n_nodes, self._n_nodes)
        )
    
    def invert(
        self, 
        regularization_coefficient: None|float = None
    ):
        """
        The main function: Invert ks values given a 
        regularization coefficient.
        
        This function will fill two output fields,
        "channel__ks_inv", and "channel__z_inv".
        """
        
        if regularization_coefficient is not None:
            self.alpha = regularization_coefficient
            
        K_sp = sp.sparse.vstack([self._A_sp, self.alpha*self._D_sp])
        rhs = np.hstack([self._z, np.zeros(self._D_sp.shape[0])])

        self._ks = sp.sparse.linalg.lsmr(K_sp, rhs)[0]
        self._grid.at_node["channel__ks_inv"][self._network_ids] = self._ks
        
        # reconstruct z, recall that we subtracted the baselevel!
        self._grid.at_node["channel__z_inv"][self._network_ids] = \
            self._A_sp @ self._ks + self._z_bl
            
        # Problem: The outlet itself is not part of network_ids, so we need to
        # set it to baselevel manually...
        outlet = self._get_receiver_id(self._network_ids[0])
        self._grid.at_node["channel__z_inv"][outlet] = self._z_bl[0]
            
        # Note, alternative way of solving using the normal equations:
        # lhs = self._A.T @ self._A + self.alpha**2 * self._D.T @ self._D
        # rhs = self._A.T @ self._z
        # ks = sp.linalg.solve(lhs, rhs)
        # This is much slower than the iterative solver!
        
        return self._ks
    
    def misfit_roughness(self):
        
        # Calculates ||A ks - z||2 and ||D ks||2 as misfit and roughness of 
        # the solution (for the given alpha) respectively.
        
        self.misfit = np.linalg.norm(self._A_sp @ self._ks - self._z)
        self.roughness = np.linalg.norm(self._D_sp @ self._ks)
        
        return (self.misfit, self.roughness)
        
    def _get_donor_ids(self, node_id):
        """
        Get ids of donors (note that this also includes
        non-channel nodes) for current node.
        """
        return np.where(self._grid.at_node["flow__receiver_node"]==node_id)[0]
    
    def _get_receiver_id(self, node_id):
        """
        Get id of the downstream receiver node.
        """
        return self._grid.at_node["flow__receiver_node"][node_id]
    
    def _get_downstream_ids(self, node_id):
        """
        Returns a list of nodes from the current node to the outlet.
        """
     
        rid = self._get_receiver_id(node_id)
        
        if rid == node_id: # Input is boundary node
            warnings.warn("Cannot return downstream path of boundary node!")
            return []
        
        ids = [node_id, rid]
        
        rid = self._get_receiver_id(ids[-1])
        while rid != ids[-1]:
            ids.append(rid)
            rid = self._get_receiver_id(ids[-1])
        
        return ids

    def delta_chi(self, node_id):
        """
        Calculate difference in chi between input node and
        its receiver
        """
        # NOTE: This also works vectorized!!
        rid = self._get_receiver_id(node_id)
        chi1 = self._grid.at_node["channel__chi_index"][node_id]
        chi0 = self._grid.at_node["channel__chi_index"][rid]
        return chi1-chi0
    
    def valid_network_nodes(self, Acrit):
        
        """
        Returns a list of valid network nodes in upstream flow order.
        
        This is useful for tracking the meaning(s) of the inversion matrices.
        """
        
        # Get IDs of network nodes that fulfill Acrit and are not boundary nodes
        # Further, we filter out channels that only consist of 
        # a boundary node + 1 other node since we cannot apply
        # any smoothness constraints here.
        
        id_net = np.logical_and(
            self._grid.at_node["drainage_area"]>=Acrit,
            self._grid.status_at_node==NodeStatus.CORE
        )
        id_net = self._grid.nodes.flatten()[id_net]
        
        # sort these nodes according to node_order_upstream!
        id_net_sorted = [
            id_nou for id_nou in self._grid.at_node["flow__upstream_node_order"] 
            if id_nou in id_net
        ]
        
        # Filter out nodes that are next to boundary 
        # and that have no donor, i.e. channels that
        # consist of two nodes exactly!
        id_net_out = []
        for id in id_net_sorted:
            rec = self._get_receiver_id(id)
            don = self._get_donor_ids(id)
            if self._grid.status_at_node[rec] != NodeStatus.CORE:
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