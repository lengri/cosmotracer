import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid
from landlab import NodeStatus

from scipy.ndimage import grey_opening

def calculate_codlean_shielding():
    print("NOT IMPLEMENTED")
    pass

def calculate_exact_shielding(
    grid: RasterModelGrid,
    field_name: str = "topographic__elevation",
    azimuth_bin_number: float = 20,
    filter_ridgelines: bool = False,
    threshold_quantile: float = 0.75,
    opening_element_size: int = 10,
    m: float = 2.3
):
    
    # calculate shielding factors for each (active) grid cell
    # in the user-defined field.
    
    # note that the -np.pi/np.pi discontinuity is in the W!!!
    bin_edges = np.linspace(-np.pi, np.pi, azimuth_bin_number+1)
    bin_width = bin_edges[1] - bin_edges[0]
    
    z = grid.at_node[field_name]
    
    # only consider open nodes
    open_node_ids = np.where(grid.status_at_node != NodeStatus.CLOSED)[0]
    open_node_mask = grid.status_at_node != NodeStatus.CLOSED
    
    shielding_factor = np.ones_like(z)
    
    if filter_ridgelines:
        
        # Use White Top Hat algorithm to select ridge line pixels
        dem = grid.at_node["topographic__elevation"].reshape(grid.shape)
        opened = grey_opening(dem, size=(opening_element_size, opening_element_size))
        wth = dem - opened 
        wth = wth.flatten()
        
        q_cutoff = np.quantile(wth, q=[threshold_quantile])
        wth_node_mask = wth > q_cutoff
    
    # for each active pixel...
    for current_id in open_node_ids:
        
        elev_node_mask = z > z[current_id]
            
        # only consider filtered pixels (mask, open, elevation)
        if filter_ridgelines:
            use_node_mask = np.logical_and(
                elev_node_mask,
                wth_node_mask,
                open_node_mask
            )
        else:
            use_node_mask = np.logical_and(
                elev_node_mask,
                open_node_mask
            )
        
        # check that there are at least some shielding pixels. If not, shielding is zero.
        if np.sum(use_node_mask) == 0:
            
            continue 
        
        else:         
            
            # create empty array of shielding values for each bin
            bin_max_skyline_angles = np.zeros(azimuth_bin_number)
            bin_x_max = np.full(azimuth_bin_number, np.nan)
            bin_y_max = np.full(azimuth_bin_number, np.nan)

            # calculate direction and distance between current pixel and all unmasked pixels 
            xt = grid.x_of_node[use_node_mask] - grid.x_of_node[current_id]
            yt = grid.y_of_node[use_node_mask] - grid.y_of_node[current_id]
            
            r = np.sqrt(xt**2 + yt**2)
            theta = np.arctan2(yt, xt)
            dz = z[use_node_mask] - z[current_id]
            skyline_angles = np.arctan(dz/r)
            
            # walk through theta in bins...
            for i, (b0, b1) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                id_node_in_bin = np.where(
                    np.logical_and(
                        theta>b0, theta<=b1
                    )
                )[0]
                
                # If there are still pixels left over, get the skyline angle now
                if len(id_node_in_bin) > 0:
                    
                    # for debugging, get the id!
                    id_max = np.argmax(skyline_angles[id_node_in_bin])
                    bin_x_max[i] = grid.x_of_node[use_node_mask][id_node_in_bin][id_max]
                    bin_y_max[i] = grid.y_of_node[use_node_mask][id_node_in_bin][id_max]
                    
                    bin_max_skyline_angles[i] = skyline_angles[id_node_in_bin][id_max]

            # sum up the shielding values
            
            shielding_factor[current_id] -= (1/(2*np.pi))*np.sum(bin_width*np.sin(bin_max_skyline_angles)**(m+1))
            
            #print("\n")
            #print(f"Current pixel ID: {current_id}")
            #print(f"\tShielding angles: {np.round(180/np.pi*bin_max_skyline_angles, 1)}")
            #print(f"\tShielding factor: {shielding_factor[current_id]}")

            if False:
                fg, ax = plt.subplots(1, 2)
                ax[0].imshow(
                    z.reshape(grid.shape), 
                    extent=[grid.x_of_node.min(), grid.x_of_node.max(), grid.y_of_node.min(), grid.y_of_node.max()],
                    origin="lower",
                    vmax=1000
                )
                ax[0].scatter(grid.x_of_node[current_id], grid.y_of_node[current_id])
                
                ax[0].scatter(bin_x_max, bin_y_max)
                
                all_theta = np.full(z.shape, np.nan)
                all_theta[use_node_mask] = theta
                print(bin_edges)
                ax[0].contour(
                    grid.x_of_node.reshape(grid.shape), 
                    grid.y_of_node.reshape(grid.shape), 
                    all_theta.reshape(grid.shape),
                    levels=bin_edges,
                    colors=["white"],
                    origin="lower"
                )
                ax[0].set_xlim(grid.x_of_node.min(), grid.x_of_node.max())
                ax[0].set_ylim(grid.y_of_node.min(), grid.y_of_node.max())
                
                ax[1].imshow(
                    all_theta.reshape(grid.shape), 
                    extent=[grid.x_of_node.min(), grid.x_of_node.max(), grid.y_of_node.min(), grid.y_of_node.max()],
                    origin="lower",
                )
                ax[1].contour(
                    grid.x_of_node.reshape(grid.shape), 
                    grid.y_of_node.reshape(grid.shape), 
                    all_theta.reshape(grid.shape),
                    levels=bin_edges,
                    colors=["white"],
                    origin="lower"
                )
                plt.show()
        
    
    return shielding_factor
            
            
if __name__ == "__main__":
    
    import cosmotracer as ct
    
    grid = ct.CosmoLEM(
        shape=(200, 200),
        xy_spacing=25,
        allow_cache=True,
        epsg=32611
    )

    grid.set_outlet_position("SW")
    
    grid.load_grid(
        T_total=50e6,
        U=1e-3,
        K_sp=1e-5,
        dt=2500
    )
    
    shielding = calculate_exact_shielding(grid)
    
    
    fg, ax = plt.subplots(1, 2)
    ax[0].imshow(grid.at_node["topographic__elevation"].reshape(grid.shape), origin="lower")
    ax[1].imshow(shielding.reshape(grid.shape), vmax=1, vmin=0, origin="lower")
    plt.show()