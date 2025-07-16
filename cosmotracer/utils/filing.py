import numpy as np
import os 
import geopandas as gpd
from shapely import Point

# For caching stuff
import h5py, platformdirs
from pathlib import Path
from datetime import datetime, timezone

def export_field_to_gpgk(
    mg,
    node_field_name : str,
    epsg : int,
    filepath : str,
    empty_value = 0.,
    
):
    """
    Takes all x, y, field data of nodes that are != empty_value
    and constructs a GeoPackage of points with attached values
    """
    
    ids = np.where(mg.at_node[node_field_name] != empty_value)
    
    x = mg.x_of_node[ids]
    y = mg.y_of_node[ids]
    z = mg.at_node[node_field_name][ids]
    
    points = [Point(xx, yy) for xx, yy in zip(x, y)]
    gdf = gpd.GeoDataFrame(geometry=points, crs=f"EPSG:{epsg}")
    gdf[node_field_name] = z
    
    gdf.to_file(filepath, layer=node_field_name, driver="GPKG")
    
def export_field_to_ascii(
    mg,
    node_field_name,
    filepath
):
    
    out = mg.at_node[node_field_name].reshape(mg.shape)
    x_ll, y_ll = mg.xy_of_lower_left
    
    header = {
        "ncols": out.shape[1],
        "nrows": out.shape[0],
        "xllcorner": x_ll,
        "yllcorner": y_ll,
        "cellsize": mg.dx,
        "nodata_value": -999999.
    }
    header_lines = [f"{key} {str(val)}" for key, val in list(header.items())]
    np.savetxt(
        filepath, np.flipud(out), header=os.linesep.join(header_lines), comments=""
    )

PACKAGE_NAME = "cosmotracer"
      
class Cache():
    
    def __init__(
        self,
        allow_cache = True,
        fail_on_overwrite = True
    ):
        
        # Global options for this Cache instance
        self.cache_allowed = allow_cache
        self.fail_on_overwrite = fail_on_overwrite
        
        # Initialise the cache if it is allowed and does not exist...
        if self.cache_allowed:
            self.cache_dir : Path = platformdirs.user_cache_path(PACKAGE_NAME)
            self.cache_dir.mkdir(parents=True, exist_ok=True)        
        
    def load_file(
        self,
        filekey
    ):
        
        # do nothing if there is no caching allowed
        if not self.cache_allowed:
            return None

        cache_filepath = self.cache_dir / filekey
        
        try:
            with h5py.File(cache_filepath, "r") as h5:
                dataset = h5["data"][()]
                return dataset
        except Exception as err:
            print(f"Could not load cache: {err}")
            return None
            
    
    def save_file(
        self, filekey, array
    ):
        
        # do nothing here as well
        if not self.cache_allowed:
            return None
        
        cache_filepath = self.cache_dir / filekey
        
        if self.fail_on_overwrite:
            if os.path.exists(cache_filepath):
                raise Exception(f"Caching file would overwrite existing file {cache_filepath}")
        
        try:
            with h5py.File(cache_filepath, "w") as h5:
                h5.create_dataset(
                    "data",
                    data=array,
                    compression="gzip",
                    compression_opts=4
                )
                now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
                h5.attrs["created"] = now_iso
        
        except Exception as err:
            print(f"Could not write to cache: {err}")

        