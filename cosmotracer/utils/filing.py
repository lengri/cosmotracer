import numpy as np
import os 
import geopandas as gpd
from shapely import Point, box

# For caching stuff
import h5py, platformdirs
from pathlib import Path
from datetime import datetime, timezone
import ujson

# For the watershed export
from affine import Affine
from scipy import ndimage

from landlab import RasterModelGrid

def get_cachedir():
    """
    Returns the directory of the cache for cosmotracer.
    """
    
    return platformdirs.user_cache_dir()

def export_points_to_gpkg(
    x : np.ndarray, 
    y : np.ndarray, 
    z : np.ndarray, 
    filepath : str, 
    epsg : int
):
    """
    Export points with coordinates (x, y, z) to a GeoPackage file.

    Parameters:
    - x, y, z: list/array of coordinates
    - epsg: EPSG code for CRS (default 4326)
    - filepath: output file path for .gpkg
    """
    geometry = [Point(x[i], y[i], z[i]) for i in range(len(x))]

    gdf = gpd.GeoDataFrame(geometry=geometry, crs=f"EPSG:{epsg}")
    gdf.to_file(filepath, driver="GPKG")
    
def export_watershed_to_gpkg(
    grid: RasterModelGrid,
    mask: np.ndarray,
    filepath: str,
    epsg: int | None = None
):
    
    try:
        if grid.epsg is not None:
            epsg = grid.epsg
    except:
        if epsg is None:
            raise Exception("EPSG not supplied by grid or function argument!")
    
    # mask has the same alignment as grid!
    # (0, 0) is lower left corner
    
    rows, cols = np.where(mask)
    # print(grid.xy_of_node[0])
    # print(grid.xy_of_lower_left)
    polygons = []
    for r, c in zip(rows, cols):
        
        ilin = np.ravel_multi_index((r, c), grid.shape)
        xll, yll = grid.xy_of_node[ilin]
        xtr, ytr = xll+grid.dx, yll+grid.dy
        
        pixel_poly = box(xll, yll, xtr, ytr)
        polygons.append(pixel_poly)

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=f"EPSG:{epsg}")
    gdf = gdf.dissolve()

    # Export
    gdf.to_file(filepath, driver="GPKG")  
    
    """# Label connected components (watersheds) in the mask
    labeled, num_features = ndimage.label(mask)

    transform = Affine.translation(xy_ll_corner[0], xy_ll_corner[1]) * Affine.scale(cellsize, -cellsize)
    
    polygons = []

    for region_id in range(1, num_features + 1):
        region = labeled == region_id
        rows, cols = np.where(region)

        for r, c in zip(rows, cols):
            # Get real-world coordinates of the pixel center
            x, y = transform * (c, r)
            x -= cellsize/2 # shift from lower left to cell center
            y += cellsize/2 # shift from lower left to cell center
            # Build pixel-sized polygon (square around pixel)
            pixel_poly = box(x, y - cellsize, x + cellsize, y)
            polygons.append(pixel_poly)

    # Merge all into one polygon (if desired)
    crs = f"EPSG:{epsg}"

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf = gdf.dissolve()

    # Export
    gdf.to_file(filepath, driver="GPKG")"""

PACKAGE_NAME = "cosmotracer"
      
class ModelCache():
    
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
        
        if self.fail_on_overwrite and os.path.exists(cache_filepath):
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

class ScalingCache():
    
    def __init__(
        self,
        allow_cache : bool = True,
        round_level : int = 1,
        scaling_param_key : str = "He"
    ):

        # Global options for this Cache instance
        self.cache_allowed = allow_cache
        self.round_level = round_level
        
        # Initialise the cache if it is allowed and does not exist...
        if self.cache_allowed:
            self.cache_dir : Path = platformdirs.user_cache_path(PACKAGE_NAME)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # make the json file if it doesn't exist
            fname = f"{scaling_param_key}_roundlvl{self.round_level}.json"
            self.filepath = self.cache_dir / fname
            
            if not os.path.isfile(self.filepath):
                with open(self.filepath,"w") as f:
                    ujson.dump({},f)
                    
            # read the cache...
    
    def _set_cachekey(
        self,
        lat, lon, elev
    ):
        self.cachekey = str((
            round(lat, self.round_level), 
            round(lon, self.round_level), 
            round(elev, self.round_level)
        ))
    
    def load_cache(
        self
    ):
        if not self.cache_allowed:
            self.cache = {}
        else: 
            try:
                with open(self.filepath, "r") as f:
                    self.cache = ujson.load(f)
            except ujson.JSONDecodeError: # empty file...
                self.cache = {}
    
    def save_cache(
        self
    ):
        if not self.cache_allowed:
            return None 
        
        with open(self.filepath, "w") as f:
            ujson.dump(self.cache, f)
        
    def get_cache_value(
        self,
        lat,
        lon,
        elev
    ):
        
        # get filekey
        self._set_cachekey(lat, lon, elev)
        
        # try to find key in cache
        try:
            return self.cache[self.cachekey]
        except:
            return None

        
    def set_cache_value(
        self,
        lat,
        lon,
        elev,
        value
    ):
        if not self.cache_allowed:
            return None
        
        self._set_cachekey(lat, lon, elev)
        self.cache[self.cachekey] = value