import os
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd

# For caching stuff
import h5py
import numpy as np
import platformdirs
import ujson
import pickle
import sqlite3

# For the watershed export
from landlab import RasterModelGrid
from shapely import Point, box

class CacheNotFoundError(Exception):
    pass

class CacheOverwriteError(Exception):
    pass

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


PACKAGE_NAME = "cosmotracer"
      
class ModelCache():
    
    def __init__(
        self,
        identifier: str,
        allow_cache: bool = True,
        fail_on_overwrite: bool = True,
        
    ):
        
        # Global options for this Cache instance
        self.cache_allowed = allow_cache
        self.fail_on_overwrite = fail_on_overwrite
        
        # Initialise the cache if it is allowed and does not exist...
        if self.cache_allowed:
            self.cache_dir: Path = platformdirs.user_cache_path(PACKAGE_NAME)
            self.cache_dir.mkdir(parents=True, exist_ok=True) 
            
            # check if the h5 file for the current identifier exists:
            self.cache_filepath = self.cache_dir / f"{identifier}.h5"
            if not os.path.isfile(self.cache_filepath):
                with h5py.File(self.cache_filepath, "w") as _:
                    pass  # no datasets/groups needed for now
    
    def _dataset_exists(self, filekey):
        
        with h5py.File(self.cache_filepath, "r") as f: # treat the actual h5 as the primary group
            group = f
            for key, value in filekey.items():
                groupkey = key + f"{value}"
                if groupkey in group:
                    group = group[groupkey]
                else:
                    return False
            return True
        
    def load_file(
        self,
        filekey
    ):
        
        # filekey is a dict containing all the information we need to identify
        # the grid we want to load/save. We check, bit by bit, if the needed datasets
        # exist in our h5 file
        # do nothing if there is no caching allowed
        if not self.cache_allowed:
            return None

        if not self._dataset_exists(filekey=filekey):
            raise CacheNotFoundError(
                f"Could not find dataset with key {filekey} "
                f"in {self.cache_filepath}"
            )
        
        with h5py.File(self.cache_filepath, "r") as f: # treat the actual h5 as the primary group
            group = f
            for key, value in filekey.items():
                groupkey = key + f"{value}"
                group = group[groupkey]
                
            # load data from final group
            return group["dataset"][:]
    
    def save_file(
        self, filekey, data
    ):

        # do nothing here as well
        if not self.cache_allowed:
            return None
        
        # we have to check if the file exists in "w" mode, otherwise we will overwrite existing info!
        if self._dataset_exists(filekey=filekey) and self.fail_on_overwrite:
            raise CacheOverwriteError(
                f"Saving dataset for current filekey "
                f"would overwrite existing dataset in {self.cache_filepath}"
            )
        
        with h5py.File(self.cache_filepath, "a") as f: # treat the actual h5 as the primary group
            group = f
            for key, value in filekey.items():
                groupkey = key + f"{value}"
                
                if not groupkey in group:

                    group = group.create_group(groupkey)
                else:
                    group = group[groupkey]

            group.create_dataset("dataset", shape=data.shape, data=data)

class ScalingCache():
    
    def __init__(
        self,
        allow_cache : bool = True,
        round_level : int = 1,
        nuclide_key : str = "He",
        
    ):

        # Global options for this Cache instance
        self.cache_allowed = allow_cache
        self.round_level = round_level
        
        # Initialise the cache if it is allowed and does not exist...
        if self.cache_allowed:
            self.cache_dir : Path = platformdirs.user_cache_path(PACKAGE_NAME)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # make the json file if it doesn't exist
            fname = f"{nuclide_key}_roundlvl{self.round_level}.json"
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
        
class SQLiteScalingCache:
    
    def __init__(
        self, 
        round_level: int = 1,
        allow_cache: bool = True,
        nuclide_key: str = "He"
    ):
        
        self.cache_allowed = allow_cache
        self.round_level = round_level
        self.conn = None

        if self.cache_allowed:
            # 1. Open the connection and create the table if allowed
            self.cache_dir : Path = platformdirs.user_cache_path(PACKAGE_NAME)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            db_name = f"{nuclide_key}_roundlvl{self.round_level}.db" 
            try:
                self.conn = sqlite3.connect(self.cache_dir / db_name)
                # Ensure the table structure exists
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS scaling_cache (
                        key TEXT PRIMARY KEY,
                        value BLOB
                    )
                ''')
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Error connecting to SQLite database: {e}. Disabling caching.")
                self.cache_allowed = False
                
    def __del__(self):
        # 2. Ensure the connection is closed when the object is destroyed
        if self.conn:
            self.conn.close()

    def _get_cachekey(self, lat: float, lon: float, elev: float) -> str:
        """Generates the unique, rounded string key for the cache."""
        # Use the tuple string as the key
        return str((
            round(lat, self.round_level), 
            round(lon, self.round_level), 
            round(elev, self.round_level)
        ))

    def get_cache_value(self, lat: float, lon: float, elev: float):
        """
        Retrieves a scaling factor array from the cache.
        Returns the array if found, None otherwise.
        """
        if not self.cache_allowed:
            return None
        
        key = self._get_cachekey(lat, lon, elev)
        
        try:
            # Select the BLOB (bytes) from the database
            cursor = self.conn.execute("SELECT value FROM scaling_cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            
            if row is None:
                return None # Not in cache
            
            # Deserialize the bytes back into a NumPy array
            return pickle.loads(row[0])
            
        except sqlite3.Error as e:
            # Handle potential read errors gracefully
            print(f"Error reading from cache: {e}")
            return None

    def set_cache_value(self, lat: float, lon: float, elev: float, array_value: np.ndarray):
        """
        Stores a calculated scaling factor array into the cache.
        """
        if not self.cache_allowed:
            return None
        
        key = self._get_cachekey(lat, lon, elev)
        
        try:
            # Serialize the NumPy array into bytes (BLOB)
            serialized_value = pickle.dumps(array_value)
            
            # UPSERT (UPDATE or INSERT) the key/value pair.
            # ON CONFLICT ensures atomicity and robustness.
            self.conn.execute(
                "INSERT INTO scaling_cache (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, serialized_value)
            )
            
            # Commit immediately to disk for safety against interrupts!
            self.conn.commit()
            
        except sqlite3.Error as e:
            # Handle potential write errors gracefully
            print(f"Error writing to cache: {e}. Attempting rollback.")
            # Rollback to ensure no partial write state
            self.conn.rollback()