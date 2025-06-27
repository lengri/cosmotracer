import pandas as pd 
import os
import numpy as np
import utm

_UTM_ZONES_WGS84 = pd.DataFrame({
    "ZoneNumber": list(range(1,61)) + list(range(1,61)),
    "Hemisphere": ["N"]*60 + ["S"]*60,
    "EPSG": list(range(32601, 32661)) + list(range(32701, 32761))
})

def convert_EPSG_to_UTM_zone(epsg):

    id = np.where(_UTM_ZONES_WGS84["EPSG"]==epsg)[0]
    
    if len(id) != 1:
        raise Exception(f"EPSG code {epsg} not found! Make sure your EPSG code is WGS84-referenced!")
    
    zone = _UTM_ZONES_WGS84["ZoneNumber"].iloc[id].item()
    hem = _UTM_ZONES_WGS84["Hemisphere"].iloc[id].item()
    
    return (zone, hem)

def convert_xy_to_latlon(
    x : np.ndarray,
    y : np.ndarray,
    epsg : int
):
    zn, hem = convert_EPSG_to_UTM_zone(epsg)
    is_northern = True if hem == "N" else False
    
    lat, lon = utm.to_latlon(x, y, zn, northern=is_northern)
    
    return (lat, lon)
    
    
if __name__ == "__main__":
    out = convert_EPSG_to_UTM_zone(32611)
    print(out)
    