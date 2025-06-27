import numpy as np
import os 
import geopandas as gpd
from shapely import Point

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