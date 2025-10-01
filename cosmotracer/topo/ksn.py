import numpy as np
from cosmotracer.utils import linear_regression

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
            par, _ = linear_regression(chi, z)
            slopes[i] = par[1]
        else: 
            slopes[i] = bad_segment_value 
            
    return slopes