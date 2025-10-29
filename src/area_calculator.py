# Compute water area given a classified map (no file saving)

import numpy as np

def compute_water_area_km2(classified_map, pixel_resolution_m=30.0):
    """
    classified_map: 2D numpy int array where water is labeled as 1
    pixel_resolution_m: pixel size in meters (default 30 for Landsat)
    returns water_area_km2 (float)
    """
    water_pixels = np.sum(classified_map == 1)
    pixel_area_m2 = float(pixel_resolution_m) * float(pixel_resolution_m)
    water_m2 = water_pixels * pixel_area_m2
    water_km2 = water_m2 / 1e6
    return water_km2
