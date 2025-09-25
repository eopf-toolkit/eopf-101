"""
Utility functions for EOPF Zarr Wild Fire Workflow.

This module provides helper functions for processing, reprojecting and analysing Sentinel-2 and Sentinel-3 data.
"""

from distributed import LocalCluster
from pystac_client import Client
import numpy as np
import xarray as xr
import time
import matplotlib.pyplot as plt
from pyproj import Transformer
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from skimage import exposure
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import box



def validate_scl(scl):

    """ 
    Creates a boolean mask to identify valid pixels in a Sentinel-2 Scene Classification (SCL) dataset by excluding invalid land cover types.
    0: No Data
    1: Saturated / Defective Pixel 
    3: Cloud Shadows
    7: Low probability Cloud
    8: Medium probability Cloud
    9: High probability Cloud

    Recieves:
    scl (xarray.DataArray): Representing the Scene Classification (SCL) band from a Sentinel-2 image.

    Returns:
    boolean mask for valid pixels

    """
# A list of SCL values to be considered invalid
    invalid = [0, 1, 3, 7, 8, 9]
# Return a boolean mask where True indicates a valid pixel (i.e., not in the invalid list)
    return ~scl.isin(invalid)

# This variation masks water bodies
def validate_scl_w(scl):
    """ 
    Creates a boolean mask to identify valid pixels in a Sentinel-2 Scene Classification (SCL) dataset by excluding invalid land cover types.
    0: No Data
    1: Saturated / Defective Pixel 
    3: Cloud Shadows
    6: Water
    7: Low probability Cloud
    8: Medium probability Cloud
    9: High probability Cloud
    Recieves:
    scl (xarray.DataArray): Representing the Scene Classification (SCL) band from a Sentinel-2 image.

    Returns:
    boolean mask for valid pixels

    """
# A list of SCL values to be considered invalid
    invalid = [0, 1, 3, 6, 7, 8, 9]
# Return a boolean mask where True indicates a valid pixel (i.e., not in the invalid list)
    return ~scl.isin(invalid)



def mask_sub_utm(zarr_asset, rows, cols):

    """
    Performs the masking over a `.zarr` asset by subsetting it to a specified rectangular area defined by row and column indices.

    Recieves:

    - zarr_asset (xarray.DataArray) : The input Zarr array asset.
    - rows (list/array) : Row indices defining the vertical extent.
    - cols (list/array): Column indices defining the horizontal extent.

    Returns:
    boolean mask for bbox defined pixels

    """

# Calculates the minimum and maximum row and column indices
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
# It then subsets the zarr_asset using these indices
    masked_asset = zarr_asset.isel(
    y=slice(row_min, row_max + 1), x=slice(col_min, col_max + 1)
)
    return masked_asset


def normalisation_str_gm(band_array, p_min, p_max, gamma_val):

    """
    Applies percentile-based contrast stretching and gamma correction to a band.

    Recieves:

    - band_array (xarray) : the extracted `xarray` for the selected band
    - p_min (int): percentile min value
    - p_max (int): percentile max value
    - gamma_val (int): gamma correction

    Returns:
    - 2d array with the normalised input values
    """

    # Calculate min and max values based on percentiles for stretching
    min_val = np.percentile(band_array[band_array > 0], p_min) if np.any(band_array > 0) else 0
    max_val = np.percentile(band_array[band_array > 0], p_max) if np.any(band_array > 0) else 1

    # Avoid division by zero if min_val equals max_val
    if max_val == min_val:
        stretched_band = np.zeros_like(band_array, dtype=np.float32)
    else:
        # Linear stretch to 0-1 range
        stretched_band = (band_array - min_val) / (max_val - min_val)

    # Clip values to ensure they are within [0, 1] after stretching
    stretched_band[stretched_band < 0] = 0
    stretched_band[stretched_band > 1] = 1

    # Apply gamma correction
    gamma_corrected_band = np.power(stretched_band, 1.0 / gamma_val)

    # Returns the corrected array:
    return gamma_corrected_band


def lat_lon_to_utm_box(bot_left,top_right,transformer):

    """
    Converts the latitude and longitude coordinates of a bounding box from a geographic system to the UTM (Universal Transverse Mercator) coordinate system.

    Recieves:

    - bot_left (tuple) : A tuple or list containing the longitude and latitude of the bottom-left corner.
    - top_right (tuple) : A tuple or list containing the longitude and latitude of the top-right corner.
    - transformer: The 

    Returns:
    tuple with resulting utm values
    """
    t = transformer
    # Longitude and latitude of the bottom-left corner
    west_utm, south_utm = t.transform(bot_left[0],bot_left[1])
    # Longitude and latitude of the top-right corner
    east_utm, north_utm = t.transform(top_right[0],top_right[1])

    # A new list with the converted UTM coordinates
    return [ west_utm, south_utm , east_utm , north_utm ]



def zarr_mask_utm ( bounding_box, zarr):

    """
    This function creates a boolean mask to identify the rows and columns within a Zarr dataset that fall within a specified bounding box, assuming UTM coordinates.

    Recieves:
    - bounding_box (tuple/list) : A tuple or list containing (min_longitude, min_latitude, max_longitude, max_latitude).
    - zarr (xarray): The input Zarr dataset, expected to have 'x' and 'y' dimensions corresponding to longitude and latitude/UTM coordinates.
    
    Returns: 
    boolean mask for utm bbox defined pixels
    """
# Unpack the bounding box coordinates for clarity
    min_lon, min_lat, max_lon, max_lat = bounding_box
# Create boolean masks for longitude and latitude dimensions based on the bounding box
    lon_mask = (zarr['x'] >= min_lon) & (zarr['x'] <= max_lon)
    lat_mask = (zarr['y'] >= min_lat) & (zarr['y'] <= max_lat)
# Combine the individual masks to create a single bounding box mask
    bbox_mask = lon_mask & lat_mask
# Find the columns and row indices where the combined mask is True
    cols, rows  = np.where(bbox_mask)

    return rows, cols


def mask_sub_latlon(zarr_asset, rows, cols):
    """
    Masks a `.zarr` asset by subsetting it to a specified rectangular area defined by row and column indices. It is intended for datasets where the dimensions are labelled as 'rows' and 'columns'.

    Recieves:

    - zarr_asset (xarray.DataArray): The input `.zarr` array asset.
    - rows (list/array): A list or array of row indices defining the vertical extent.
    - cols (list/array): A list or array of column indices defining the horizontal extent.
    
    Returns:
    boolean mask for lat and lon bbox defined pixels
    """
# Calculates the minimum and maximum row and column indices
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
# It then subsets the zarr_asset using these indices
    masked_asset = zarr_asset.isel(
    rows=slice(row_min, row_max + 1), columns=slice(col_min, col_max + 1)
)
    return masked_asset


def zarr_mask_latlon ( bounding_box, zarr):

    """
    Allows the creation of a boolean mask to identify the rows and columns within a Zarr dataset that fall within a specified bounding box, assuming latitude and longitude dimensions.

    Recieves:

    - bounding_box (tuple): A tuple or list containing (min_longitude, min_latitude, max_longitude, max_latitude).
    - zarr (xarray): The input Zarr dataset, expected to have 'longitude' and 'latitude' dimensions.
    
    Returns:
    boolean mask for lat lon bbox defined pixels 
    """
# Unpack the bounding box coordinates for clarity
    min_lon, min_lat, max_lon, max_lat = bounding_box
# Create boolean masks for longitude and latitude dimensions based on the bounding box
    lon_mask = (zarr['longitude'] >= min_lon) & (zarr['longitude'] <= max_lon)
    lat_mask = (zarr['latitude'] >= min_lat) & (zarr['latitude'] <= max_lat)
# Combine the individual masks to create a single bounding box mask
    bbox_mask = lon_mask & lat_mask
# Find the row and column indices where the combined mask is True
    cols, rows  = np.where(bbox_mask)

    return rows, cols

