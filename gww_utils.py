# Core & utilities
import math
import numpy as np
import pandas as pd
from datetime import datetime
import xarray as xr                        

# Image processing for water detection
from skimage.filters import threshold_otsu      # Thresholding (for MNDWI classification)
from skimage.feature import canny               # Edge detection
from skimage.morphology import dilation         # Edge dilation
from skimage.measure import label               # Connected components labeling

# Helper function to collect the found scenes
def list_found_elements(search_result):
    ids = []
    coll = []
    for item in search_result.items():
        ids.append(item.id)
        coll.append(item.collection_id)
    return ids, coll

# helper function to load datasets
def load_dataset(path, band_codes):
    return xr.open_dataset(
        path,
        engine="eopf-zarr",
        chunks={},
        variables=band_codes,
        resolution=20,
    )

# helper function to find the utm zone
def get_utm_zone(lon: float) -> int:
    return math.ceil((180 + lon) / 6)

# Helper function to subset and rename bands, and add time dimension
def preprocess_dataset(ds, minx, maxx, miny, maxy, band_map):
    # Subset to relevant bands and rename
    ds = ds[list(band_map.keys())].rename(band_map)

    # Subset spatially
    ds_subset = ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny)  # reversed if y decreases
    )

    # Extract time from attributes and add as time coordinate
    date_str = ds.attrs.get("mean_sensing_time", None)
    date = pd.to_datetime(date_str).date()

    ds_subset = ds_subset.expand_dims(time=[np.datetime64(date)])

    return ds_subset

def gww(slice_mndwi: np.ndarray, slice_wo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute water mask, water fill, and total water mask for a single 2D slice.
    
    Parameters:
        slice_mndwi: 2D numpy array of MNDWI values
        slice_wo: 2D numpy array of auxiliary water occurrence or water probability
    
    Returns:
        water: sure water mask
        water_fill: water filled from WO dataset
        total_water: combination of both
    """
    # Mask of invalid pixels
    nanmask = np.isnan(slice_mndwi)
    
    # Edge detection
    edge_image = canny(slice_mndwi, sigma=0.7, low_threshold=0.5, high_threshold=1)
    
    # Dilate edges
    dilated = dilation(edge_image, footprint=np.ones((3, 3)))
    
    # Mask dilated edges with NaNs
    dilated = np.ma.array(dilated, mask=nanmask, fill_value=np.nan)
    
    # Mask MNDWI values using dilated edges
    mndwi_edge = np.ma.array(slice_mndwi, mask=np.logical_or(nanmask, ~dilated), fill_value=np.nan)
    
    # Flatten for Otsu threshold
    flat = mndwi_edge[~mndwi_edge.mask]
    flat = flat[~np.isnan(flat)]
    
    # Compute Otsu threshold
    th = threshold_otsu(flat) if len(flat) > 0 else 0.0
    
    # Generate sure water mask
    water = slice_mndwi > th
    water[nanmask] = False
    
    # Mask WO values with same edge mask
    wo_edge = np.ma.array(slice_wo, mask=np.logical_or(nanmask, ~dilated), fill_value=np.nan)
    wo_flat = wo_edge[~wo_edge.mask]
    wo_flat = wo_flat[~np.isnan(wo_flat)]
    
    # Compute threshold from WO dataset (median)
    p = np.median(wo_flat) if len(wo_flat) > 0 else 0.0
    
    # Generate water fill from WO dataset
    water_fill_JRC = slice_wo > p
    water_fill_JRC[nanmask] = False
    
    # Identify non-water pixels from MNDWI
    nonwater = slice_mndwi < -0.15
    
    # Final filled water mask
    water_fill = np.logical_and(nonwater, water_fill_JRC)
    
    # Combine sure water and filled water
    total_water = np.logical_or(water, water_fill)
    
    return water, water_fill, total_water

# function to extract largest connected component
def largest_connected_component(segmentation):
    """Extract the largest connected component from a binary segmentation mask."""
    labels = label(segmentation, connectivity=2)
    largest_cc = labels == np.argmax(np.bincount(labels[segmentation]))
    return largest_cc

def compute_area_km2(mask_2d: np.ndarray, pixel_size: float = 20.0) -> float:
    """Compute area (km²) from a binary mask."""
    return (np.sum(mask_2d) * (pixel_size ** 2)) / 1000000