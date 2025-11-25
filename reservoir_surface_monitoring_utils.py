# Core & utilities
import math
import numpy as np
import pandas as pd
import xarray as xr                        
from rasterio.enums import Resampling           # Resampling method for reprojection

# Image processing for water detection
from skimage.filters import threshold_otsu      # Thresholding (for NDWI classification)
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

# Helper function to load datatrees
def load_datatrees(path):
    return xr.open_datatree(
        path,
        engine="zarr",
        chunks={},
        decode_timedelta=False
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

def _old_preprocess_datatree(
    dtree,
    minx_utm,
    miny_utm,
    maxx_utm,
    maxy_utm,
    crs_code
):
    """
    Preprocess a single datatree:
    - Clip to UTM bounds
    - Reproject 10m band to 20m
    - Merge bands into one dataset
    - Add time coordinate
    
    Parameters
    ----------
    dtree : xr.DataTree
        Loaded datatree for one scene
    minx_utm, miny_utm, maxx_utm, maxy_utm : float
        Clipping bounds in UTM
    crs_code : int or str
        CRS to assign
    
    Returns
    -------
    xr.Dataset
        Dataset with bands 'green' and 'swir' and a time dimension
    """
    
    # Extract bands
    green_10m = dtree.measurements.reflectance.r10m.ds["b03"]
    swir_20m  = dtree.measurements.reflectance.r20m.ds["b11"]
    
    # Clip to UTM bounds
    green_clip = green_10m.sel(
        x=slice(minx_utm, maxx_utm),
        y=slice(maxy_utm, miny_utm)
    )
    swir_clip = swir_20m.sel(
        x=slice(minx_utm, maxx_utm),
        y=slice(maxy_utm, miny_utm)
    )
    
    # Assign CRS
    green_clip = green_clip.rio.write_crs(crs_code, inplace=False)
    swir_clip = swir_clip.rio.write_crs(crs_code, inplace=False)
    
    # Reproject high-res to match low-res
    green_20m = green_clip.rio.reproject_match(
        swir_clip,
        resampling=Resampling.nearest,
        )
        
    # Merge bands
    ds_merged = xr.Dataset({
        "green": green_20m,
        "swir": swir_clip
    })
    
    # Add time coordinate
    date_str = dtree.attrs["stac_discovery"]["properties"]["start_datetime"]
    date = pd.to_datetime(date_str).date()
    ds_merged = ds_merged.expand_dims(time=[np.datetime64(date)])
    
    return ds_merged

def preprocess_datatree(
    dtree,
    minx_utm,
    miny_utm,
    maxx_utm,
    maxy_utm,
):
    """
    Preprocess a single datatree:
    - Extract relevant bands
    - Clip to UTM bounds
    - Merge bands into one dataset
    - Add time coordinate
    
    Parameters
    ----------
    dtree : xr.DataTree
        Loaded datatree for one scene
    minx_utm, miny_utm, maxx_utm, maxy_utm : float
        Clipping bounds in UTM
    
    Returns
    -------
    xr.Dataset
        Dataset with bands 'green' and 'swir' and a time dimension
    """
    
    # Extract bands
    green_10m = dtree.measurements.reflectance.r10m.ds["b03"]
    nir_10m  = dtree.measurements.reflectance.r10m.ds["b08"]
    
    # Clip to UTM bounds
    green_clip = green_10m.sel(
        x=slice(minx_utm, maxx_utm),
        y=slice(maxy_utm, miny_utm)
    )
    nir_clip = nir_10m.sel(
        x=slice(minx_utm, maxx_utm),
        y=slice(maxy_utm, miny_utm)
    )
        
    # Merge bands
    ds_merged = xr.Dataset({
        "green": green_clip,
        "nir": nir_clip
    })
    
    # Add time coordinate
    date_str = dtree.attrs["stac_discovery"]["properties"]["start_datetime"]
    date = pd.to_datetime(date_str).date()
    ds_merged = ds_merged.expand_dims(time=[np.datetime64(date)])
    
    return ds_merged

def gww(slice_mndwi: np.ndarray, slice_wo: np.ndarray,
        canny_sigma: float = 0.7,
        canny_low: float = 0.5,
        canny_high: float = 1.0,
        nonwater_thresh: float = -0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    edge_image = canny(
        slice_mndwi, 
        sigma=canny_sigma, 
        low_threshold=canny_low, 
        high_threshold=canny_high
        )
    
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
    # p = np.median(wo_flat) if len(wo_flat) > 0 else 0.0
    p = np.percentile(wo_flat, 80) if len(wo_flat) > 0 else 0.0

    # Generate water fill from WO dataset
    water_fill_JRC = slice_wo > p
    water_fill_JRC[nanmask] = False
    
    # Identify non-water pixels from NDWI
    nonwater = slice_mndwi < nonwater_thresh
    
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

def compute_area_km2(mask_2d: np.ndarray, pixel_size: float = 10.0) -> float:
    """Compute area (km²) from a binary mask."""
    return (np.sum(mask_2d) * (pixel_size ** 2)) / 1000000