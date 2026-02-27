"""
Script Name: zarr_s1_utils.py
Author: Walid Ghariani
Date: 2026-01-09
Description: Set of code to process Sentinel-1 data stored in Zarr format for coastal water dynamics analysis.
Version: 0.0.1 - Support for Sentinel-1 Zarr data processing.
"""
from typing import Tuple, NamedTuple

import numpy as np
import xarray as xr
import rioxarray
from odc.geo.geobox import GeoBox
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu


def aoi_slices(
    gcp_ds: xr.Dataset,
    aoi_bounds: list[float] | tuple[float, float, float, float],
    offset: int = 2,
) -> dict[str, slice]:
    """
    Get the azimuth_time and ground_range slices for cropping around an AOI.

    Parameters
    ----------
    gcp_ds : xarray.Dataset
        Dataset with 'latitude' and 'longitude' variables.
    aoi_bounds : list or tuple
        [min_lon, min_lat, max_lon, max_lat]
    offset : int
        Number of GCP grid cells to include around the nearest point.

    Returns
    -------
    dict
        {"azimuth_time": az_slice, "ground_range": gr_slice}
    """
    min_lon, min_lat, max_lon, max_lat = aoi_bounds
    lat_c = (min_lat + max_lat) / 2
    lon_c = (min_lon + max_lon) / 2

    dist = (gcp_ds.latitude - lat_c) ** 2 + (gcp_ds.longitude - lon_c) ** 2

    flat_index = dist.argmin().values
    i, j = np.unravel_index(flat_index, gcp_ds.latitude.shape)

    def clamp(index, dim):
        start = max(0, index - offset)
        end = min(dim - 1, index + offset)
        return slice(start, end + 1)

    az_slice = clamp(i, gcp_ds.sizes["azimuth_time"])
    gr_slice = clamp(j, gcp_ds.sizes["ground_range"])

    return {"azimuth_time": az_slice, "ground_range": gr_slice}


def subset(
    grd: xr.Dataset,
    gcp_ds: xr.Dataset,
    aoi_bounds: list[float] | tuple[float, float, float, float],
    offset=2,
) -> xr.Dataset:
    """
    Crop GRD to AOI and return the subset.

    Parameters
    ----------
    grd : xarray.DataArray or Dataset
        GRD data to crop.
    gcp_ds : xarray.Dataset
        GCP dataset for slicing.
    aoi_bounds : list or tuple
        [min_lon, min_lat, max_lon, max_lat]
    offset : int
        Number of GCP grid cells to include around the nearest point.

    Returns
    -------
    xarray.DataArray or Dataset
        Cropped and masked GRD subset.
    """
    slices = aoi_slices(gcp_ds, aoi_bounds, offset)
    gcp_crop = gcp_ds.isel(**slices)

    az_min, az_max = (
        gcp_crop.azimuth_time.min().values,
        gcp_crop.azimuth_time.max().values,
    )
    gr_min, gr_max = (
        gcp_crop.ground_range.min().values,
        gcp_crop.ground_range.max().values,
    )
    grd_crop = grd.sel(
        azimuth_time=slice(az_min, az_max),
        ground_range=slice(gr_min, gr_max),
    )

    gcp_interp = gcp_crop.interp_like(grd_crop)
    grd_crop = grd_crop.assign_coords(
        latitude=gcp_interp.latitude,
        longitude=gcp_interp.longitude,
    )

    minx, miny, maxx, maxy = aoi_bounds
    mask = (
        (grd_crop.latitude >= miny)
        & (grd_crop.latitude <= maxy)
        & (grd_crop.longitude >= minx)
        & (grd_crop.longitude <= maxx)
    )
    mask = mask.compute()

    return grd_crop.where(mask, drop=True)


def radiometric_calibration(
    grd: xr.DataArray,
    calibration_ds: xr.Dataset,
    calibration_type="sigma_nought",
):
    """
    Perform radiometric calibration on a grd data array.

    Parameters:
    -----------
    grd : xarray.DataArray
        The data array to calibrate.
    calibration_ds : xarray.Dataset
        The calibration dataset.
    calibration_type : str, optional
        The name of the calibration type in the calibration dataset.
        Default is "sigma_nought".

    Returns:
    --------
    xarray.DataArray
        The calibrated GRD data.
    """
    calibration_matrix = calibration_ds.interp_like(grd)
    return (grd / calibration_matrix[calibration_type]) ** 2


def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Numpy-based Lee filter for a single 2D array.
    (Internal helper used by lee_filter_dask)
    Adapted from reference: https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
    """
    img = img.astype(np.float32)
    mask_valid = np.isfinite(img)
    img_filled = np.where(mask_valid, img, 0)

    img_mean = uniform_filter(img_filled, size)
    img_sqr_mean = uniform_filter(img_filled**2, size)
    img_variance = img_sqr_mean - img_mean**2
    img_variance = np.maximum(img_variance, 0)

    valid_pixels = img[mask_valid]
    overall_variance = np.var(valid_pixels) if valid_pixels.size > 0 else 0.0

    img_weights = img_variance / (img_variance + overall_variance + 1e-6)
    img_output = img_mean + img_weights * (img_filled - img_mean)
    img_output[~mask_valid] = np.nan
    return img_output


def lee_filter_dask(da: xr.DataArray, size: int = 5) -> xr.DataArray:
    """
    Apply a Lee speckle filter to an xarray.DataArray (Dask-compatible).

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray (float32 preferred). May be Dask-backed.
    size : int
        Window size (odd number recommended)

    Returns
    -------
    xr.DataArray
        Filtered DataArray (Dask-backed if input was Dask-backed)
    """
    filtered = xr.apply_ufunc(
        lee_filter,
        da,
        kwargs={"size": size},
        input_core_dims=[["azimuth_time", "ground_range"]],
        output_core_dims=[["azimuth_time", "ground_range"]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    filtered.attrs.update(da.attrs)
    filtered.attrs.update(attrs={"speckle filter method": "lee"})

    return filtered


class GridInfo(NamedTuple):
    """Container for grid information."""

    gbox: GeoBox
    xs: np.ndarray  # -> 1D longitude array
    ys: np.ndarray  # -> 1D latitude array
    xg: np.ndarray  # -> 2D meshgrid longitude
    yg: np.ndarray  # -> 2D meshgrid latitude


def _extract_valid_data(
    sar_da: xr.DataArray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts valid longitude, latitude, and data values from a DataArray.
    """
    lon = sar_da.longitude.values.flatten()
    lat = sar_da.latitude.values.flatten()
    data = sar_da.values.flatten()
    mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(data)
    return lon[mask], lat[mask], data[mask]


def _build_geobox(
    bounds: list[float] | Tuple[float, float, float, float],
    resolution: Tuple[float, float],
    crs: str = "EPSG:4326",
) -> GridInfo:
    """
    Creates a GeoBox and returns grid information for interpolation.
    """
    gbox = GeoBox.from_bbox(bounds, crs=crs, resolution=resolution)
    coords = gbox.coordinates
    xs = coords["longitude"].values
    ys = coords["latitude"].values
    xg, yg = np.meshgrid(xs, ys)
    return GridInfo(gbox, xs, ys, xg, yg)


def _interpolate_to_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    data: np.ndarray,
    grid_info: GridInfo,
    method: str = "nearest",
) -> xr.DataArray:
    """
    Interpolates scattered data onto a regular grid and returns a DataArray with CRS and GeoBox.
    """
    data_interp = griddata(
        (lon, lat), data, (grid_info.xg, grid_info.yg), method=method
    )
    da = xr.DataArray(
        data_interp,
        dims=("y", "x"),
        coords={"x": grid_info.xs, "y": grid_info.ys},
        attrs={"interpolation": method},
    )
    da = da.rio.write_crs(grid_info.gbox.crs)
    da = da.rio.write_transform(grid_info.gbox.transform)
    da.attrs["odcgeobox"] = grid_info.gbox
    return da


def regrid(
    da: xr.DataArray,
    bounds: list[float] | Tuple[float, float, float, float],
    resolution: Tuple[float, float],
    crs: str = "EPSG:4326",
    method: str = "nearest",
) -> xr.DataArray:
    """
    Regrids DataArray to a regular grid using ODC GeoBox and scipy griddata.
    """
    lon, lat, data = _extract_valid_data(da)
    grid_info = _build_geobox(bounds, resolution, crs)
    da_out = _interpolate_to_grid(lon, lat, data, grid_info, method)
    return da_out


def xr_threshold_otsu(
    da: xr.DataArray,
    mask_nan: bool = True,
    return_threshold: bool = False,
    mask_name: str = None,
) -> xr.DataArray:
    """
    Compute Otsu's threshold and generate a binary mask from a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with values to threshold.
    mask_nan : bool, optional
        If True, mask NaN values before thresholding.
    return_threshold : bool, optional
        If True, return the threshold value as an attribute.
    mask_name : str, optional
        Name for the binary mask DataArray (useful for metadata).

    Returns
    -------
    xr.DataArray
        A binary mask DataArray with the same spatial metadata as the input.
        If return_threshold is True, the threshold value is stored in attrs.
    """
    values = da.values
    if mask_nan:
        values = values[~np.isnan(values)]

    threshold = threshold_otsu(values)
    mask = (da > threshold).astype(np.uint8)

    mask_da = xr.DataArray(
        mask,
        dims=da.dims,
        coords=da.coords,
        name=mask_name,
        attrs={"threshold": threshold} if return_threshold else {},
    )
    return mask_da