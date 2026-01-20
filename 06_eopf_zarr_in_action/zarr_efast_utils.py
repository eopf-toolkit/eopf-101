import gc
from pathlib import Path

import numpy as np
from pyresample import geometry, kd_tree
from tqdm import tqdm
import xarray as xr

from zarr_wf_utils import validate_scl


def s2_preprocess(s2_urls, bands, resolution, output_dir):
    """
    Pre-process Sentinel-2:
    - Load from Zarr
    - Select bands at given resolution
    - Apply SCL-based cloud mask
    - Save masked bands as GeoTIFF
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, s2_url in enumerate(tqdm(s2_urls)):
        s2_zarr = xr.open_datatree(
            s2_url, engine="zarr", chunks={}, decode_timedelta=False
        )
        zarr_meas = s2_zarr.measurements.reflectance[f"r{resolution}m"]
        l2a_class = s2_zarr.conditions.mask.l2a_classification[f"r{resolution}m"].scl
        valid_mask = validate_scl(l2a_class)
        band_data = zarr_meas.to_dataset()[bands]
        band_data = xr.where(valid_mask, band_data, 0)
        s2_crs = s2_zarr.attrs["other_metadata"]["horizontal_CRS_code"]
        band_data.rio.write_crs(s2_crs, inplace=True)

        # Export as geotif
        out_path = output_dir / (Path(s2_url).stem + "_REFL.tif")
        band_data.rio.to_raster(out_path, nodata=0, compress="ZSTD", predictor=1)

    print("Sentinel-2 preprocessing finished.")


def s3_preprocess(
    s3_urls,
    bands,
    search_bbox,
    target_resolution_deg,
    output_dir,
    radius_of_influence=500,
):
    """
    Pre-process Sentinel-3 OLCI scenes:
    - Load swath data from Zarr
    - Select bands
    - Resample to regular lat/lon grid (binning)
    - Apply scale factors
    - Save as GeoTIFF
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define target grid
    width = int((search_bbox[2] - search_bbox[0]) / target_resolution_deg)
    height = int((search_bbox[3] - search_bbox[1]) / target_resolution_deg)
    grid = geometry.AreaDefinition(
        area_id="olci_grid",
        description="OLCI projected grid",
        proj_id="latlon",
        projection="EPSG:4326",
        width=width,
        height=height,
        area_extent=search_bbox,
    )
    # Build coordinates
    x_res = (grid.area_extent[2] - grid.area_extent[0]) / grid.width
    y_res = (grid.area_extent[3] - grid.area_extent[1]) / grid.height
    x_coords = np.linspace(
        grid.area_extent[0] + x_res / 2, grid.area_extent[2] - x_res / 2, grid.width
    )
    y_coords = np.linspace(
        grid.area_extent[3] - y_res / 2,
        grid.area_extent[1] + y_res / 2,
        grid.height,
    )
    
    for i, s3_url in enumerate(tqdm(s3_urls)):
        s3_zarr = xr.open_datatree(s3_url, engine="zarr")
        band_data_s3 = s3_zarr.measurements.to_dataset()[bands]

        # Define swath geometry
        s3_swath = geometry.SwathDefinition(
            lons=band_data_s3["longitude"].values * 1_000_000,
            lats=band_data_s3["latitude"].values * 1_000_000,
        )
        
        # Resample (binning)
        resampled = kd_tree.resample_nearest(
            s3_swath,
            np.stack([band_data_s3[band].values for band in bands], axis=2),
            grid,
            radius_of_influence=radius_of_influence,
            fill_value=np.nan,
        )
        # Apply scale factors
        #for bi, band in enumerate(bands):
        #    scale = band_data_s3[band].attrs["_eopf_attrs"]["scale_factor"]
        #    resampled[:, :, bi] *= scale

        # Reorder to (band, y, x) as required by rioxarray
        resampled = np.transpose(resampled, (2, 0, 1))

        # Create DataArray
        band_data_s3_proj = xr.DataArray(
            resampled,
            dims=("band", "y", "x"),
            coords={"band": bands, "y": y_coords, "x": x_coords},
        )
        band_data_s3_proj.rio.write_crs("EPSG:4326", inplace=True)
        # Save to GeoTIFF
        out_path = output_dir / (Path(s3_url).stem + ".tif")
        band_data_s3_proj.rio.to_raster(
            out_path, nodata=0, compress="ZSTD", predictor=2
        )

        # Clean up memory
        s3_zarr.close()
        band_data_s3.close()
        band_data_s3_proj.close()
        resampled = None
        s3_swath = None
        gc.collect()
        
    print("Sentinel-3 preprocessing finished.")