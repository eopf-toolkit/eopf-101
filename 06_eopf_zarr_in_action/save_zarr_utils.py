# ===---------------------------------------------------------------------=== #
#    Copyright © 2026, Geomatys, SAS. All rights reserved.
#    http://www.geomatys.com
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you
#    may not use this file except in compliance with the License. You may
#    obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#    or implied. See the License for the specific language governing
#    permissions and limitations under the License.
# ===---------------------------------------------------------------------=== #

"""This is an adaptation of the saving part of the  `io` module of biophysical.

It contains I/O functions for saving biophysical processing results to GeoTIFF
and Zarr.

The Zarr output (``prep_zarr``) conforms to the Climate and Forecast (CF)
Metadata Conventions v1.13, and it is adaptation of the original ``save_zarr``
(https://cfconventions.org/cf-conventions/cf-conventions.html).

"""

from __future__ import annotations

__author__: str = "David Meaux"
__version__: str = "1.0.0"


from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS
from rasterio.transform import Affine as RasterioAffine

from biophysicalop.models import Sentinel2ProductIdentifier
from biophysicalop.xaffine import Affine

from biophysicalop.io import (_resolve_spatial_info, _build_variable_cf_attrs, _collect_mask_layers,
                            _build_cf_global_attrs)

if TYPE_CHECKING:
    from biophysicalop.processing import BiophysicalOpProcess


## adaption from save_zarr

def prep_zarr(
    op_process: BiophysicalOpProcess,
    dt: xr.DataTree,
    output_path: str | Path,
    reflectance_group_path: str,
    provenance_info: ProvenanceInfo | None = None,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
) -> Path:
    """Save the biophysical processing result as a CF-1.13 compliant Zarr dataset.

    The primary output is stored as a variable named after the biophysical
    variable label (e.g. ``"LAI"``).  Validation masks are stored as
    separate variables.  The dataset is structured to meet CF-1.13 conventions:

    - Global attributes include ``Conventions``, ``title``, ``source``,
      ``history``, ``references``, and product provenance fields.
    - A scalar ``crs`` variable holds the full CRS description (WKT,
      EPSG code, and CF grid-mapping parameters).
    - Every spatial variable carries a ``grid_mapping="crs"`` attribute.
    - Projected coordinate variables (``x``, ``y``) carry ``standard_name``,
      ``long_name``, ``units``, and ``axis`` attributes.
    - Data variables carry ``long_name``, ``units``, ``valid_min``,
      ``valid_max`` (or ``flag_values`` / ``flag_meanings`` for masks).

    Parameters
    ----------
    op_process : BiophysicalOpProcess
        The completed biophysical processing result.
    dt : xr.DataTree
        The source Sentinel-2 DataTree (used for spatial reference).
    output_path : str or Path
        Destination directory path for the Zarr store.
    reflectance_group_path : str
        Path to the reflectance group in the DataTree, e.g.
        ``"measurements/reflectance/r20m"``.
    provenance_info : ProvenanceInfo or None, optional
        STAC search context for CF history metadata.  When ``None`` only the
        processing step is recorded in the ``history`` attribute.

    Returns
    -------
    Path
        The path to the written Zarr store.
    """
    output_path = Path(output_path)
    result = op_process.result

    if result.data is None:
        raise ValueError("Result data is None; nothing to save.")

    transform, crs, height, width, product_id = _resolve_spatial_info(
        dt, reflectance_group_path
    )

    processing_time = datetime.now(timezone.utc)

    # Get coordinate arrays from the DataTree
    ref_group = dt[reflectance_group_path]
    ref_var_name = next(iter(ref_group.data_vars))
    ref_da = ref_group[ref_var_name]
    y_coords = ref_da.coords["y"].values
    x_coords = ref_da.coords["x"].values

    # Apply optional AOI spatial slicing to coordinates and spatial metadata
    if y_slice is not None or x_slice is not None:
        _ys = y_slice if y_slice is not None else slice(None)
        _xs = x_slice if x_slice is not None else slice(None)
        y_coords = y_coords[_ys]
        x_coords = x_coords[_xs]
        height = len(y_coords)
        width = len(x_coords)
        res_x = float(x_coords[1] - x_coords[0]) if width > 1 else float(transform.a)
        res_y = float(y_coords[0] - y_coords[1]) if height > 1 else float(-transform.e)
        transform = RasterioAffine(
            res_x,
            0.0,
            float(x_coords[0]) - res_x / 2,
            0.0,
            -res_y,
            float(y_coords[0]) + res_y / 2,
        )

    op_label = op_process.biophysical_op.value.label
    op_unit = op_process.biophysical_op.value.unit

    # Primary data variable
    data = np.asarray(result.data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(height, width)

    data_vars: dict[str, tuple] = {
        op_label: (
            ["y", "x"],
            data,
            _build_variable_cf_attrs(op_label, op_label, op_unit),
        ),
    }

    # Mask variables
    mask_layers = _collect_mask_layers(op_process)
    for name, mask_arr in mask_layers:
        data_vars[name] = (
            ["y", "x"],
            mask_arr.astype(np.uint8),
            _build_variable_cf_attrs(name, op_label, op_unit),
        )

    # CF-compliant coordinate attributes
    coord_x = xr.DataArray(
        x_coords,
        dims=["x"],
        attrs={
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection",
            "units": "m",
            "axis": "X",
        },
    )
    coord_y = xr.DataArray(
        y_coords,
        dims=["y"],
        attrs={
            "standard_name": "projection_y_coordinate",
            "long_name": "y coordinate of projection",
            "units": "m",
            "axis": "Y",
        },
    )

    ds = xr.Dataset(
        data_vars={
            k: xr.DataArray(v[1], dims=v[0], attrs=v[2]) for k, v in data_vars.items()
        },
        coords={"y": coord_y, "x": coord_x},
        attrs=_build_cf_global_attrs(
            op_process, product_id, crs, provenance_info, processing_time
        ),
    )

    # --- CRS grid-mapping variable (CF-1.13 §5.6) ---
    # pyproj.CRS.to_cf() returns a full dict of CF grid-mapping parameters.
    crs_cf_attrs = crs.to_cf()
    # Always include the full WKT for unambiguous round-trip decoding
    crs_cf_attrs["crs_wkt"] = crs.to_wkt()
    crs_cf_attrs["epsg_code"] = f"EPSG:{crs.to_epsg()}"
    crs_cf_attrs["spatial_ref"] = crs.to_wkt()  # GDAL-compatible alias

    ds = ds.assign(
        crs=xr.DataArray(
            np.int32(crs.to_epsg()),
            attrs=crs_cf_attrs,
        )
    )

    # Also preserve the transform and legacy CRS fields for tooling that
    # relies on them (e.g. rioxarray).
    ds.attrs["crs_epsg"] = crs.to_epsg()
    ds.attrs["crs_wkt"] = crs.to_wkt()
    ds.attrs["transform"] = [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]

    return ds