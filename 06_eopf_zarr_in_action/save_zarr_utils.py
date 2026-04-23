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


Both functions accept an optional :class:`ProvenanceInfo` argument that embeds
a full step-by-step ``history`` attribute describing how the source data were
retrieved and processed.
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

if TYPE_CHECKING:
    from biophysicalop.processing import BiophysicalOpProcess


# ---------------------------------------------------------------------------
# Provenance tracking
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceInfo:
    """Provenance information for CF-compliant metadata history tracking.

    Construct this object immediately after opening the source DataTree so
    that ``access_time`` accurately records when the data were accessed.  All
    fields except ``access_time`` must be supplied explicitly because they
    are not embedded in the DataTree itself.

    Attributes
    ----------
    stac_api_endpoint : str
        URL of the STAC API endpoint used to discover the product, e.g.
        ``"https://stac.core.eopf.eodc.eu/"``.
    stac_collection : str
        STAC collection name, e.g. ``"sentinel-2-l2a"``.
    product_href : str
        Full URL or path to the source product Zarr store.
    bbox : tuple[float, float, float, float]
        Area-of-interest bounding box ``(minx, miny, maxx, maxy)`` in
        WGS84 geographic coordinates (degrees).
    temporal_range : str
        ISO 8601 temporal range used for the STAC search, e.g.
        ``"2025-07-25T00:00:00Z/2025-07-25T23:59:59Z"``.
    cloud_cover_max : float
        Maximum cloud cover percentage used as STAC filter, e.g. ``35.0``.
    spatial_resolution_meters : int
        Processing spatial resolution in metres; must be 10 or 20.
    access_time : datetime
        UTC datetime when the source data were accessed (i.e. when
        ``xr.open_datatree`` was called).  Defaults to the current UTC time
        at construction — create this object immediately after opening the
        DataTree so the timestamp is accurate.
    """

    stac_api_endpoint: str
    stac_collection: str
    product_href: str
    bbox: tuple[float, float, float, float]
    temporal_range: str
    cloud_cover_max: float
    spatial_resolution_meters: int
    access_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_sensing_time(datatake_sensing_start_time: str) -> str:
    """Convert ``YYYYMMDDTHHMMSS`` to ISO 8601 UTC string."""
    dt = datetime.strptime(datatake_sensing_start_time, "%Y%m%dT%H%M%S")
    return dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def epsg_from_tile_id(tile_id: str) -> int:
    """Derive the EPSG code from a Sentinel-2 MGRS tile identifier.

    The tile ID format is ``Tzzaaa`` where ``zz`` is the UTM zone number
    (01–60) and the first letter of ``aaa`` indicates the latitude band.
    Bands C–M are in the southern hemisphere (EPSG 327xx) and bands N–X
    are in the northern hemisphere (EPSG 326xx).

    Parameters
    ----------
    tile_id : str
        Sentinel-2 tile identifier, e.g. ``"T31TDJ"``.

    Returns
    -------
    int
        EPSG code (e.g. 32631 for UTM zone 31N).
    """
    # Strip leading 'T' if present
    mgrs = tile_id.lstrip("T")
    zone = int(mgrs[:2])
    latitude_band = mgrs[2].upper()
    # Latitude bands C-M are south, N-X are north
    if latitude_band <= "M":
        epsg = 32700 + zone
    else:
        epsg = 32600 + zone
    # Validate via pyproj
    CRS.from_epsg(epsg)
    return epsg


def _to_rasterio_transform(affine: Affine) -> RasterioAffine:
    """Convert a custom ``xaffine.Affine`` to a ``rasterio.transform.Affine``."""
    return RasterioAffine(
        float(affine.a),
        float(affine.b),
        float(affine.c),
        float(affine.d),
        float(affine.e),
        float(affine.f),
    )


def _collect_mask_layers(
    op_process: BiophysicalOpProcess,
) -> list[tuple[str, NDArray]]:
    """Return a list of ``(name, 2d_array)`` for each non-None mask in the result."""
    result = op_process.result
    layers: list[tuple[str, NDArray]] = []

    mask_fields = (
        ("input_out_of_range", result.input_out_of_range_mask),
        ("output_set_to_min", result.output_threshold_set_to_min_output_mask),
        ("output_set_to_max", result.output_threshold_set_to_max_output_mask),
        ("output_too_low", result.output_too_low_mask),
        ("output_too_high", result.output_too_high_mask),
    )
    for name, mask in mask_fields:
        if mask is None:
            continue
        arr = np.asarray(mask)
        # input_out_of_range_mask may be 3D (n_bands, H, W); collapse to 2D
        if arr.ndim == 3:
            arr = arr.any(axis=0)
        layers.append((name, arr.astype(np.uint8)))

    return layers


def _get_metadata_tags(op_process: BiophysicalOpProcess) -> dict[str, str]:
    """Build a dict of metadata tags from the process attributes."""
    sensor = op_process.sensor
    biophysical_op = op_process.biophysical_op
    return {
        "sensor": sensor.name,
        "sensor_type": sensor.value.sensor_type.value,
        "spatial_resolution_meters": str(sensor.value.spatial_resolution_meters),
        "bands": ", ".join(sensor.value.bands),
        "biophysical_variable": biophysical_op.value.label,
        "biophysical_variable_description": biophysical_op.value.description,
        "biophysical_variable_unit": biophysical_op.value.unit,
    }


def _resolve_spatial_info(
    dt: xr.DataTree,
    reflectance_group_path: str,
) -> tuple[RasterioAffine, CRS, int, int, Sentinel2ProductIdentifier]:
    """Extract affine transform, CRS, height, width and product ID from the DataTree.

    Returns
    -------
    tuple
        (rasterio_transform, crs, height, width, product_id)
    """
    ref_group = dt[reflectance_group_path]
    # Pick the first data variable to derive coordinates
    ref_var_name = next(iter(ref_group.data_vars))
    ref_da = ref_group[ref_var_name]

    affine = Affine.from_dataarray(ref_da)
    transform = _to_rasterio_transform(affine)
    height, width = ref_da.sizes["y"], ref_da.sizes["x"]

    # Derive CRS from the product filename embedded in dt.encoding["source"].
    # The source URL/path always contains the Sentinel-2 product name, from
    # which the MGRS tile ID (e.g. "T31TDJ") can be reliably parsed.
    from biophysicalop.utils import parse_sentinel2_product_name

    source = dt.encoding.get("source", "")
    if not source:
        raise ValueError(
            "DataTree has no 'source' encoding entry. "
            "Cannot determine CRS from the product filename."
        )
    product_id = parse_sentinel2_product_name(Path(source))
    epsg = epsg_from_tile_id(product_id.tile_id)

    crs = CRS.from_epsg(epsg)
    return transform, crs, height, width, product_id


# ---------------------------------------------------------------------------
# CF metadata builders (public so callers can inspect or extend)
# ---------------------------------------------------------------------------


def build_history(
    op_process: BiophysicalOpProcess,
    product_id: Sentinel2ProductIdentifier,
    provenance: ProvenanceInfo | None,
    processing_time: datetime,
) -> str:
    """Build the CF-conventions ``history`` global attribute string.

    Produces one or two timestamped entries:

    1. **Data retrieval** (only when *provenance* is not ``None``): records
       the STAC endpoint, collection, search parameters (bounding box,
       temporal range, cloud cover filter), the selected product, and the
       source URL so that a third party can re-download the exact same input.
    2. **Processing**: records the biophysical variable computed, sensor,
       spatial resolution, input bands, angle computation method, neural
       network algorithm, and validation steps applied.

    Parameters
    ----------
    op_process : BiophysicalOpProcess
        Completed biophysical processing object (sensor / operation metadata).
    product_id : Sentinel2ProductIdentifier
        Parsed Sentinel-2 product identifier (used for the clean product name).
    provenance : ProvenanceInfo or None
        STAC search context.  When ``None`` only the processing step is
        included in the history string.
    processing_time : datetime
        UTC datetime to stamp the processing step (typically ``datetime.now``
        captured just before calling ``save_geotiff`` / ``save_zarr``).

    Returns
    -------
    str
        Newline-separated CF history string.
    """
    lines: list[str] = []

    # --- Step 1: Data retrieval ---
    if provenance is not None:
        ts = provenance.access_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        bbox_str = "[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*provenance.bbox)
        product_name = Path(provenance.product_href).stem
        lines.append(
            f"{ts}: Sentinel-2 L2A product retrieved from EOPF STAC API. "
            f"Endpoint: {provenance.stac_api_endpoint}. "
            f"Collection: {provenance.stac_collection}. "
            f"Search parameters: bbox={bbox_str} (WGS84, [minx, miny, maxx, maxy]), "
            f"datetime={provenance.temporal_range}, "
            f"eo:cloud_cover<{provenance.cloud_cover_max}. "
            f"Selected product: {product_name}. "
            f"Source URL: {provenance.product_href}. "
            f"To reproduce: open the Zarr store at the source URL with "
            f"xarray (engine='zarr') and rerun the processing pipeline with "
            f"the parameters recorded below."
        )

    # --- Step 2: Processing ---
    ts = processing_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    op_label = op_process.biophysical_op.value.label
    sensor_name = op_process.sensor.name
    res = op_process.sensor.value.spatial_resolution_meters
    bands_str = ", ".join(op_process.sensor.value.bands)
    sensing_ts = _format_sensing_time(product_id.datatake_sensing_start_time)
    lines.append(
        f"{ts}: {op_label} computed using ESA SNAP S2ToolBox BiophysicalOp "
        f"neural network algorithm. "
        f"Source product: {product_id.full_identifier} "
        f"(sensing time: {sensing_ts}, tile: {product_id.tile_id}). "
        f"Sensor: {sensor_name}, spatial resolution: {res} m. "
        f"Input reflectance bands (bottom-of-atmosphere, BOA reflectance): "
        f"{bands_str}. "
        f"Additional NN inputs: cosine of mean view zenith angle, cosine of "
        f"sun zenith angle, cosine of relative azimuth angle (sun minus view). "
        f"Viewing angles: per-band bilinear interpolation of 23x23 geometry "
        f"grids from EOPF product conditions/geometry group, merged using "
        f"per-detector footprint masks, then averaged across all input bands. "
        f"Sun angles: bilinear interpolation of 23x23 geometry grids. "
        f"Reflectance values read from "
        f"measurements/reflectance/r{res}m group. "
        f"Two-layer neural network: tanh hidden layer (5 neurons), linear "
        f"output, with min-max normalisation / denormalisation from ESA "
        f"S2ToolBox coefficient files (processing baseline "
        f"{product_id.processing_baseline_number}). "
        f"Input validated against definition domain (stage 1: per-band "
        f"min-max; stage 2: convex hull grid). "
        f"Output validated against extreme-cases thresholds."
    )

    return "\n".join(lines)


def _build_cf_global_attrs(
    op_process: BiophysicalOpProcess,
    product_id: Sentinel2ProductIdentifier,
    crs: CRS,
    provenance: ProvenanceInfo | None,
    processing_time: datetime,
) -> dict[str, str | int | float]:
    """Build CF-1.13 compliant global attributes for the output dataset.

    Parameters
    ----------
    op_process : BiophysicalOpProcess
        Completed processing object.
    product_id : Sentinel2ProductIdentifier
        Parsed product identifier.
    crs : CRS
        Coordinate reference system of the output grid.
    provenance : ProvenanceInfo or None
        STAC search provenance.
    processing_time : datetime
        Time the output file was written.

    Returns
    -------
    dict
        Global attributes dict suitable for assignment to ``ds.attrs`` or
        inclusion in a ``rasterio`` tag dict.
    """
    op_label = op_process.biophysical_op.value.label
    op_desc = op_process.biophysical_op.value.description
    sensor_name = op_process.sensor.name
    res = op_process.sensor.value.spatial_resolution_meters
    bands_str = ", ".join(op_process.sensor.value.bands)
    product_name = Path(product_id.full_identifier).stem
    sensing_ts = _format_sensing_time(product_id.datatake_sensing_start_time)

    attrs: dict[str, str | int | float] = {
        # --- CF required / strongly recommended ---
        "Conventions": "CF-1.13",
        "title": (f"Sentinel-2 {op_label} - {product_name}"),
        "institution": "",
        "source": (
            f"ESA SNAP S2ToolBox BiophysicalOp neural network algorithm "
            f"applied to Sentinel-2 L2A ({sensor_name}) bottom-of-atmosphere "
            f"reflectances at {res} m resolution. "
            f"Input bands: {bands_str} + illumination/viewing geometry cosines."
        ),
        "history": build_history(op_process, product_id, provenance, processing_time),
        "references": (
            "Weiss, M., Baret, F. (2016). S2ToolBox Level 2 products: LAI, "
            "FAPAR, FCOVER. Algorithm Theoretical Basis Document (ATBD), "
            "ESA. https://step.esa.int/docs/extra/ATBD_S2ToolBox_V1.1.pdf"
        ),
        "comment": op_desc,
        # --- Product provenance ---
        "product_id": product_name,
        "mission_id": product_id.mission_id,
        "sensing_time": sensing_ts,
        "tile_id": product_id.tile_id,
        "processing_baseline": product_id.processing_baseline_number,
        "relative_orbit": product_id.relative_orbit_number,
        # --- Spatial metadata ---
        "geospatial_bounds_crs": f"EPSG:{crs.to_epsg()}",
        "spatial_resolution_meters": str(res),
        # --- Creation metadata ---
        "date_created": processing_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Add STAC search context when available
    if provenance is not None:
        attrs["stac_api_endpoint"] = provenance.stac_api_endpoint
        attrs["stac_collection"] = provenance.stac_collection
        attrs["stac_search_bbox"] = "[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
            *provenance.bbox
        )
        attrs["stac_search_datetime"] = provenance.temporal_range
        attrs["stac_search_cloud_cover_max"] = str(provenance.cloud_cover_max)
        attrs["source_product_url"] = provenance.product_href
        attrs["source_access_time"] = provenance.access_time.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    return attrs


def _build_variable_cf_attrs(
    var_name: str,
    op_label: str,
    op_unit: str,
) -> dict[str, object]:
    """Return CF-compliant variable-level attributes for *var_name*.

    Parameters
    ----------
    var_name : str
        Name of the output variable (e.g. ``"LAI"`` or ``"input_out_of_range"``).
    op_label : str
        Biophysical variable label (e.g. ``"LAI"``).
    op_unit : str
        Unit string from the biophysical variable definition (may be empty).

    Returns
    -------
    dict
        Attribute dict suitable for ``da.attrs``.
    """
    # CF requires "1" for dimensionless quantities, not an empty string
    units_cf = op_unit if op_unit else "1"

    _VARIABLE_ATTRS: dict[str, dict[str, object]] = {
        op_label: {
            "standard_name": "leaf_area_index" if op_label == "LAI" else op_label,
            "long_name": ("Leaf Area Index" if op_label == "LAI" else op_label),
            "units": units_cf,
            "valid_min": np.float32(0.0),
            "valid_max": np.float32(8.0),
            "grid_mapping": "crs",
        },
        "input_out_of_range": {
            "long_name": "Input spectral bands outside definition domain",
            "flag_values": np.array([0, 1], dtype=np.uint8),
            "flag_meanings": "valid out_of_range",
            "grid_mapping": "crs",
        },
        "output_set_to_min": {
            "long_name": "Output thresholded to minimum value",
            "flag_values": np.array([0, 1], dtype=np.uint8),
            "flag_meanings": "not_thresholded thresholded_to_min",
            "grid_mapping": "crs",
        },
        "output_set_to_max": {
            "long_name": "Output thresholded to maximum value",
            "flag_values": np.array([0, 1], dtype=np.uint8),
            "flag_meanings": "not_thresholded thresholded_to_max",
            "grid_mapping": "crs",
        },
        "output_too_low": {
            "long_name": "Output below acceptable minimum",
            "flag_values": np.array([0, 1], dtype=np.uint8),
            "flag_meanings": "valid below_min",
            "grid_mapping": "crs",
        },
        "output_too_high": {
            "long_name": "Output above acceptable maximum",
            "flag_values": np.array([0, 1], dtype=np.uint8),
            "flag_meanings": "valid above_max",
            "grid_mapping": "crs",
        },
    }

    return dict(_VARIABLE_ATTRS.get(var_name, {}))


# ---------------------------------------------------------------------------
# Public save functions
# ---------------------------------------------------------------------------


def save_geotiff(
    op_process: BiophysicalOpProcess,
    dt: xr.DataTree,
    output_path: str | Path,
    reflectance_group_path: str,
    provenance_info: ProvenanceInfo | None = None,
) -> Path:
    """Save the biophysical processing result as a multi-band GeoTIFF.

    Band 1 contains the primary output data (float32).  Subsequent bands
    contain the validation masks (uint8, 0/1) for any masks that are
    present in the result.

    CF-inspired metadata is embedded as GDAL string tags on the file.
    Variable-level descriptions and units are stored as individual band-level
    tags.  If *provenance_info* is supplied, the ``history`` tag records the
    full step-by-step data retrieval and processing provenance so that the
    output can be reproduced by a third party.  GeoTIFF is not a CF-compliant
    format; use ``save_zarr`` for fully CF-1.13 compliant output.

    Parameters
    ----------
    op_process : BiophysicalOpProcess
        The completed biophysical processing result.
    dt : xr.DataTree
        The source Sentinel-2 DataTree (used for spatial reference).
    output_path : str or Path
        Destination file path for the GeoTIFF.
    reflectance_group_path : str
        Path to the reflectance group in the DataTree, e.g.
        ``"measurements/reflectance/r20m"``.
    provenance_info : ProvenanceInfo or None, optional
        STAC search context for CF history metadata.  When ``None`` only the
        processing step is recorded in the ``history`` tag.

    Returns
    -------
    Path
        The path to the written GeoTIFF file.
    """
    output_path = Path(output_path)
    result = op_process.result

    if result.data is None:
        raise ValueError("Result data is None; nothing to save.")

    transform, crs, height, width, product_id = _resolve_spatial_info(
        dt, reflectance_group_path
    )

    processing_time = datetime.now(timezone.utc)

    # Collect layers: primary data + masks
    mask_layers = _collect_mask_layers(op_process)
    band_count = 1 + len(mask_layers)

    op_label = op_process.biophysical_op.value.label
    descriptions = [op_label] + [name for name, _ in mask_layers]

    # Build CF-compliant tags for the file-level metadata
    cf_tags = _build_cf_global_attrs(
        op_process, product_id, crs, provenance_info, processing_time
    )
    # Merge with legacy per-process tags for backward compatibility
    tags = {**_get_metadata_tags(op_process), **cf_tags}

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=band_count,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        # Band 1: primary data
        data = np.asarray(result.data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(height, width)
        dst.write(data, 1)
        dst.set_band_description(1, descriptions[0])

        # Per-band CF tags
        op_unit = op_process.biophysical_op.value.unit or "1"
        dst.update_tags(
            1,
            long_name="Leaf Area Index" if op_label == "LAI" else op_label,
            units=op_unit,
            valid_min="0.0",
            valid_max="8.0",
        )

        # Remaining bands: masks as float32 (0.0 / 1.0)
        for i, (name, mask_arr) in enumerate(mask_layers, start=2):
            dst.write(mask_arr.astype(np.float32), i)
            dst.set_band_description(i, descriptions[i - 1])
            var_attrs = _build_variable_cf_attrs(
                name, op_label, op_process.biophysical_op.value.unit
            )
            band_tags = {
                k: str(v)
                for k, v in var_attrs.items()
                if k not in ("flag_values", "grid_mapping", "coordinates")
            }
            dst.update_tags(i, **band_tags)

        dst.update_tags(**{k: str(v) for k, v in tags.items()})

    return output_path

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