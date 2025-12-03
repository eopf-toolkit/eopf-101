"""
ipyleaflet_multiscales.py - Utilities for visualizing multiscale GeoZarr datasets with ipyleaflet

This module provides helper functions for creating interactive Leaflet maps with
automatic overview level selection based on multiscales metadata and cell_size.

Functions
---------
create_rgb_image_base64(overview_datasets, level_idx, band_names=None, max_size=512)
    Create RGB composite as base64-encoded PNG for ImageOverlay

transform_bbox_to_latlon(proj_epsg, proj_bbox)
    Transform bounding box from projected CRS to EPSG:4326 (lat/lon)

select_level_for_zoom(multiscales, zoom, lat)
    Smart selection of overview level based on Leaflet zoom and cell_size metadata

create_interactive_map(overview_datasets, multiscales, metadata, initial_level=4, initial_zoom=10)
    Create complete interactive Leaflet map with auto-switching overview levels

Example Usage
-------------
>>> import xarray as xr
>>> from ipyleaflet_multiscales import create_interactive_map
>>>
>>> # Load dataset with multiscales
>>> dataset = xr.open_dataset("data.zarr", engine="zarr")
>>> multiscales = dataset.attrs["multiscales"]
>>>
>>> # Load all overview levels
>>> overview_datasets = load_overview_datasets(dataset_path, multiscales)
>>>
>>> # Create interactive map
>>> map_widget = create_interactive_map(
>>>     overview_datasets,
>>>     multiscales,
>>>     metadata=dataset.b02.attrs
>>> )
>>> display(map_widget)
"""

import math
import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, Optional
import xarray as xr
from io import BytesIO
import base64
from PIL import Image
from pyproj import Transformer
from ipyleaflet import Map, ImageOverlay, LayersControl, basemaps
from ipywidgets import IntSlider, VBox, Label
from IPython.display import display, HTML


def create_rgb_image_base64(overview_datasets: Dict[str, xr.Dataset], level_idx: int, band_names: Optional[Dict[str, str]] = None, max_size: int = 512) -> Tuple[str, Tuple[int, int, int]]:
    """
    Create RGB composite as base64-encoded PNG for ImageOverlay.

    Parameters
    ----------
    overview_datasets : dict
        Dictionary mapping level IDs (e.g., "L0", "L1") to xarray.Dataset
    level_idx : int
        Overview level index (0=L0, 1=L1, etc.)
    band_names : dict, optional
        Band names for RGB channels: {"r": "b04", "g": "b03", "b": "b02"}
        Default uses Sentinel-2 band names
    max_size : int
        Maximum dimension size for display (default: 512px)

    Returns
    -------
    tuple
        (base64_encoded_png_string, image_shape)
    """
    # Default to Sentinel-2 band names
    if band_names is None:
        band_names = {"r": "b04", "g": "b03", "b": "b02"}

    level_id = f"L{level_idx}"
    ds = overview_datasets[level_id]

    # Extract RGB bands
    r = ds[band_names["r"]].values
    g = ds[band_names["g"]].values
    b = ds[band_names["b"]].values
    h, w = r.shape

    # Downsample if too large (for faster rendering)
    if max(h, w) > max_size:
        step = max(h, w) // max_size
        r, g, b = r[::step, ::step], g[::step, ::step], b[::step, ::step]

    # Normalize each band using percentile stretching (2%-98%)
    def normalize(band: npt.NDArray) -> npt.NDArray:
        valid_pixels = band[~np.isnan(band)]  # ← ADD THIS LINE
        if len(valid_pixels) == 0:
            return np.zeros_like(band)
        p2, p98 = np.percentile(valid_pixels, [2, 98])  # ← USE valid_pixels
        if p98 == p2:
            return np.zeros_like(band)
        normalized = (band - p2) / (p98 - p2)
        return np.clip(normalized, 0, 1)

    # Stack into RGB array and convert to uint8
    rgb = np.dstack([normalize(r), normalize(g), normalize(b)])
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # Convert to PNG base64
    img = Image.fromarray(rgb_uint8, mode="RGB")
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_base64}", rgb_uint8.shape


def transform_bbox_to_latlon(proj_epsg: int, proj_bbox: Tuple[float, float, float, float]) -> Dict[str, any]:
    """
    Transform bounding box from projected CRS to EPSG:4326 (lat/lon).

    Parameters
    ----------
    proj_epsg : int
        Source EPSG code (e.g., 32632 for UTM Zone 32N)
    proj_bbox : list
        Bounding box in source CRS: [x_min, y_min, x_max, y_max]

    Returns
    -------
    dict
        Dictionary with keys: "center" [lat, lon], "bounds" [[lat_min, lon_min], [lat_max, lon_max]]
    """
    # Transform bbox from source EPSG to WGS84 lat/lon (EPSG:4326)
    transformer = Transformer.from_crs(f"EPSG:{proj_epsg}", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(proj_bbox[0], proj_bbox[1])
    lon_max, lat_max = transformer.transform(proj_bbox[2], proj_bbox[3])

    # Calculate center in EPSG:4326 (lat/lon)
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    return {
        "center": [center_lat, center_lon],
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "center_lat": center_lat,
        "center_lon": center_lon
    }


def select_level_for_zoom(multiscales: Dict[str, any], zoom: int, lat: float) -> int:
    """
    Select best overview level based on Leaflet zoom and cell_size metadata.

    Uses Web Mercator tile pyramid formula to calculate ground resolution,
    then finds the overview level whose cell_size is closest to that resolution.

    Parameters
    ----------
    multiscales : dict
        Multiscales metadata dictionary from dataset.attrs["multiscales"]
    zoom : int
        Current Leaflet zoom level (typically 6-18)
    lat : float
        Latitude for Mercator projection correction

    Returns
    -------
    int
        Best overview level index (0=L0, 1=L1, etc.)

    Notes
    -----
    Web Mercator ground resolution formula:
        ground_resolution (m/px) = (156543.03 * cos(lat)) / (2^zoom)

    Examples:
        At zoom 10 and lat 52°: ~84 m/px → selects L3 (80m cell_size)
        At zoom 13 and lat 52°: ~10 m/px → selects L0 (10m cell_size)
    """
    # Calculate ground resolution at this zoom level (meters/pixel)
    lat_radians = math.radians(lat)
    ground_res = (156543.03 * math.cos(lat_radians)) / (2 ** zoom)

    # Find the level with cell_size closest to ground_res
    best_level = 0
    min_diff = float('inf')

    for entry in multiscales["layout"]:
        level_id = entry["id"]
        level_idx = 0 if level_id == "L0" else int(level_id[1:])
        cell_size = entry["cell_size"][0]  # Use x resolution (assumes square pixels)

        # Find closest match by minimizing difference
        diff = abs(cell_size - (ground_res/2))
        if diff < min_diff:
            min_diff = diff
            best_level = level_idx

    return best_level


def enable_crisp_rendering() -> None:
    """
    Enable crisp (pixelated) rendering for Leaflet image layers.

    Disables image smoothing to show pixels clearly instead of blurred.
    Call this function before displaying the map.
    """
    display(HTML("""
    <style>
    .leaflet-image-layer {
        image-rendering: pixelated;
        image-rendering: -moz-crisp-edges;
        image-rendering: crisp-edges;
    }
    </style>
    """))


def create_interactive_map(overview_datasets: Dict[str, xr.Dataset], multiscales: Dict[str, any], metadata: Dict[str, any],
                          initial_level: int = 5, initial_zoom: int = 7, band_names: Optional[Dict[str, str]] = None, max_size: int = 512) -> VBox:
    """
    Create complete interactive Leaflet map with auto-switching overview levels.

    The map automatically selects the optimal overview level based on zoom and
    cell_size metadata. Users can also manually switch levels using a slider.

    Parameters
    ----------
    overview_datasets : dict
        Dictionary mapping level IDs (e.g., "L0", "L1") to xarray.Dataset
    multiscales : dict
        Multiscales metadata from dataset.attrs["multiscales"]
    metadata : dict
        Variable metadata containing "proj:epsg" and "proj:bbox"
    initial_level : int
        Initial overview level to display (default: 4)
    initial_zoom : int
        Initial map zoom level (default: 10)
    band_names : dict, optional
        Band names for RGB: {"r": "b04", "g": "b03", "b": "b02"}
    max_size : int
        Maximum image dimension for rendering (default: 512px)

    Returns
    -------
    ipywidgets.VBox
        Widget containing label, slider, and map

    Example
    -------
    >>> map_widget = create_interactive_map(
    ...     overview_datasets,
    ...     multiscales,
    ...     metadata=dataset.b02.attrs
    ... )
    >>> display(map_widget)
    """
    # Extract geospatial metadata
    proj_epsg = metadata["proj:epsg"]
    proj_bbox = metadata["proj:bbox"]

    # Transform to lat/lon
    geo_info = transform_bbox_to_latlon(proj_epsg, proj_bbox)
    center = geo_info["center"]
    bounds = geo_info["bounds"]
    center_lat = geo_info["center_lat"]
    center_lon = geo_info["center_lon"]

    print(f"🗺️  Creating interactive map with overview levels...")
    print(f"   Center: [{center_lat:.4f}, {center_lon:.4f}] (lat/lon)")
    print(f"   Bounds: [[{bounds[0][0]:.4f}, {bounds[0][1]:.4f}], [{bounds[1][0]:.4f}, {bounds[1][1]:.4f}]]")
    print()

    # Enable crisp pixel rendering
    enable_crisp_rendering()

    # Create map
    m = Map(center=center, zoom=initial_zoom, basemap=basemaps.OpenStreetMap.Mapnik)

    # Create initial image overlay
    print(f"Creating initial overlay for L{initial_level}...")
    img_base64, shape = create_rgb_image_base64(overview_datasets, initial_level, band_names, max_size)
    print(f"   Image shape: {shape}")
    image_overlay = ImageOverlay(url=img_base64, bounds=bounds, name=f"Overview L{initial_level}", opacity=0.8)

    # Add overlay and layer control
    m.add(image_overlay)
    m.add(LayersControl())

    # Determine max level from overview_datasets
    max_level = len(overview_datasets) - 1

    # Create slider and label with initial info
    level_slider = IntSlider(min=0, max=max_level, value=initial_level,
                            description='Level:', continuous_update=False)

    # Calculate initial ground resolution
    lat_radians = math.radians(center_lat)
    initial_ground_res = (156543.03 * math.cos(lat_radians)) / (2 ** initial_zoom)

    # Get initial cell_size
    initial_cell_size = None
    for entry in multiscales["layout"]:
        if entry["id"] == f"L{initial_level}" and "cell_size" in entry:
            initial_cell_size = entry["cell_size"][0]
            break

    # Build initial label
    initial_shape = overview_datasets[f"L{initial_level}"][list(overview_datasets[f"L{initial_level}"].data_vars)[0]].shape
    initial_factor = 2 ** initial_level if initial_level > 0 else 1
    initial_label = f'Displaying L{initial_level} ({initial_shape[0]}×{initial_shape[1]} px) - {initial_factor}× downsampled'
    if initial_cell_size:
        initial_label += f' | Cell size: {initial_cell_size:.1f}m'
    initial_label += f' | Zoom: {initial_zoom} | Ground res: {initial_ground_res:.1f}m/px'

    level_label = Label(value=initial_label)

    # Event handlers
    def update_overlay_to_level(level_idx: int, zoom: Optional[int] = None) -> None:
        """Update image overlay to a specific level."""
        level_id = f"L{level_idx}"
        ds_level = overview_datasets[level_id]
        shape_full = ds_level[list(ds_level.data_vars)[0]].shape
        factor = 2 ** level_idx if level_idx > 0 else 1

        # Get cell_size for this level
        cell_size = None
        for entry in multiscales["layout"]:
            if entry["id"] == level_id and "cell_size" in entry:
                cell_size = entry["cell_size"][0]
                break

        # Update overlay with new image
        img_base64, shape_display = create_rgb_image_base64(overview_datasets, level_idx, band_names, max_size)
        image_overlay.url = img_base64
        image_overlay.name = f"Overview {level_id}"

        # Build label with zoom and ground resolution info
        label_text = f'Displaying {level_id} ({shape_full[0]}×{shape_full[1]} px) - {factor}× downsampled'
        if cell_size:
            label_text += f' | Cell size: {cell_size:.1f}m'
        if zoom is not None:
            lat_radians = math.radians(center_lat)
            ground_res = (156543.03 * math.cos(lat_radians)) / (2 ** zoom)
            label_text += f' | Zoom: {zoom} | Ground res: {ground_res:.1f}m/px'

        level_label.value = label_text

    def on_slider_change(change: Dict[str, any]) -> None:
        """Handle manual slider changes."""
        # Get current zoom from map
        current_zoom = m.zoom
        update_overlay_to_level(change['new'], zoom=current_zoom)

    def on_zoom_change(change: Dict[str, any]) -> None:
        """Automatically select appropriate level based on zoom and cell_size."""
        zoom = change['new']
        suggested_level = select_level_for_zoom(multiscales, zoom, center_lat)
        update_overlay_to_level(suggested_level, zoom=zoom)
        level_slider.value = suggested_level

    # Attach event handlers
    level_slider.observe(on_slider_change, names='value')
    m.observe(on_zoom_change, names='zoom')

    # Display success message
    print("✓ Map created successfully!")
    print("   - Zoom in/out to automatically switch overview levels (smart selection based on cell_size)")
    print("   - Or use the slider to manually select a level\n")

    # Return widget
    return VBox([level_label, level_slider, m])
