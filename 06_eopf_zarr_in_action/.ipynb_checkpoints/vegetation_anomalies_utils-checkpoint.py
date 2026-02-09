import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps

import pystac_client
import dask
import os

from pyproj import Transformer

from zarr_wf_utils import validate_scl


def get_items(lat, lon, start_date="2017-01-01", end_date="2024-12-31", return_as_dicts=False):
    """
    Query the EOPF STAC API for Sentinel-2 L2A items intersecting a given latitude/longitude point
    within a specified date range.

    Parameters
    ----------
    lat : float
        Latitude of the point of interest (WGS84).
    lon : float
        Longitude of the point of interest (WGS84).
    start_date : str, default="2017-01-01"
        Start of the date range.
    end_date : str, default="2024-12-31"
        End of the date range.
    return_as_dicts : bool, default=False
        If True, return items as dictionaries; otherwise return STAC Item objects.

    Returns
    -------
    list
        List of STAC Items (or dictionaries) matching the query.
    """

    # Connect to the STAC API
    client = pystac_client.Client.open("https://stac.core.eopf.eodc.eu/")
    
    # Define a GeoJSON Point for spatial filtering
    point = {"type": "Point", "coordinates": [lon, lat]}
    
    # Search for Sentinel-2 L2A items intersecting the point and within the date range
    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=point,
        datetime=f"{start_date}/{end_date}"
    )

    # Return items either as STAC objects or plain dictionaries
    if return_as_dicts:
        items = list(search.items_as_dicts())
    else:
        items = list(search.items())

    return items


def latlon_to_buffer_bbox(lat, lon, epsg, buffer=500):
    """
    Convert a latitude/longitude point to a projected coordinate system (EPSG),
    then generate a square bounding box centered on that point.

    Parameters
    ----------
    lat : float
        Latitude of the point of interest (WGS84).
    lon : float
        Longitude of the point of interest (WGS84).
    epsg : str or int
        Target projected coordinate system (e.g., EPSG:32632). Used for distance-based buffering.
    buffer : float, default=500
        Half-size of the square buffer (in meters). The output box will span
        2*buffer in width and height.

    Returns
    -------
    tuple
        (minx, miny, maxx, maxy) bounding box coordinates in the target EPSG.
    """

    # Transformer: convert geographic coordinates (lon, lat) to projected (x, y)
    transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)

    # Perform the coordinate transformation
    x, y = transformer.transform(lon, lat)

    # Construct the bounding box by applying the buffer in projected units (meters)
    minx = x - buffer
    maxx = x + buffer
    miny = y - buffer
    maxy = y + buffer

    return (minx, miny, maxx, maxy)


@dask.delayed
def open_and_curate_data(item, lat, lon, bands=["b04", "b8a"], resolution=20, buffer=500, items_as_dicts=False):
    """
    Open a Sentinel-2 STAC item (Zarr), spatially subset it around a point,
    apply SCL-based masking, and return an xarray Dataset containing only
    selected reflectance bands with a time dimension.

    Parameters
    ----------
    item : pystac.Item or dict
        STAC item describing the Sentinel-2 observation. Can be provided
        either as a dict (e.g., after JSON serialization) or as a pystac.Item.
    lat : float
        Latitude of the point of interest (WGS84).
    lon : float
        Longitude of the point of interest (WGS84).
    bands : list
        Bands to retrieve.
    resolution : int, optional
        Spatial resolution to load for reflectance bands and SCL. Default is 20 m.
    buffer : int, optional
        Half-size of the square buffer (in meters) around the point of interest.
        The function extracts a bounding box of size (2*buffer) centered on (lat, lon).
    items_as_dicts : bool, optional
        If True, treat `item` as a Python dictionary with STAC-like structure.
        If False, treat it as a pystac.Item object.

    Returns
    -------
    ds : xr.Dataset (wrapped in dask.delayed)
        Dataset with dimensions (time, y, x) containing:
        - reflectances
        - only valid pixels according to SCL filtering
        - only pixels inside the buffered bounding box
        A time dimension is added so datasets can be concatenated later.
    """

    # --------------------------------------------------------------
    # 1. Extract STAC asset HREF and timestamp
    # --------------------------------------------------------------

    # Standard STAC read depending on input format
    if items_as_dicts:
        href = item["assets"]["product"]["href"]
        datetime_value = item["properties"]["datetime"]
    else:
        href = item.assets["product"].href
        datetime_value = item.properties["datetime"]

    # Convert the STAC datetime to daily precision numpy datetime
    # (removing the trailing "Z" timezone indicator)
    time = np.datetime64(datetime_value.replace("Z", "")).astype("datetime64[D]")

    # Resolution code used by S2 Zarr hierarchy (e.g., "r20m")
    resolution = f"r{resolution}m"

    # --------------------------------------------------------------
    # 2. Open Zarr datatree
    # --------------------------------------------------------------
    ds = xr.open_datatree(
        href,
        engine="zarr",
        consolidated=True,
        chunks="auto"
    )

    # --------------------------------------------------------------
    # 3. Determine projection and build projected bounding box
    # --------------------------------------------------------------

    # EPSG code for the scene (e.g., 32632 for Sentinel-2 tile)
    epsg = ds.attrs["other_metadata"]["horizontal_CRS_code"]

    # Convert (lat, lon) to a projected bounding box centered on the point
    # Buffer is in meters, so this is done in projected CRS
    minx, miny, maxx, maxy = latlon_to_buffer_bbox(lat, lon, epsg, buffer)

    # --------------------------------------------------------------
    # 4. Extract SCL (Scene Classification Layer) and build valid mask
    # --------------------------------------------------------------

    # Access the classification layer at the correct resolution
    scl = ds.conditions.mask.l2a_classification[resolution].scl

    # Convert SCL to a boolean mask indicating valid surface reflectance pixels
    valid_mask = validate_scl(scl)

    # --------------------------------------------------------------
    # 5. Extract reflectance bands and apply mask
    # --------------------------------------------------------------

    # reflectance[...] is a datatree, convert to dataset then select bands
    ds = (
        ds.measurements.reflectance[resolution]
        .to_dataset()[bands]
        .where(valid_mask)          # Apply SCL mask (invalid → NaN)
    )

    # --------------------------------------------------------------
    # 6. Spatial subsetting to bounding box
    # --------------------------------------------------------------

    ds = ds.where(
        (ds.x >= minx) & (ds.x <= maxx) &
        (ds.y >= miny) & (ds.y <= maxy),
        drop=True
    )

    # --------------------------------------------------------------
    # 7. Add time dimension for temporal stacking
    # --------------------------------------------------------------

    # Enforce the shape (time, y, x) so multiple calls can concatenate cleanly
    ds = ds.expand_dims(time=[time])

    return ds


def curate_gpp(dataset="DE-Hai", percentile=0.1, consecutive_weeks=2):
    """
    Load a GPP time series, compute weekly anomalies, identify extreme low-GPP
    events, and filter extremes to retain only events with at least a specified
    number of consecutive weeks.

    Parameters
    ----------
    dataset : str
        Name of the CSV file (without extension) found in ./data/.
    percentile : float
        Lower-tail percentile used to define extreme negative anomalies
        (e.g., 0.1 => 10th percentile of the anomaly distribution).
    consecutive_weeks : int
        Minimum run length of consecutive extreme weeks required for an
        extreme event to be retained.

    Returns
    -------
    df : pandas.DataFrame
        A dataframe indexed by weekly timestamps, containing GPP, anomalies,
        week-of-year, and a final binary 'extreme' flag.
    """

    # --------------------------------------------------------------
    # 1. Load & preprocess time series
    # --------------------------------------------------------------
    df = pd.read_csv(os.path.join("data", f"{dataset}.csv"))

    # Ensure 'time' is parsed as a datetime
    df["time"] = pd.to_datetime(df["time"])

    # Optionally restrict to a start date for consistency
    df = df[df.time >= "2017-01-01"]

    # Set time as index for easier resampling and time-based operations
    df = df.set_index("time")

    # --------------------------------------------------------------
    # 2. Temporal aggregation: convert to weekly median GPP
    # --------------------------------------------------------------
    df = df.resample("1W").median()

    # Extract week-of-year for building a weekly climatology
    df["weekofyear"] = df.index.isocalendar().week

    # --------------------------------------------------------------
    # 3. Compute weekly climatology (multi-year mean per week)
    # --------------------------------------------------------------
    df_msc = df.groupby("weekofyear")["GPP_NT_VUT_REF"].mean()

    # Weekly anomaly = observed GPP - weekly climatological mean
    df["anomaly"] = df["GPP_NT_VUT_REF"] - df["weekofyear"].map(df_msc)

    # --------------------------------------------------------------
    # 4. Identify extreme anomalies using a Gaussian percentile threshold
    # --------------------------------------------------------------
    # Fit a normal distribution to the anomaly series
    dist = sps.norm(
        loc=df["anomaly"].mean(),
        scale=df["anomaly"].std()
    )

    # Lower-tail anomaly threshold corresponding to the chosen percentile
    q = np.abs(dist.ppf(percentile))

    # Initial binary extreme flag: 1 = extreme low anomaly
    df["extreme"] = 0
    df.loc[df["anomaly"] <= -q, "extreme"] = 1

    # --------------------------------------------------------------
    # 5. Filter extremes: keep only runs with >= consecutive_weeks
    # --------------------------------------------------------------
    s = df["extreme"]

    # Identify contiguous groups of identical values (0-runs and 1-runs)
    groups = (s != s.shift()).cumsum()

    # Compute the run length for each time step
    run_lengths = s.groupby(groups).transform("size")

    # Final extreme flag: extreme only if in a run of sufficient length
    df["extreme"] = ((s == 1) & (run_lengths >= consecutive_weeks)).astype(int)

    return df