import xarray as xr
import zarr
import json
from pathlib import Path

def create_overviews(dataset, variable_names, scales, x_dim="x", y_dim="y"):
    """
    Generate in-memory overview (downscaled) datasets for selected variables.
    Performs computation only - no I/O.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset containing the variables.
    variable_names : list[str]
        Names of the variables to downscale.
    scales : list[int]
        Scale factors relative to input (e.g. [2, 4, 8]).
    x_dim, y_dim : str
        Names of spatial dimensions (default: "x", "y").

    Returns
    -------
    dict[str, xarray.Dataset]
        Mapping of level IDs ("L1", "L2", …) to coarsened datasets (excludes L0).
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset.")

    # Check that coordinate dimensions exist
    if x_dim not in dataset.dims:
        raise ValueError(f"Dimension '{x_dim}' not found in dataset.")
    if y_dim not in dataset.dims:
        raise ValueError(f"Dimension '{y_dim}' not found in dataset.")

    # Check that all requested variables exist
    for v in variable_names:
        if v not in dataset:
            raise KeyError(f"Variable '{v}' not found in dataset.")

    overviews = {}
    for i, factor in enumerate(scales):
        level_id = f"L{i+1}"

        # Coarsen the dataset on x and y using mean aggregation
        coarsened = dataset.coarsen({x_dim: factor, y_dim: factor}, boundary="trim").mean()

        # Keep only the selected variables and coordinates at this level
        overviews[level_id] = coarsened[variable_names]

    return overviews





def create_multiscales_metadata(dataset, variable_names, scales, overview_path=".", resampling_method="average", version="1.0", x_dim="x", y_dim="y"):
    """
    Attach GeoZarr-compliant multiscales metadata to dataset.attrs["multiscales"].
    Modifies dataset in place.

    Parameters
    ----------
    dataset : xarray.Dataset
        Base (L0) dataset to which metadata will be attached.
    variable_names : list[str]
        Variables included in the overview hierarchy.
    scales : list[int]
        Scale factors for each overview level (e.g. [2, 4, 8]).
    overview_path : str
        Relative path for overviews - "." for direct children, "overviews" for subfolder (default: ".").
    resampling_method : str
        Default resampling method (default: "average").
    version : str
        Multiscales schema version (default: "1.0").
    x_dim, y_dim : str
        Names of spatial coordinate dimensions (default: "x", "y").

    Returns
    -------
    xarray.Dataset
        Input dataset with multiscales metadata attached (for chaining).
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset.")

    # Get native cell size for L0

    x_res = abs(float(dataset[x_dim].values[1] - dataset[x_dim].values[0]))
    y_res = abs(float(dataset[y_dim].values[1] - dataset[y_dim].values[0]))
    native_cell_size = [x_res, y_res]
    # Build the layout array
    # L0 uses path "." to indicate native data at current group level
    layout = [{"id": "L0", "path": ".", "cell_size": native_cell_size}]  # Base level (native data)

    for i, factor in enumerate(scales):
        level_id = f"L{i+1}"

        # Construct path based on overview_path parameter
        if overview_path == ".":
            level_path = level_id
        else:
            level_path = f"{overview_path}/{level_id}"

        # Calculate cell size for this overview level
        # Each level's cell size is the native cell size multiplied by the scale factor
        level_cell_size = [native_cell_size[0] * factor, native_cell_size[1] * factor]

        # Each entry defines one overview level and its derivation
        layout.append({
            "id": level_id,
            "path": level_path,
            "derived_from": "L0" if i == 0 else f"L{i}",
            "factors": [factor, factor],
            "resampling_method": resampling_method,
            "cell_size": level_cell_size
        })

    # Assemble the multiscales attribute following the specification
    multiscales = {
        "version": version,
        "resampling_method": resampling_method,
        "variables": variable_names,
        "layout": layout
    }

    # Attach metadata to dataset attributes
    dataset.attrs["multiscales"] = multiscales

    return dataset


def write_overviews(zarr_group_path, overviews, overview_path=".", mode="a", zarr_version=None):
    """
    Persist computed overview datasets to Zarr storage.
    Writes L1, L2, ... as subgroups (L0 should be written separately).
    Creates all necessary intermediate groups automatically.

    Parameters
    ----------
    zarr_group_path : str or Path
        Path to parent Zarr group where overviews will be written.
    overviews : dict[str, xarray.Dataset]
        Dictionary mapping level IDs to datasets (output of create_overviews()).
    overview_path : str
        Relative path for overviews - "." for direct children, "overviews" for subfolder (default: ".").
    mode : str
        Write mode - "w" (overwrite), "a" (append), "w-" (write if new) (default: "a").
    zarr_version : int, optional
        Zarr format version (2 or 3). If None, uses xarray default.
    """
    zarr_group_path = Path(zarr_group_path)

    # Determine the base path for overviews
    if overview_path == ".":
        base_path = zarr_group_path
    else:
        base_path = zarr_group_path / overview_path
        # Create intermediate groups if they don't exist
        base_path.mkdir(parents=True, exist_ok=True)

    # Write each overview level to its subgroup
    for level_id, level_dataset in overviews.items():
        level_path = base_path / level_id

        # Convert to absolute path string for xarray
        level_path_str = str(level_path.absolute())

        # Write the dataset to the Zarr subgroup
        write_kwargs = {"mode": mode}
        if zarr_version is not None:
            write_kwargs["zarr_version"] = zarr_version

        level_dataset.to_zarr(level_path_str, **write_kwargs)


def write_metadata(zarr_group_path, dataset_with_attrs, zarr_version=None):
    """
    Update metadata (.zattrs) of an existing Zarr group without rewriting data arrays.
    Used to add multiscales metadata after overview levels are written.

    Parameters
    ----------
    zarr_group_path : str or Path
        Path to Zarr group where metadata will be updated.

    dataset_with_attrs : xarray.Dataset
        Dataset with multiscales metadata in attrs (from create_multiscales_metadata()).
    zarr_version : int, optional
        Zarr format version (2 or 3). If None, uses xarray default (2).

    Notes
    -----
    This function uses xarray.to_zarr() in append mode to update only attributes.
    It writes coordinate variables (which are small) but skips data variables,
    effectively updating only the group's .zattrs file.
    """
    if "multiscales" not in dataset_with_attrs.attrs:
        raise KeyError("Dataset must have 'multiscales' in attrs. Use create_multiscales_metadata() first.")

    zarr_group_path = Path(zarr_group_path)
    zarr_group_path_str = str(zarr_group_path.absolute())

    # Create a minimal dataset with only coordinates and attrs (no data variables)
    # This ensures we only update metadata without rewriting large data arrays
    coords_only = xr.Dataset(
        coords=dataset_with_attrs.coords,
        attrs=dataset_with_attrs.attrs
    )

    # Write in append mode to update attributes
    write_kwargs = {"mode": "a"}
    if zarr_version is not None:
        write_kwargs["zarr_version"] = zarr_version

    coords_only.to_zarr(zarr_group_path_str, **write_kwargs)


