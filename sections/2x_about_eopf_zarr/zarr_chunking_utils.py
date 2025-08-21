"""
Utility functions for EOPF Zarr chunking demonstrations.

This module provides helper functions for creating, analyzing, and comparing
different Zarr chunking strategies for Sentinel-2 data.
"""

import numpy as np
import xarray as xr
import zarr
import pystac_client
from pystac_client import Client as STACClient
from pystac import MediaType
import dask.array as da
from dask.distributed import Client
import pandas as pd
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def create_dask_client(n_workers: int = 4, 
                       threads_per_worker: int = 2, 
                       memory_limit: str = '4GB') -> Client:
    """
    Create a Dask client optimized for Zarr operations.
    
    Parameters:
    -----------
    n_workers : int
        Number of worker processes
    threads_per_worker : int
        Number of threads per worker
    memory_limit : str
        Memory limit per worker
    
    Returns:
    --------
    Client : Dask client instance
    """
    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        silence_logs=40
    )
    print(f"Dask dashboard available at: {client.dashboard_link}")
    return client


def get_sentinel2_data(bbox: List[float], 
                       start_date: str, 
                       end_date: str,
                       max_items: int = 5,
                       cloud_cover: int = 20) -> List:
    """
    Retrieve Sentinel-2 L2A data from EOPF STAC catalog.
    
    Parameters:
    -----------
    bbox : List[float]
        Bounding box [min_lon, min_lat, max_lon, max_lat]
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    max_items : int
        Maximum number of items to retrieve
    cloud_cover : int
        Maximum cloud cover percentage
    
    Returns:
    --------
    List : List of STAC items with cloud storage URLs
    """
    print(f"Searching for Sentinel-2 data from EOPF STAC Catalog...")
    print(f"  Area: {bbox}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Max cloud cover: {cloud_cover}%")
    
    try:
        # Connect to EOPF STAC API
        eopf_stac_api_root_endpoint = "https://stac.core.eopf.eodc.eu/"
        catalog = STACClient.open(url=eopf_stac_api_root_endpoint)
        
        # Search for Sentinel-2 L2A data
        search = catalog.search(
            collections='sentinel-2-l1c',
            bbox=bbox,
            datetime=f'{start_date}T00:00:00Z/{end_date}T23:59:59Z',
            max_items=max_items
        )
        
        # Get collection for item access
        c_sentinel2 = catalog.get_collection('sentinel-2-l1c')
        
        items_with_urls = []
        for item in search.items():
            # Get the cloud storage URL
            stac_item = c_sentinel2.get_item(id=item.id)
            item_assets = stac_item.get_assets(media_type=MediaType.ZARR)
            
            if 'product' in item_assets:
                cloud_storage_url = item_assets['product'].href
                items_with_urls.append({
                    'id': item.id,
                    'datetime': item.datetime,
                    'cloud_storage_url': cloud_storage_url,
                    'properties': item.properties
                })
        
        print(f"Found {len(items_with_urls)} Sentinel-2 acquisitions with cloud storage URLs")
        return items_with_urls
        
    except Exception as e:
        print(f"Warning: Could not connect to EOPF STAC API: {e}")
        print("Falling back to simulated data for demonstration...")
        return create_simulated_items(start_date, end_date, max_items)


def create_simulated_items(start_date: str, end_date: str, max_items: int) -> List[Dict]:
    """Create simulated items when STAC API is not available."""
    dates = pd.date_range(start_date, end_date, periods=max_items)
    items = []
    
    for date in dates:
        items.append({
            'id': f'S2_L2A_{date.strftime("%Y%m%d")}',
            'datetime': date.isoformat(),
            'cloud_storage_url': None,  # Will trigger sample data creation
            'properties': {
                'eo:cloud_cover': np.random.randint(0, 20)
            }
        })
    
    return items


def create_sample_data(shape: Tuple[int, int, int], 
                      bands: List[str]) -> xr.Dataset:
    """
    Create sample Sentinel-2 like data for demonstration.
    
    Parameters:
    -----------
    shape : Tuple[int, int, int]
        Shape of the data (time, y, x)
    bands : List[str]
        List of band names
    
    Returns:
    --------
    xr.Dataset : Sample dataset
    """
    data_vars = {}
    
    for band in bands:
        # Create realistic-looking data with spatial patterns
        data = np.random.randn(*shape) * 1000 + 5000
        
        # Add some spatial correlation
        from scipy.ndimage import gaussian_filter
        for t in range(shape[0]):
            data[t] = gaussian_filter(data[t], sigma=5)
        
        data_vars[band] = (['time', 'y', 'x'], data.astype(np.uint16))
    
    # Create coordinates
    times = pd.date_range('2024-06-01', periods=shape[0], freq='5D')
    
    ds = xr.Dataset(
        data_vars,
        coords={
            'time': times,
            'y': np.arange(shape[1]),
            'x': np.arange(shape[2])
        }
    )
    
    # Add attributes
    ds.attrs['description'] = 'Sample Sentinel-2 L2A data'
    ds.attrs['platform'] = 'Sentinel-2'
    
    for band in bands:
        ds[band].attrs['units'] = 'reflectance'
        ds[band].attrs['scale_factor'] = 0.0001
    
    return ds


def create_multitemporal_zarr_from_eopf(items: List[Dict],
                                        output_path: str,
                                        chunk_strategy: Dict,
                                        bands_10m: List[str] = ['b02', 'b03', 'b04', 'b08'],
                                        compression: str = 'zstd',
                                        compression_level: int = 3,
                                        use_sample_data: bool = False) -> xr.Dataset:
    """
    Create a multi-temporal Zarr dataset from EOPF STAC items.
    
    Parameters:
    -----------
    items : List[Dict]
        List of STAC items with cloud storage URLs
    output_path : str
        Path to save the Zarr dataset
    chunk_strategy : Dict
        Chunking configuration {'time': int, 'y': int, 'x': int}
    bands_10m : List[str]
        List of 10m bands to include
    compression : str
        Compression algorithm
    compression_level : int
        Compression level
    use_sample_data : bool
        If True, use sample data instead of real EOPF data
    
    Returns:
    --------
    xr.Dataset : Created dataset
    """
    datasets = []
    
    print(f"Processing {len(items)} acquisitions...")
    
    # If no real data available or forced to use sample data
    if not items or items[0]['cloud_storage_url'] is None or use_sample_data:
        print("Using sample data for demonstration...")
        # Create sample data
        sample_dataset = create_sample_data(
            shape=(len(items), 2048, 2048), 
            bands=['B02', 'B03', 'B04', 'B08']
        )
        return create_multitemporal_zarr(
            sample_dataset, output_path, chunk_strategy, 
            compression, compression_level
        )
    
    # Process real EOPF data
    for item in items:
        try:
            # Open the EOPF Zarr dataset
            dt = xr.open_datatree(
                item['cloud_storage_url'],
                engine="zarr"
            )
            
            # Extract 10m bands
            band_data = {}
            for band in bands_10m:
                # EOPF structure: measurements/reflectance/r10m/{band}
                band_path = f'/measurements/reflectance/r10m'
                if band_path in dt.groups:
                    ds_10m = dt[band_path].to_dataset()
                    if band in ds_10m:
                        band_data[band.upper()] = ds_10m[band]
            
            if band_data:
                # Create dataset for this acquisition
                ds = xr.Dataset(band_data)
                
                # Add time dimension
                acquisition_time = pd.to_datetime(item['datetime'])
                ds = ds.expand_dims(time=[acquisition_time])
                
                datasets.append(ds)
            
        except Exception as e:
            print(f"Warning: Could not process item {item['id']}: {e}")
            continue
    
    if not datasets:
        print("No valid datasets found, using sample data...")
        sample_dataset = create_sample_data(
            shape=(len(items), 2048, 2048), 
            bands=['B02', 'B03', 'B04', 'B08']
        )
        return create_multitemporal_zarr(
            sample_dataset, output_path, chunk_strategy, 
            compression, compression_level
        )
    
    # Concatenate all acquisitions along time dimension
    combined_ds = xr.concat(datasets, dim='time')
    
    # Save with chunking strategy
    return create_multitemporal_zarr(
        combined_ds, output_path, chunk_strategy, 
        compression, compression_level
    )


def create_multitemporal_zarr(dataset: xr.Dataset,
                              output_path: str,
                              chunk_strategy: Dict,
                              compression: str = 'zstd',
                              compression_level: int = 3) -> xr.Dataset:
    """
    Save a dataset to Zarr with specified chunking strategy.
    
    Parameters:
    -----------
    dataset : xr.Dataset
        Input dataset
    output_path : str
        Path to save the Zarr dataset
    chunk_strategy : Dict
        Chunking configuration {'time': int, 'y': int, 'x': int}
    compression : str
        Compression algorithm
    compression_level : int
        Compression level
    
    Returns:
    --------
    xr.Dataset : Saved dataset
    """
    # Rechunk according to strategy
    dataset = dataset.chunk(chunk_strategy)
    
    # Set up encoding with compression
    encoding = {}
    compressor = zarr.Blosc(cname=compression, clevel=compression_level)
    
    for var in dataset.data_vars:
        encoding[var] = {
            'compressor': compressor,
            'chunks': tuple(chunk_strategy.values())
        }
    
    # Save to Zarr
    print(f"Saving to {output_path}")
    print(f"  Chunking: {chunk_strategy}")
    print(f"  Compression: {compression} (level {compression_level})")
    
    dataset.to_zarr(output_path, mode='w', encoding=encoding, consolidated=True)
    
    return dataset


def analyze_zarr_performance(zarr_path: str, 
                            test_reads: bool = True) -> Dict:
    """
    Analyze performance characteristics of a Zarr dataset.
    
    Parameters:
    -----------
    zarr_path : str
        Path to the Zarr dataset
    test_reads : bool
        Whether to test read performance
    
    Returns:
    --------
    Dict : Performance metrics
    """
    # Open the Zarr store
    store = zarr.open(zarr_path, mode='r')
    
    # Get dataset info
    ds = xr.open_zarr(zarr_path, consolidated=True)
    
    metrics = {}
    
    # Storage metrics
    total_size = 0
    chunk_sizes = []
    compression_ratios = []
    
    for array_name in store.array_keys():
        array = store[array_name]
        
        # Calculate storage size
        nbytes_stored = array.nbytes_stored
        nbytes_uncompressed = array.nbytes
        total_size += nbytes_stored
        
        # Get chunk information
        chunks = array.chunks
        chunk_size = np.prod(chunks) * array.dtype.itemsize
        chunk_sizes.append(chunk_size)
        
        # Calculate compression ratio
        if nbytes_stored > 0:
            compression_ratio = nbytes_uncompressed / nbytes_stored
            compression_ratios.append(compression_ratio)
        
        metrics[f'{array_name}_chunks'] = chunks
        metrics[f'{array_name}_chunk_size_mb'] = chunk_size / (1024 * 1024)
        metrics[f'{array_name}_compression_ratio'] = compression_ratio if nbytes_stored > 0 else 0
    
    metrics['total_size_mb'] = total_size / (1024 * 1024)
    metrics['avg_chunk_size_mb'] = np.mean(chunk_sizes) / (1024 * 1024) if chunk_sizes else 0
    metrics['avg_compression_ratio'] = np.mean(compression_ratios) if compression_ratios else 0
    metrics['num_chunks'] = sum([array.nchunks for array in store.arrays()])
    
    # Test read performance if requested
    if test_reads and ds.dims['time'] > 0:
        print("  Testing read performance...")
        
        # Get dimensions
        t_size = min(ds.dims['time'], 1)
        y_size = min(ds.dims['y'], 1000)
        x_size = min(ds.dims['x'], 1000)
        
        # Spatial slice (single time step, spatial subset)
        start_time = time.time()
        spatial_data = ds.isel(
            time=0, 
            y=slice(0, y_size), 
            x=slice(0, x_size)
        ).compute()
        metrics['spatial_read_time_s'] = time.time() - start_time
        
        # Temporal slice (time series for single pixel)
        start_time = time.time()
        temporal_data = ds.isel(y=y_size//2, x=x_size//2).compute()
        metrics['temporal_read_time_s'] = time.time() - start_time
    
    return metrics


def compare_chunking_strategies(strategies: Dict[str, Dict],
                                dataset: xr.Dataset,
                                output_dir: Path) -> pd.DataFrame:
    """
    Compare multiple chunking strategies.
    
    Parameters:
    -----------
    strategies : Dict[str, Dict]
        Dictionary of strategy names and chunk configurations
    dataset : xr.Dataset
        Dataset to chunk
    output_dir : Path
        Directory to save Zarr datasets
    
    Returns:
    --------
    pd.DataFrame : Comparison results
    """
    results = {}
    
    for strategy_name, chunk_config in strategies.items():
        print(f"\nProcessing {strategy_name} strategy...")
        
        # Create output path
        output_path = output_dir / f'sentinel2_{strategy_name}.zarr'
        
        # Remove existing if present
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
        
        # Create Zarr with this strategy
        create_multitemporal_zarr(
            dataset=dataset,
            output_path=str(output_path),
            chunk_strategy=chunk_config
        )
        
        # Analyze performance
        metrics = analyze_zarr_performance(str(output_path))
        results[strategy_name] = metrics
    
    # Create comparison dataframe
    comparison_data = []
    for strategy, metrics in results.items():
        comparison_data.append({
            'Strategy': strategy,
            'Total Size (MB)': round(metrics['total_size_mb'], 2),
            'Avg Chunk Size (MB)': round(metrics['avg_chunk_size_mb'], 2),
            'Compression Ratio': round(metrics['avg_compression_ratio'], 2),
            'Number of Chunks': metrics['num_chunks'],
            'Spatial Read (s)': round(metrics.get('spatial_read_time_s', 0), 3),
            'Temporal Read (s)': round(metrics.get('temporal_read_time_s', 0), 3)
        })
    
    return pd.DataFrame(comparison_data)


def visualize_chunk_layout(chunk_strategy: Dict, 
                          data_shape: Dict) -> None:
    """
    Visualize how chunks divide the data array.
    
    Parameters:
    -----------
    chunk_strategy : Dict
        Chunking configuration
    data_shape : Dict
        Shape of the data array
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Spatial view (x-y plane at t=0)
    ax = axes[0]
    ax.set_title('Spatial Chunking (X-Y plane)')
    ax.set_xlabel('X dimension')
    ax.set_ylabel('Y dimension')
    
    # Draw chunk boundaries
    for y in range(0, data_shape['y'], chunk_strategy['y']):
        ax.axhline(y, color='red', linewidth=0.5)
    for x in range(0, data_shape['x'], chunk_strategy['x']):
        ax.axvline(x, color='red', linewidth=0.5)
    
    ax.set_xlim(0, data_shape['x'])
    ax.set_ylim(0, data_shape['y'])
    ax.invert_yaxis()
    
    # Add chunk size annotation
    ax.text(chunk_strategy['x']/2, chunk_strategy['y']/2, 
           f"{chunk_strategy['x']}×{chunk_strategy['y']}", 
           ha='center', va='center', fontsize=10, color='blue')
    
    # Temporal view (t-y plane at x=0)
    ax = axes[1]
    ax.set_title('Temporal Chunking (Time-Y plane)')
    ax.set_xlabel('Time dimension')
    ax.set_ylabel('Y dimension')
    
    # Draw chunk boundaries
    for t in range(0, data_shape['time'], chunk_strategy['time']):
        ax.axvline(t, color='red', linewidth=0.5)
    for y in range(0, data_shape['y'], chunk_strategy['y']):
        ax.axhline(y, color='red', linewidth=0.5)
    
    ax.set_xlim(0, data_shape['time'])
    ax.set_ylim(0, data_shape['y'])
    ax.invert_yaxis()
    
    # Add chunk size annotation
    ax.text(chunk_strategy['time']/2, chunk_strategy['y']/2, 
           f"{chunk_strategy['time']}×{chunk_strategy['y']}", 
           ha='center', va='center', fontsize=10, color='blue')
    
    plt.suptitle(f"Chunk Layout Visualization\nStrategy: time={chunk_strategy['time']}, y={chunk_strategy['y']}, x={chunk_strategy['x']}")
    plt.tight_layout()
    plt.show()
