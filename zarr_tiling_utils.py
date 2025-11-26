"""
Zarr Tiling Utilities for EOPF-101 Notebooks

This module provides reusable functions for zarr rechunking, tiling performance analysis,
and optimization experiments. Extracted and adapted from eopf-explorer geozarr.py.

Key capabilities:
- Rechunking Zarr datasets with different strategies
- Calculating optimal chunk sizes for tiling workloads
- Performance benchmarking utilities
- Overview/pyramid level calculations
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import psutil
import os


@dataclass
class ChunkingStrategy:
    """Configuration for a chunking strategy."""
    name: str
    chunk_size: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'chunk_size': self.chunk_size,
            'description': self.description
        }


@dataclass
class PerformanceMetrics:
    """Container for tiling performance measurements."""
    chunk_size: int
    tile_generation_time: float
    memory_usage_mb: float
    http_requests: int
    data_transferred_mb: float
    zoom_level: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'chunk_size': self.chunk_size,
            'tile_generation_time_s': self.tile_generation_time,
            'memory_usage_mb': self.memory_usage_mb,
            'http_requests': self.http_requests,
            'data_transferred_mb': self.data_transferred_mb,
            'zoom_level': self.zoom_level
        }


def calculate_aligned_chunk_size(dimension: int, target_chunk: int) -> int:
    """
    Calculate chunk size that evenly divides the dimension.

    This ensures chunks align properly with data boundaries, avoiding
    partial chunks which can degrade performance.

    Parameters
    ----------
    dimension : int
        Size of the data dimension
    target_chunk : int
        Desired chunk size

    Returns
    -------
    int
        Aligned chunk size that evenly divides dimension

    Examples
    --------
    >>> calculate_aligned_chunk_size(10980, 1024)
    915  # 10980 / 12 = 915
    >>> calculate_aligned_chunk_size(5490, 512)
    915  # 5490 / 6 = 915
    """
    if target_chunk >= dimension:
        return dimension

    # Find divisor of dimension closest to target_chunk
    best_chunk = dimension
    min_diff = abs(dimension - target_chunk)

    for divisor in range(1, int(np.sqrt(dimension)) + 1):
        if dimension % divisor == 0:
            # Check both divisor and its complement
            for candidate in [divisor, dimension // divisor]:
                diff = abs(candidate - target_chunk)
                if diff < min_diff and candidate <= target_chunk * 1.5:
                    best_chunk = candidate
                    min_diff = diff

    return best_chunk


def calculate_optimal_chunks_for_tiling(
    width: int,
    height: int,
    tile_size: int = 256,
    target_zoom_levels: Optional[List[int]] = None
) -> Dict[int, Tuple[int, int]]:
    """
    Calculate optimal chunk sizes for different zoom levels.

    For web mapping applications, chunk size should align with tile access
    patterns at different zoom levels to minimize HTTP requests.

    Parameters
    ----------
    width : int
        Dataset width in pixels
    height : int
        Dataset height in pixels
    tile_size : int, default 256
        Web tile size (typically 256 or 512)
    target_zoom_levels : List[int], optional
        Specific zoom levels to optimize for

    Returns
    -------
    Dict[int, Tuple[int, int]]
        Mapping of zoom level to (y_chunk, x_chunk)

    Examples
    --------
    >>> calculate_optimal_chunks_for_tiling(10980, 10980, tile_size=256)
    {8: (1830, 1830), 12: (915, 915), 16: (456, 456)}
    """
    if target_zoom_levels is None:
        target_zoom_levels = [8, 12, 16]

    optimal_chunks = {}

    for zoom in target_zoom_levels:
        # At each zoom level, calculate how many tiles cover the dataset
        tiles_at_zoom = 2 ** zoom
        pixels_per_tile = width / tiles_at_zoom

        # Target chunk size should be a multiple of tile_size
        target_chunk = max(tile_size, int(pixels_per_tile))

        # Align chunk size to evenly divide dimensions
        chunk_y = calculate_aligned_chunk_size(height, target_chunk)
        chunk_x = calculate_aligned_chunk_size(width, target_chunk)

        optimal_chunks[zoom] = (chunk_y, chunk_x)

    return optimal_chunks


def rechunk_dataset(
    ds: xr.Dataset,
    chunk_size: int,
    output_path: str,
    variables: Optional[List[str]] = None,
    spatial_dims: Tuple[str, str] = ('y', 'x')
) -> xr.Dataset:
    """
    Rechunk an xarray Dataset with a new spatial chunk size.

    This creates a new Zarr store with the specified chunking. Useful for
    experimenting with different chunk strategies.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset to rechunk
    chunk_size : int
        New spatial chunk size (applied to both y and x)
    output_path : str
        Path for output Zarr store
    variables : List[str], optional
        Specific variables to rechunk (default: all data variables)
    spatial_dims : Tuple[str, str], default ('y', 'x')
        Names of spatial dimensions

    Returns
    -------
    xr.Dataset
        Rechunked dataset

    Examples
    --------
    >>> ds_rechunked = rechunk_dataset(ds, chunk_size=512, output_path='./test.zarr')
    """
    if variables is None:
        variables = list(ds.data_vars.keys())

    # Create chunking dict
    chunks = {spatial_dims[0]: chunk_size, spatial_dims[1]: chunk_size}

    # Apply rechunking
    ds_rechunked = ds[variables].chunk(chunks)

    # Write to Zarr
    print(f"Writing rechunked dataset to {output_path}")
    print(f"  Chunk size: {chunk_size}x{chunk_size}")
    print(f"  Variables: {variables}")

    ds_rechunked.to_zarr(
        output_path,
        mode='w',
        consolidated=True,
        compute=True
    )

    # Reload to verify
    ds_reloaded = xr.open_zarr(output_path, chunks='auto')
    print(f"✅ Rechunking complete. New chunks: {ds_reloaded[variables[0]].chunks}")

    return ds_reloaded


def benchmark_tile_generation(
    ds: xr.Dataset,
    tile_size: int,
    zoom_level: int,
    num_tiles: int = 10,
    bands: Optional[List[str]] = None
) -> PerformanceMetrics:
    """
    Benchmark tile generation performance for a given chunk configuration.

    Measures:
    - Tile generation time
    - Memory usage
    - Estimated HTTP requests
    - Data transferred

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to generate tiles from
    tile_size : int
        Size of tiles to generate
    zoom_level : int
        Zoom level for tile generation
    num_tiles : int, default 10
        Number of tiles to generate for averaging
    bands : List[str], optional
        Specific bands to use

    Returns
    -------
    PerformanceMetrics
        Performance measurements

    Examples
    --------
    >>> metrics = benchmark_tile_generation(ds, tile_size=256, zoom_level=12)
    >>> print(f"Avg time per tile: {metrics.tile_generation_time:.3f}s")
    """
    if bands is None:
        bands = list(ds.data_vars.keys())[:3]  # Use first 3 bands

    # Get memory before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # Generate tiles and measure time
    times = []
    for i in range(num_tiles):
        # Random tile coordinates
        y_offset = np.random.randint(0, max(1, ds.dims['y'] - tile_size))
        x_offset = np.random.randint(0, max(1, ds.dims['x'] - tile_size))

        start = time.time()
        tile_data = ds[bands].isel(
            y=slice(y_offset, y_offset + tile_size),
            x=slice(x_offset, x_offset + tile_size)
        ).compute()
        times.append(time.time() - start)

    # Get memory after
    mem_after = process.memory_info().rss / (1024 * 1024)
    memory_delta = mem_after - mem_before

    # Estimate HTTP requests (chunks accessed)
    chunk_y, chunk_x = ds[bands[0]].chunks[0][0], ds[bands[0]].chunks[1][0]
    chunks_per_tile = np.ceil(tile_size / chunk_y) * np.ceil(tile_size / chunk_x)
    http_requests = int(chunks_per_tile * len(bands))

    # Estimate data transferred
    bytes_per_chunk = chunk_y * chunk_x * ds[bands[0]].dtype.itemsize
    data_transferred_mb = (bytes_per_chunk * http_requests) / (1024 * 1024)

    avg_time = np.mean(times)

    return PerformanceMetrics(
        chunk_size=chunk_y,
        tile_generation_time=avg_time,
        memory_usage_mb=memory_delta,
        http_requests=http_requests,
        data_transferred_mb=data_transferred_mb,
        zoom_level=zoom_level
    )


def calculate_overview_levels(
    native_width: int,
    native_height: int,
    min_dimension: int = 256,
    tile_width: int = 256
) -> List[Dict[str, int]]:
    """
    Calculate overview/pyramid levels following COG /2 downsampling logic.

    This is used for creating multi-scale datasets optimized for different
    zoom levels.

    Parameters
    ----------
    native_width : int
        Width of native resolution data
    native_height : int
        Height of native resolution data
    min_dimension : int, default 256
        Stop creating overviews when dimension is smaller than this
    tile_width : int, default 256
        Tile width for TMS compatibility

    Returns
    -------
    List[Dict[str, int]]
        List of overview level dictionaries

    Examples
    --------
    >>> levels = calculate_overview_levels(10980, 10980)
    >>> for level in levels:
    ...     print(f"Level {level['level']}: {level['width']}x{level['height']}")
    Level 0: 10980x10980
    Level 1: 5490x5490
    Level 2: 2745x2745
    Level 3: 1372x1372
    Level 4: 686x686
    Level 5: 343x343
    """
    overview_levels = []
    level = 0
    current_width = native_width
    current_height = native_height

    while min(current_width, current_height) >= min_dimension:
        # Calculate zoom level for TMS compatibility
        zoom_for_width = max(0, int(np.ceil(np.log2(current_width / tile_width))))
        zoom_for_height = max(0, int(np.ceil(np.log2(current_height / tile_width))))
        zoom = max(zoom_for_width, zoom_for_height)

        overview_levels.append({
            'level': level,
            'zoom': zoom,
            'width': current_width,
            'height': current_height,
            'scale_factor': 2**level
        })

        level += 1
        current_width = native_width // (2**level)
        current_height = native_height // (2**level)

    return overview_levels


def downsample_2d_array(
    data: np.ndarray,
    target_height: int,
    target_width: int,
    method: str = 'mean'
) -> np.ndarray:
    """
    Downsample a 2D array to target dimensions.

    Parameters
    ----------
    data : np.ndarray
        Source 2D array
    target_height : int
        Target height
    target_width : int
        Target width
    method : str, default 'mean'
        Downsampling method ('mean', 'nearest', 'max')

    Returns
    -------
    np.ndarray
        Downsampled array

    Examples
    --------
    >>> data = np.random.rand(1000, 1000)
    >>> downsampled = downsample_2d_array(data, 500, 500)
    >>> downsampled.shape
    (500, 500)
    """
    from skimage.transform import resize

    if method == 'mean':
        return resize(data, (target_height, target_width), anti_aliasing=True, preserve_range=True)
    elif method == 'nearest':
        return resize(data, (target_height, target_width), order=0, preserve_range=True)
    elif method == 'max':
        # Block max pooling
        block_y = data.shape[0] // target_height
        block_x = data.shape[1] // target_width
        result = np.zeros((target_height, target_width), dtype=data.dtype)
        for i in range(target_height):
            for j in range(target_width):
                result[i, j] = np.max(data[i*block_y:(i+1)*block_y, j*block_x:(j+1)*block_x])
        return result
    else:
        raise ValueError(f"Unknown method: {method}")


def print_performance_summary(results: Dict[str, List[PerformanceMetrics]]) -> None:
    """
    Print a formatted summary of benchmarking results.

    Parameters
    ----------
    results : Dict[str, List[PerformanceMetrics]]
        Results from compare_chunking_strategies()
    """
    print("\\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    for strategy_name, metrics_list in results.items():
        print(f"\\n{strategy_name.upper()}:")
        print(f"{'Zoom':<8} {'Time (s)':<12} {'Memory (MB)':<15} {'HTTP Reqs':<12} {'Data (MB)':<12}")
        print("-" * 80)

        for m in metrics_list:
            print(f"{m.zoom_level:<8} {m.tile_generation_time:<12.3f} "
                  f"{m.memory_usage_mb:<15.2f} {m.http_requests:<12} "
                  f"{m.data_transferred_mb:<12.2f}")

    print("="*80)
    

def visualize_chunks_and_tiles(ds, tile_size=256, num_sample_tiles=4):
    """
    Visualize spatial relationship between Zarr chunks and tile requests.
    
    Shows chunk boundaries on actual data with example tile requests overlaid.
    """
    import matplotlib.patches as mpatches
    
    # Get chunk and dimension info from the ACTUAL dataset chunks
    chunk_y = ds['b04'].chunks[0][0]
    chunk_x = ds['b04'].chunks[1][0]
    height, width = ds.dims['y'], ds.dims['x']
    
    print(f"📐 Using ACTUAL chunk dimensions from dataset:")
    print(f"   Chunk size: {chunk_y}×{chunk_x} pixels (from ds['b04'].chunks)")
    print(f"   Dataset size: {height}×{width} pixels")
    print(f"   Tile size for demo: {tile_size}×{tile_size} pixels\n")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    # Display band data as background
    band_data = ds['b04'].values
    
    # Simple contrast stretch
    p2, p98 = np.percentile(band_data[band_data > 0], [2, 98])
    band_stretched = np.clip((band_data - p2) / (p98 - p2), 0, 1)
    
    ax.imshow(band_stretched, cmap='gray', extent=[0, width, height, 0], alpha=0.7)
    
    # Draw chunk grid using ACTUAL chunk dimensions (NOT tile_size)
    for i in range(0, height + 1, chunk_y):
        ax.axhline(y=i, color='cyan', linewidth=2, alpha=0.8, linestyle='-')
    for j in range(0, width + 1, chunk_x):
        ax.axvline(x=j, color='cyan', linewidth=2, alpha=0.8, linestyle='-')
    
    # Add chunk labels
    for i_chunk in range(int(np.ceil(height / chunk_y))):
        for j_chunk in range(int(np.ceil(width / chunk_x))):
            y_pos = i_chunk * chunk_y + chunk_y / 2
            x_pos = j_chunk * chunk_x + chunk_x / 2
            if y_pos < height and x_pos < width:
                ax.text(x_pos, y_pos, f'C{i_chunk},{j_chunk}', 
                       ha='center', va='center', fontsize=8, color='cyan',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Draw example tile requests
    np.random.seed(112)
    tile_colors = ['red', 'yellow', 'lime', 'magenta']
    tile_info = []
    
    for i in range(num_sample_tiles):
        # Random tile position
        margin = tile_size
        x_start = np.random.randint(margin, width - tile_size - margin)
        y_start = np.random.randint(margin, height - tile_size - margin)
        
        # Calculate which chunks this tile intersects
        chunk_y_start = int(y_start / chunk_y)
        chunk_y_end = int((y_start + tile_size - 1) / chunk_y)
        chunk_x_start = int(x_start / chunk_x)
        chunk_x_end = int((x_start + tile_size - 1) / chunk_x)
        
        chunks_accessed = (chunk_y_end - chunk_y_start + 1) * (chunk_x_end - chunk_x_start + 1)
        tile_info.append({
            'tile': i+1,
            'chunks': chunks_accessed,
            'chunk_range': f'Y:{chunk_y_start}-{chunk_y_end}, X:{chunk_x_start}-{chunk_x_end}'
        })
        
        # Draw tile boundary
        rect = mpatches.Rectangle(
            (x_start, y_start), tile_size, tile_size,
            linewidth=4, edgecolor=tile_colors[i], facecolor='none',
            linestyle='--', label=f'Tile {i+1} ({chunks_accessed} chunks)'
        )
        ax.add_patch(rect)
        
        # Add tile label
        ax.text(x_start + tile_size/2, y_start + tile_size/2, f'T{i+1}',
               color=tile_colors[i], fontsize=16, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    ax.set_title(f'Chunk Grid (cyan, {chunk_y}×{chunk_x}px) with {tile_size}×{tile_size}px Tile Requests', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    
    plt.tight_layout()
    plt.show()
    
    # Print tile information
    print("\n📊 Tile-to-Chunk Mapping:")
    print(f"{'Tile':<8} {'Chunks Accessed':<18} {'Chunk Range'}")
    print("=" * 60)
    for info in tile_info:
        print(f"T{info['tile']:<7} {info['chunks']:<18} {info['chunk_range']}")
    
    return tile_info

def compare_chunking_strategies(ds_variants, tile_size=256, tile_x=512, tile_y=512):
    """
    Compare how a single tile request maps to chunks in different strategies.
    Calculates chunks accessed and data transfer volume for each strategy.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()  # Flatten 2D array to 1D for easy indexing

    results = {}

    for idx, (name, ds) in enumerate(ds_variants.items()):
        ax = axes[idx]

        # Get chunk info
        chunk_y = ds['b04'].chunks[0][0]
        chunk_x = ds['b04'].chunks[1][0]
        height, width = ds.dims['y'], ds.dims['x']

        # Get dtype size
        dtype_size = ds['b04'].dtype.itemsize
        num_bands = len([v for v in ds.data_vars if v.startswith('b')])

        # Display data
        band_data = ds['b04'].values
        p2, p98 = np.percentile(band_data[band_data > 0], [2, 98])
        band_stretched = np.clip((band_data - p2) / (p98 - p2), 0, 1)
        ax.imshow(band_stretched, cmap='gray', extent=[0, width, height, 0], alpha=0.5)

        # Draw chunk grid
        for i in range(0, height + 1, chunk_y):
            ax.axhline(y=i, color='cyan', linewidth=1.5, alpha=0.7)
        for j in range(0, width + 1, chunk_x):
            ax.axvline(x=j, color='cyan', linewidth=1.5, alpha=0.7)

        # Draw the test tile
        import matplotlib.patches as mpatches
        rect = mpatches.Rectangle(
            (tile_x, tile_y), tile_size, tile_size,
            linewidth=4, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

        # Calculate chunks accessed
        chunk_y_start = int(tile_y / chunk_y)
        chunk_y_end = int((tile_y + tile_size - 1) / chunk_y)
        chunk_x_start = int(tile_x / chunk_x)
        chunk_x_end = int((tile_x + tile_size - 1) / chunk_x)

        chunks_accessed = (chunk_y_end - chunk_y_start + 1) * (chunk_x_end - chunk_x_start + 1)

        # Calculate data transfer volumes
        # Actual tile data needed
        tile_data_mb = (tile_size * tile_size * dtype_size * num_bands) / (1024 * 1024)

        # Data that must be transferred (full chunks)
        chunk_size_bytes = chunk_y * chunk_x * dtype_size * num_bands
        transferred_mb = (chunks_accessed * chunk_size_bytes) / (1024 * 1024)

        # Overhead ratio
        overhead_ratio = transferred_mb / tile_data_mb if tile_data_mb > 0 else 0

        # Highlight accessed chunks
        for cy in range(chunk_y_start, chunk_y_end + 1):
            for cx in range(chunk_x_start, chunk_x_end + 1):
                rect_chunk = mpatches.Rectangle(
                    (cx * chunk_x, cy * chunk_y), chunk_x, chunk_y,
                    facecolor='red', alpha=0.2, edgecolor='red', linewidth=2
                )
                ax.add_patch(rect_chunk)

        ax.set_title(f'{name}\n{chunks_accessed} chunk{"s" if chunks_accessed > 1 else ""} accessed | {transferred_mb:.2f} MBits transferred',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        results[name] = {
            'chunks_accessed': chunks_accessed,
            'tile_data_mb': tile_data_mb,
            'transferred_mb': transferred_mb,
            'overhead_ratio': overhead_ratio,
            'chunk_size': f'{chunk_y}×{chunk_x}'
        }
    
    plt.tight_layout()
    plt.show()

    # Summary
    print(f"\n🎯 Chunk Access and Data Transfer Comparison for {tile_size}×{tile_size}px Tile:")
    print("=" * 100)
    print(f"{'Strategy':<30} {'Chunk Size':<12} {'Chunks':<8} {'Tile Data':<12} {'Transferred':<14} {'Overhead':<10} {'Efficiency'}")
    print("-" * 100)

    for name, metrics in results.items():
        chunks = metrics['chunks_accessed']
        overhead = metrics['overhead_ratio']

        # Efficiency considers both chunk count (HTTP requests) and data overhead
        if chunks == 1 and overhead <= 2.0:
            efficiency_label = "✅ Optimal"
        elif chunks <= 4 and overhead <= 4.0:
            efficiency_label = "✅ Good"
        elif chunks <= 4 and overhead <= 8.0:
            efficiency_label = "⚠️ Acceptable"
        elif chunks > 4:
            efficiency_label = "⚠️ Many requests"
        elif overhead > 8.0:
            efficiency_label = "❌ High overhead"
        else:
            efficiency_label = "❌ Inefficient"

        print(f"{name:<20} {metrics['chunk_size']:<12} {chunks:<8} "
              f"{metrics['tile_data_mb']:>8.2f} MB  {metrics['transferred_mb']:>10.2f} MB  "
              f"{metrics['overhead_ratio']:>8.2f}x  {efficiency_label}")

    print("=" * 100)
    print(f"\n💡 Interpretation:")
    print(f"   • Tile Data: Actual data needed for the {tile_size}×{tile_size}px tile")
    print("   • Transferred: Total data that must be read from storage (full chunks)")
    print("   • Overhead: Ratio of transferred/needed (lower is better, 1.0x is perfect)")
    print("\n   Efficiency considers BOTH chunk count (HTTP requests) AND data overhead:")
    print("   • ✅ Optimal: 1 chunk + low overhead (≤2x)")
    print("   • ✅ Good: 2-4 chunks + low overhead (≤2x)")
    print("   • ⚠️ Acceptable: Trade-offs between chunk count and overhead")
    print("   • ⚠️ Many requests: Many small chunks but minimal wasted data")
    print("   • ❌ High overhead: Reading >10x more data than needed")
    print("   • ❌ Inefficient: Poor performance on both metrics")

    return results


if __name__ == "__main__":
    # Example usage
    print("Zarr Tiling Utilities for EOPF-101")
    print("="*50)
    print("\\nExample: Calculate optimal chunks for 10980x10980 S2 scene")

    optimal = calculate_optimal_chunks_for_tiling(10980, 10980, tile_size=256)
    for zoom, chunks in optimal.items():
        print(f"  Zoom {zoom}: {chunks[0]}x{chunks[1]} chunks")

    print("\\nExample: Calculate overview levels")
    levels = calculate_overview_levels(10980, 10980)
    for level in levels[:5]:
        print(f"  Level {level['level']}: {level['width']}x{level['height']} "
              f"(scale 1:{level['scale_factor']})")
