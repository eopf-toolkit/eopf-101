"""
Helpers for Zarr chunking vs. web-tile visualization in EOPF-101 notebooks.

Used by `58_rio_tiler_chunks_and_tiles.ipynb` for chunk grid overlays,
chunk-strategy comparison, and rio-tiler `XarrayReader.tile()` benchmarks.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from rio_tiler.io import XarrayReader


def benchmark_rio_tiler(
    da: xr.DataArray,
    tile_x: int,
    tile_y: int,
    zoom: int,
    tilesize: int = 256,
    n_warm: int = 1,
    n_iter: int = 5,
) -> List[float]:
    """Time rio-tiler tile serving via ``XarrayReader.tile()`` (seconds per run)."""
    with XarrayReader(da) as src:
        for _ in range(n_warm):
            src.tile(tile_x, tile_y, zoom, tilesize=tilesize)
        times: List[float] = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            src.tile(tile_x, tile_y, zoom, tilesize=tilesize)
            times.append(time.perf_counter() - t0)
    return times


def visualize_chunks_and_tiles(
    ds: xr.Dataset, tile_size: int = 256, num_sample_tiles: int = 4
) -> List[Dict[str, Any]]:
    """
    Visualize spatial relationship between Zarr chunks and tile requests.

    Shows chunk boundaries on actual data with example tile requests overlaid.
    """
    # Get chunk and dimension info from the ACTUAL dataset chunks
    chunk_y = ds["b04"].chunks[0][0]
    chunk_x = ds["b04"].chunks[1][0]
    height, width = int(ds.sizes["y"]), int(ds.sizes["x"])

    print("📐 Using ACTUAL chunk dimensions from dataset:")
    print(f"   Chunk size: {chunk_y}×{chunk_x} pixels (from ds['b04'].chunks)")
    print(f"   Dataset size: {height}×{width} pixels")
    print(f"   Tile size for demo: {tile_size}×{tile_size} pixels\n")

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    band_data = ds["b04"].values
    p2, p98 = np.percentile(band_data[band_data > 0], [2, 98])
    band_stretched = np.clip((band_data - p2) / (p98 - p2), 0, 1)

    ax.imshow(band_stretched, cmap="gray", extent=[0, width, height, 0], alpha=0.7)

    for i in range(0, height + 1, chunk_y):
        ax.axhline(y=i, color="cyan", linewidth=2, alpha=0.8, linestyle="-")
    for j in range(0, width + 1, chunk_x):
        ax.axvline(x=j, color="cyan", linewidth=2, alpha=0.8, linestyle="-")

    for i_chunk in range(int(np.ceil(height / chunk_y))):
        for j_chunk in range(int(np.ceil(width / chunk_x))):
            y_pos = i_chunk * chunk_y + chunk_y / 2
            x_pos = j_chunk * chunk_x + chunk_x / 2
            if y_pos < height and x_pos < width:
                ax.text(
                    x_pos,
                    y_pos,
                    f"C{i_chunk},{j_chunk}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="cyan",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
                )

    np.random.seed(112)
    tile_colors = ["red", "yellow", "lime", "magenta"]
    tile_info: List[Dict[str, Any]] = []

    for i in range(num_sample_tiles):
        margin = tile_size
        x_start = np.random.randint(margin, width - tile_size - margin)
        y_start = np.random.randint(margin, height - tile_size - margin)

        chunk_y_start = int(y_start / chunk_y)
        chunk_y_end = int((y_start + tile_size - 1) / chunk_y)
        chunk_x_start = int(x_start / chunk_x)
        chunk_x_end = int((x_start + tile_size - 1) / chunk_x)

        chunks_accessed = (chunk_y_end - chunk_y_start + 1) * (
            chunk_x_end - chunk_x_start + 1
        )
        tile_info.append(
            {
                "tile": i + 1,
                "chunks": chunks_accessed,
                "chunk_range": f"Y:{chunk_y_start}-{chunk_y_end}, X:{chunk_x_start}-{chunk_x_end}",
            }
        )

        rect = mpatches.Rectangle(
            (x_start, y_start),
            tile_size,
            tile_size,
            linewidth=4,
            edgecolor=tile_colors[i],
            facecolor="none",
            linestyle="--",
            label=f"Tile {i + 1} ({chunks_accessed} chunks)",
        )
        ax.add_patch(rect)

        ax.text(
            x_start + tile_size / 2,
            y_start + tile_size / 2,
            f"T{i + 1}",
            color=tile_colors[i],
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.8),
        )

    ax.set_title(
        f"Chunk Grid (cyan, {chunk_y}×{chunk_x}px) with {tile_size}×{tile_size}px Tile Requests",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    plt.tight_layout()
    plt.show()

    print("\n📊 Tile-to-Chunk Mapping:")
    print(f"{'Tile':<8} {'Chunks Accessed':<18} {'Chunk Range'}")
    print("=" * 60)
    for info in tile_info:
        print(f"T{info['tile']:<7} {info['chunks']:<18} {info['chunk_range']}")

    return tile_info


def compare_chunking_strategies(
    ds_variants: Dict[str, xr.Dataset],
    tile_size: int = 256,
    tile_x: int = 512,
    tile_y: int = 512,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare how a single tile request maps to chunks in different strategies.
    Calculates chunks accessed and data transfer volume for each strategy.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    results: Dict[str, Dict[str, Any]] = {}

    for idx, (name, ds) in enumerate(ds_variants.items()):
        ax = axes[idx]

        chunk_y = ds["b04"].chunks[0][0]
        chunk_x = ds["b04"].chunks[1][0]
        height, width = int(ds.sizes["y"]), int(ds.sizes["x"])

        dtype_size = ds["b04"].dtype.itemsize
        num_bands = len([v for v in ds.data_vars if v.startswith("b")])

        band_data = ds["b04"].values
        p2, p98 = np.percentile(band_data[band_data > 0], [2, 98])
        band_stretched = np.clip((band_data - p2) / (p98 - p2), 0, 1)
        ax.imshow(band_stretched, cmap="gray", extent=[0, width, height, 0], alpha=0.5)

        for i in range(0, height + 1, chunk_y):
            ax.axhline(y=i, color="cyan", linewidth=1.5, alpha=0.7)
        for j in range(0, width + 1, chunk_x):
            ax.axvline(x=j, color="cyan", linewidth=1.5, alpha=0.7)

        rect = mpatches.Rectangle(
            (tile_x, tile_y),
            tile_size,
            tile_size,
            linewidth=4,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

        chunk_y_start = int(tile_y / chunk_y)
        chunk_y_end = int((tile_y + tile_size - 1) / chunk_y)
        chunk_x_start = int(tile_x / chunk_x)
        chunk_x_end = int((tile_x + tile_size - 1) / chunk_x)

        chunks_accessed = (chunk_y_end - chunk_y_start + 1) * (
            chunk_x_end - chunk_x_start + 1
        )

        tile_data_mb = (tile_size * tile_size * dtype_size * num_bands) / (1024 * 1024)

        chunk_size_bytes = chunk_y * chunk_x * dtype_size * num_bands
        transferred_mb = (chunks_accessed * chunk_size_bytes) / (1024 * 1024)

        overhead_ratio = transferred_mb / tile_data_mb if tile_data_mb > 0 else 0

        for cy in range(chunk_y_start, chunk_y_end + 1):
            for cx in range(chunk_x_start, chunk_x_end + 1):
                rect_chunk = mpatches.Rectangle(
                    (cx * chunk_x, cy * chunk_y),
                    chunk_x,
                    chunk_y,
                    facecolor="red",
                    alpha=0.2,
                    edgecolor="red",
                    linewidth=2,
                )
                ax.add_patch(rect_chunk)

        ax.set_title(
            f"{name}\n{chunks_accessed} chunk{'s' if chunks_accessed > 1 else ''} "
            f"accessed | {transferred_mb:.2f} MB transferred",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        results[name] = {
            "chunks_accessed": chunks_accessed,
            "tile_data_mb": tile_data_mb,
            "transferred_mb": transferred_mb,
            "overhead_ratio": overhead_ratio,
            "chunk_size": f"{chunk_y}×{chunk_x}",
        }

    plt.tight_layout()
    plt.show()

    print(
        f"\n🎯 Chunk Access and Data Transfer Comparison for {tile_size}×{tile_size}px Tile:"
    )
    print("=" * 100)
    print(
        f"{'Strategy':<30} {'Chunk Size':<12} {'Chunks':<8} {'Tile Data':<12} "
        f"{'Transferred':<14} {'Overhead':<10} {'Efficiency'}"
    )
    print("-" * 100)

    for name, metrics in results.items():
        chunks = metrics["chunks_accessed"]
        overhead = metrics["overhead_ratio"]

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

        print(
            f"{name:<30} {metrics['chunk_size']:<12} {chunks:<8} "
            f"{metrics['tile_data_mb']:>8.2f} MB  {metrics['transferred_mb']:>10.2f} MB  "
            f"{overhead:>8.2f}x  {efficiency_label}"
        )

    print("=" * 100)
    print("\n💡 Interpretation:")
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
