"""
Compare WorldPop population raster datasets (OneDrive vs local raw) and plot regional density.

This module contains utilities to:
- load and clean population GeoTIFF rasters using rioxarray
- compute population density (people per km²) from per-cell counts
- plot population density for a specified region (example: Europe)
- compare total population sums between files in two directories and print a summary

Quick usage examples
- From the command line (run comparisons):
    python python_code/population/a_compare_worldpop_datasets.py

- From another script (import functions):
    from python_code.population.a_compare_worldpop_datasets import plot_population, compare_data
    plot_population(file_path=path, vmin=1, vmax=100, coarsen_factor=20, extent_eu=(-25,40,34,72))

Requirements / notes
- Expects `my_config.Dirs` to provide Path-like attributes `dir_population` and `dir_pop_raw`.
- Requires rioxarray (and its GDAL/native backends), xarray, numpy, matplotlib, cartopy, pandas, tqdm.
- Large rasters may be processed with dask (chunks=True) to avoid memory issues.

Common failure modes
- Missing native GDAL/netCDF libraries (pyogrio/fiona import errors) will break raster I/O.
- Very large rasters: plotting without coarsening can be very slow or memory-heavy.
"""

import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm
from my_config import Dirs
import pandas as pd
from tqdm import tqdm


def load_and_clean_raster(path, target_crs="EPSG:4326"):
    """
    Loads a raster, fixes NoData values, removes negatives, and standardizes coordinates.
    """
    print(f"Loading: {path.name}")

    # masked=True automatically converts 'NoData' (e.g., -3.4e38) to NaN
    da = rioxarray.open_rasterio(path, masked=True).squeeze()

    # Force negative values (artifacts) to 0 and fill NaNs with 0 for summation safety
    da = da.where(da >= 0, 0).fillna(0)

    # Standardize coordinate names
    if "x" in da.dims:
        da = da.rename({"x": "longitude", "y": "latitude"})

    # Ensure correct CRS
    if da.rio.crs is None or da.rio.crs.to_epsg() != 4326:
        print(f"  Reprojecting to {target_crs}...")
        da = da.rio.reproject(target_crs)
        da = da.where(da >= 0, 0).fillna(0)

    return da


def compute_density(da_counts):
    """
    Converts population counts to density (people/km²) based on latitude.
    """
    # 1. Calculate cell resolution in degrees
    res_lat = abs(da_counts.latitude[1] - da_counts.latitude[0]).item()
    res_lon = abs(da_counts.longitude[1] - da_counts.longitude[0]).item()

    # 2. Calculate cell dimensions in km
    km_per_deg = 111.32
    height_km = res_lat * km_per_deg
    width_km = res_lon * km_per_deg * np.cos(np.deg2rad(da_counts.latitude))

    # 3. Calculate Area and Density
    area_km2 = width_km * height_km
    density = da_counts / area_km2

    return density


def plot_population(file_path, vmin, vmax, coarsen_factor, extent_eu):
    """
    Orchestrates the loading, processing, and plotting for a single file.

    Parameters
    - file_path: Path-like to a GeoTIFF population raster
    - vmin, vmax: numeric limits for the log color scale (people/km²)
    - coarsen_factor: integer factor to aggregate the raster for faster plotting
    - extent_eu: tuple (west, east, south, north) bounding box in degrees

    This function displays a matplotlib/Cartopy plot and prints the total population
    inside the requested bounding box. It does not return a value.
    """
    # 1. Load
    pop = load_and_clean_raster(file_path)

    # 2. Subset to Extent
    # Using slice(max, min) for latitude because it's usually descending (North -> South)
    pop_subset = pop.sel(
        longitude=slice(extent_eu[0], extent_eu[1]),
        latitude=slice(extent_eu[3], extent_eu[2]),
    )

    # Fallback if latitude is ascending
    if pop_subset.size == 0:
        pop_subset = pop.sel(
            longitude=slice(extent_eu[0], extent_eu[1]),
            latitude=slice(extent_eu[2], extent_eu[3]),
        )

    # 3. Coarsen (Aggregating Sums)
    if coarsen_factor > 1:
        print(f"  Coarsening by factor {coarsen_factor}...")
        pop_subset = pop_subset.coarsen(
            latitude=coarsen_factor, longitude=coarsen_factor, boundary="trim"
        ).sum()

    # 4. Compute Density
    pop_density = compute_density(pop_subset)

    # 5. Plot
    fig, ax = plt.subplots(
        figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    img = pop_density.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        # enforce the same limits for both plots
        norm=LogNorm(vmin=vmin, vmax=vmax),
        add_colorbar=False,  # We add a custom one below to control layout better
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.set_title(f"Density: {file_path.name}\n(Coarseness: {coarsen_factor})")

    # Custom colorbar
    plt.colorbar(img, ax=ax, label="People / km²", shrink=0.8, extend="both")

    print(f"  Total Pop in View: {pop_subset.sum().item():,.0f}")
    plt.show()


def plot_data():
    """
    Simple driver function to demonstrate `plot_population`.

    Edit the inputs inside this function to point to the correct files and region. By
    default it will attempt to plot two files (OneDrive and local raw) for Europe.
    """
    # --- USER INPUTS ---
    files = [
        Dirs.dir_pop_raw / "global_f_0_2015_1km.tif",
        Dirs.dir_population / "global_f_0_2015_1km.tif",
    ]

    # Region: (West, East, South, North)
    europe_extent = (-25, 40, 34, 72)

    # Visualization Parameters
    # Set these manually so both plots share the EXACT same scale
    plot_vmin = 1  # Minimum density (people/km2)
    plot_vmax = 100  # Maximum density

    # Processing Parameter
    # Increase this to speed up plotting (e.g., 5, 10, or 20)
    coarseness = 20

    # --- EXECUTION ---
    for file_path in files:
        plot_population(
            file_path=file_path,
            vmin=plot_vmin,
            vmax=plot_vmax,
            coarsen_factor=coarseness,
            extent_eu=europe_extent,
        )


def get_total_population(file_path):
    """
    Opens a raster safely and calculates the total sum.
    Returns None if file is invalid.
    """
    try:
        # 1. Open with masked=True to handle NoData values (the -3.4e38 issue)
        # chunks=True allows processing files larger than RAM
        da = rioxarray.open_rasterio(file_path, masked=True, chunks=True).squeeze()

        # 2. Force negatives to 0 (cleaning artifacts)
        da = da.where(da >= 0, 0)

        # 3. Sum and Compute
        # .compute() forces Dask to calculate the number NOW
        # .item() converts the numpy/dask scalar to a standard Python float
        total_pop = da.sum().compute().item()

        return total_pop
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None


def compare_data():
    """
    Compare total populations for matching files in OneDrive vs local HD.

    The function looks for .tif files in `Dirs.dir_population` (OneDrive) and tries to
    find files with the same name in `Dirs.dir_pop_raw`. For each matching pair it
    computes the total population and prints a small summary table showing counts and
    percent differences.
    """
    # 1. Get list of .tif files in the OneDrive folder
    # Change pattern to "*.tif" if they are standard tiffs
    files_onedrive = list(Dirs.dir_population.glob("*.tif"))

    results = []

    print(f"Found {len(files_onedrive)} files in OneDrive. Starting comparison...")

    for file_path_a in tqdm(files_onedrive):
        filename = file_path_a.name

        # 2. Construct path to the matching file on the Hard Drive
        file_path_b = Dirs.dir_pop_raw / filename

        if not file_path_b.exists():
            print(f"Missing in HD: {filename}")
            continue

        # 3. Calculate Totals
        pop_a = get_total_population(file_path_a)  # OneDrive
        pop_b = get_total_population(file_path_b)  # HD

        if pop_a is None or pop_b is None:
            continue

        # 4. Compute Stats
        diff = pop_b - pop_a

        # Avoid division by zero
        if pop_a != 0:
            pct_change = (diff / pop_a) * 100
        else:
            pct_change = 0.0 if diff == 0 else float("inf")

        results.append(
            {
                "File": filename,
                "OneDrive_Pop": pop_a,
                "HD_Pop": pop_b,
                "Diff_Count": diff,
                "Diff_Percent": pct_change,
            }
        )

    # 5. Display Results
    df = pd.DataFrame(results)

    # Format for nice printing
    pd.options.display.float_format = "{:,.2f}".format

    print("\n--- COMPARISON RESULTS ---")
    if not df.empty:
        # Sort by biggest absolute difference
        df["Abs_Diff"] = df["Diff_Count"].abs()
        df = df.sort_values("Abs_Diff", ascending=False).drop(columns="Abs_Diff")

        print(df.to_string(index=False))

        # Optional: Save to CSV
        # df.to_csv("population_comparison.csv", index=False)
    else:
        print("No matching files found or processed.")


if __name__ == "__main__":
    # plot_data()
    compare_data()

    """
    --- COMPARISON RESULTS ---
                       File  OneDrive_Pop        HD_Pop    Diff_Count  Diff_Percent
    global_f_0_2015_1km.tif 68,458,592.00 64,590,904.00 -3,867,688.00         -5.65
    global_m_0_2015_1km.tif 72,927,488.00 69,099,288.00 -3,828,200.00         -5.25
    global_f_0_2016_1km.tif 68,084,512.00 65,315,532.00 -2,768,980.00         -4.07
    global_m_0_2016_1km.tif 72,438,592.00 69,819,432.00 -2,619,160.00         -3.62
    global_f_0_2017_1km.tif 68,467,288.00 65,912,672.00 -2,554,616.00         -3.73
    global_m_0_2017_1km.tif 72,766,336.00 70,396,944.00 -2,369,392.00         -3.26
    global_m_0_2020_1km.tif 69,671,200.00 71,934,224.00  2,263,024.00          3.25
    global_f_0_2020_1km.tif 65,832,080.00 67,476,128.00  1,644,048.00          2.50
    global_f_0_2018_1km.tif 67,980,384.00 66,424,240.00 -1,556,144.00         -2.29
    global_m_0_2018_1km.tif 72,092,584.00 70,904,352.00 -1,188,232.00         -1.65
    global_m_0_2019_1km.tif 70,465,784.00 71,367,968.00    902,184.00          1.28
    global_f_0_2019_1km.tif 66,552,560.00 66,902,240.00    349,680.00          0.53
    """
