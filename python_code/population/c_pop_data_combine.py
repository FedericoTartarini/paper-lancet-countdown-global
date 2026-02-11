"""
Combine Population Data Module

This module combines:
1. Historical Population (Pre-2000): Coarse resolution (0.25°), upsampled to ERA5-Land (0.1°).
2. Modern Population (2000-2025): High resolution WorldPop, already regridded to ERA5-Land.

It handles the 'Ocean Pixel' issue during upsampling and stitches the time series together.
"""

import os
from pathlib import Path
import warnings

import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path.cwd()

from my_config import Vars, VarsWorldPop, DirsLocal, FilesLocal

# Suppress simple warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


def upsample_population_with_mask(pop_data, era_template):
    """
    Upsamples Coarse Population to Fine ERA5 Grid.
    PREVENTS 'Ocean People' by masking with ERA5 land-sea mask and
    conserving total population count.
    """
    # 1. Shift Longitude (0..360 -> -180..180) to match ERA5
    # (Only if necessary, assumes pop_data might be 0-360)
    if pop_data.longitude.max() > 180:
        pop_data = pop_data.assign_coords(
            longitude=(((pop_data.longitude + 180) % 360) - 180)
        )
        pop_data = pop_data.sortby("longitude")

    # Ensure CRS is set
    pop_data = pop_data.rio.write_crs("EPSG:4326")

    # 2. Create a Land Mask from ERA5 Template
    # ERA5-Land variables are usually NaN over the ocean.
    # We create a boolean mask: 1.0 = Land, 0.0 = Ocean
    land_mask = era_template.notnull().astype(float)

    # 3. Naive Upsample (Bilinear)
    # This creates the "fine" grid but might spill people into the ocean
    pop_fine_naive = pop_data.rio.reproject_match(
        era_template, resampling=Resampling.bilinear
    )

    # Rename dimensions if rioxarray changed them to x/y
    if "x" in pop_fine_naive.dims:
        pop_fine_naive = pop_fine_naive.rename({"y": "latitude", "x": "longitude"})

    # Ensure coords match exactly for proper alignment
    pop_fine_naive = pop_fine_naive.assign_coords(
        latitude=era_template.latitude, longitude=era_template.longitude
    )

    # 4. Apply Land Mask
    # Zeros out population in ocean pixels
    pop_fine_masked = pop_fine_naive * land_mask

    # 5. CONSERVATION OF MASS (Critical)
    # We must ensure the Total Population didn't change due to masking.
    old_total = pop_data.sum()
    new_total = pop_fine_masked.sum()

    # Avoid division by zero
    if new_total == 0:
        return pop_fine_masked

    # Calculate correction factor to restore the total count
    correction_factor = old_total / new_total
    pop_final = pop_fine_masked * correction_factor

    return pop_final


def load_population_data(age_group, years, suffix="era5_regridded.nc"):
    """Loads processed annual population files for the modern period."""
    datasets = []

    print(f"Loading {age_group} data for {len(years)} years...")

    for year in years:
        file_path = DirsLocal.pop_e5l_grid / f"t_{age_group}_{year}_{suffix}"

        if file_path.exists():
            ds = xr.open_dataset(file_path)
            # Standardize time dimension to 'year'
            if "time" in ds.dims:
                ds = ds.rename({"time": "year"})

            # Ensure it has a 'year' coordinate if missing
            if "year" not in ds.coords:
                ds = ds.expand_dims(year=[year])

            datasets.append(ds)
        else:
            print(f"Warning: Missing file {file_path}")

    if not datasets:
        raise FileNotFoundError(f"No files found for age group: {age_group}")

    # Concatenate over time
    return xr.concat(datasets, dim="year", data_vars="minimal", coords="minimal")


def plot_pre_post_upsampling(pre_data, post_data, year, region_name="Med_Coast"):
    """
    Plots pre and post upsampling.
    Uses strict sorting to ensure the map is not flipped.
    """
    # Define region (Mediterranean)
    # We use (min, max) and let xarray handle the slice direction automatically
    lat_min, lat_max = 35, 45
    lon_min, lon_max = 10, 20

    # Ensure data is sorted ascending before slicing for consistent behavior
    pre_data = pre_data.sortby("latitude").sortby("longitude")
    post_data = post_data.sortby("latitude").sortby("longitude")

    try:
        pre_region = pre_data.sel(
            year=year,
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )
        post_region = post_data.sel(
            year=year,
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )
    except KeyError:
        print(f"Year {year} not found for plotting.")
        return

    if pre_region.sum() == 0:
        print("Selected region has no data. Skipping plot.")
        return

    # Setup Plot
    fig, axs = plt.subplots(
        1, 2, figsize=(14, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Common arguments
    plot_kwargs = {
        "transform": ccrs.PlateCarree(),
        "cmap": "viridis",
        "add_colorbar": True,
        "vmin": 0,
        # Robust=True automatically cuts off extreme outliers (like the 99th quantile)
        "robust": True,
    }

    # 1. Plot Coarse
    pre_region.plot(
        ax=axs[0], **plot_kwargs, cbar_kwargs={"label": "Population", "shrink": 0.6}
    )
    axs[0].set_title(f"Pre-Upsampling (Coarse) - {year}")
    axs[0].add_feature(cfeature.COASTLINE, linewidth=1, color="black")
    axs[0].add_feature(cfeature.BORDERS, linestyle=":")

    # 2. Plot Fine
    post_region.plot(
        ax=axs[1], **plot_kwargs, cbar_kwargs={"label": "Population", "shrink": 0.6}
    )
    axs[1].set_title(f"Post-Upsampling (Fine) - {year}")
    axs[1].add_feature(cfeature.COASTLINE, linewidth=1, color="black")
    axs[1].add_feature(cfeature.BORDERS, linestyle=":")

    plt.suptitle(f"Upsampling Verification: {region_name}")
    # plt.savefig(DirsLocal.dir_figures / f"upsampling_check_{region_name}_{year}_xarray.png", bbox_inches="tight")
    plt.show()
    # plt.close()


def plot_worldpop_region(data, year, region_name="Med_Coast"):
    """
    Plots WorldPop data for a specific region and year.
    """
    # Define region (Mediterranean)
    lat_min, lat_max = 35, 45
    lon_min, lon_max = 10, 20

    # Ensure data is sorted
    data = data.sortby("latitude").sortby("longitude")

    try:
        region_data = data.sel(
            year=year,
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )
    except KeyError:
        print(f"Year {year} not found for plotting.")
        return

    if region_data.pop.sum() == 0:
        print("Selected region has no data. Skipping plot.")
        return

    # Plot
    fig, ax = plt.subplots(
        figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    region_data.pop.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        vmin=0,
        robust=True,
        cbar_kwargs={"label": "Population", "shrink": 0.6},
    )
    ax.set_title(f"WorldPop Infants - {region_name} ({year})")
    ax.add_feature(cfeature.COASTLINE, linewidth=1, color="black")
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    # plt.savefig(f"worldpop_{region_name}_{year}.png", dpi=150, bbox_inches="tight")
    plt.show()
    # plt.close()


def main():
    print("--- Starting Population Combination ---")

    # 1. Define Years
    # Ideally get this from config, but falling back to provided range
    modern_years = range(2000, 2026)
    historical_years = range(1980, 2000)  # Adjust based on available data

    # 2. Load Modern Data (Already Regridded to ERA5-Land)
    # ---------------------------------------------------------
    print("\n[1/4] Loading Modern WorldPop Data...")
    infants_worldpop = load_population_data(age_group="under_1", years=modern_years)
    elderly_worldpop = load_population_data(age_group="65_over", years=modern_years)

    # Ensure dimensions order
    infants_worldpop = infants_worldpop.transpose("year", "latitude", "longitude")
    elderly_worldpop = elderly_worldpop.transpose("year", "latitude", "longitude")

    # Plot WorldPop for verification
    plot_worldpop_region(data=infants_worldpop, year=2020)

    # 3. Load & Process Historical Data (Pre-2000)
    # ---------------------------------------------------------
    print("\n[2/4] Processing Historical Data (Pre-2000)...")

    # Prepare ERA5 Template for upsampling
    era_files = list(DirsLocal.e5l_d.glob("*.nc"))
    if not era_files:
        raise FileNotFoundError("No ERA5 files found to use as grid template.")

    with xr.open_dataset(era_files[0]) as era_ds:
        # Use a 2D slice (t_max, first time step) as the spatial template
        era_template = era_ds["t_max"].isel(time=0).drop_vars("time", errors="ignore")
        era_template = era_template.rio.write_crs("EPSG:4326")

    # Load the coarse pre-2000 dataset
    demographics_totals = xr.open_dataarray(FilesLocal.pop_before_2000)

    # --- INFANTS PRE-2000 ---
    # Note: Assuming age_band=0 represents the 0-4 age group in the historical dataset.
    # We divide by 5 later to approximate the <1 year group.
    infants_lancet = demographics_totals.sel(
        year=slice(historical_years.start, historical_years.stop - 1),
        age_band_lower_bound=0,
    )

    print(
        f"  > Pre-2000 Infants (Coarse) Total: {infants_lancet.sum().values / 1e6:.2f} M"
    )

    # Transpose and Upsample
    infants_lancet = infants_lancet.transpose("year", "latitude", "longitude")
    infants_lancet_fine = upsample_population_with_mask(infants_lancet, era_template)

    # Verify conservation
    print(
        f"  > Pre-2000 Infants (Fine) Total:   {infants_lancet_fine.sum().values / 1e6:.2f} M"
    )

    # Plot verification for one year
    plot_pre_post_upsampling(
        pre_data=infants_lancet, post_data=infants_lancet_fine, year=1990
    )

    # Apply heuristic: Convert 0-4 age band to <1 year (divide by 5)
    infants_lancet_fine = infants_lancet_fine / 5.0

    # --- ELDERLY PRE-2000 ---
    elderly_lancet = demographics_totals.sel(
        year=slice(historical_years.start, historical_years.stop - 1),
        age_band_lower_bound=65,
    )
    elderly_lancet = elderly_lancet.transpose("year", "latitude", "longitude")
    elderly_lancet_fine = upsample_population_with_mask(elderly_lancet, era_template)

    # # 4. Concatenate Historical + Modern
    # # ---------------------------------------------------------
    # print("\n[3/4] Combining Datasets...")
    #
    # # Convert DataArrays to Datasets for merging if necessary, or ensure variable names match
    # # Usually we want a Dataset with variable 'pop'
    #
    # def prep_dataset(da, name="pop"):
    #     ds = da.to_dataset(name=name) if isinstance(da, xr.DataArray) else da
    #     if "demographic_totals" in ds:
    #         ds = ds.rename({"demographic_totals": name})
    #     return ds
    #
    # ds_inf_hist = prep_dataset(infants_lancet_fine)
    # ds_inf_mod = prep_dataset(infants_worldpop)
    #
    # ds_eld_hist = prep_dataset(elderly_lancet_fine)
    # ds_eld_mod = prep_dataset(elderly_worldpop)
    #
    # # Concatenate
    # infants_final = xr.concat([ds_inf_hist, ds_inf_mod], dim="year")
    # elderly_final = xr.concat([ds_eld_hist, ds_eld_mod], dim="year")

    # # 5. Save Results
    # # ---------------------------------------------------------
    # print("\n[4/4] Saving to NetCDF...")
    #
    # def save_clean(ds, path):
    #     if path.exists():
    #         os.remove(path)
    #     # Ensure encoding for compression
    #     comp = {"zlib": True, "complevel": 5}
    #     encoding = {var: comp for var in ds.data_vars}
    #     ds.to_netcdf(path, encoding=encoding)
    #     print(f"  Saved: {path}")
    #
    # save_clean(infants_final, DirsLocal.dir_pop_infants_file)
    # save_clean(elderly_final, DirsLocal.dir_pop_elderly_file)
    #
    # print("\n✅ Processing Complete.")


if __name__ == "__main__":
    main()
