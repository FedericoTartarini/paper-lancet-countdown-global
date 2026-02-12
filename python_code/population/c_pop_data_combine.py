"""
Combine Population Data Module - FIXED & ROBUST

This module combines:
1. Historical Population (Pre-2000): Coarse resolution (0.25°), upsampled to ERA5-Land (0.1°).
2. Modern Population (2000-2025): High resolution WorldPop, already regridded to ERA5-Land.

FIXES:
- Enforces strict -180..180 longitude convention on ALL inputs.
- Validates grid alignment before merging to prevent mixed coordinate systems.
"""

from pathlib import Path
import warnings
import xarray as xr
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from scipy.spatial import cKDTree

# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path.cwd()

from my_config import Vars, VarsWorldPop, DirsLocal, FilesLocal

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def standardize_longitude(ds):
    """
    Forces Longitude to be in the range -180 to 180.
    """
    # Check for common longitude names
    lon_name = "longitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        ds = ds.rename({"lon": "longitude"})
    elif "x" in ds.dims and "longitude" not in ds.dims:
        ds = ds.rename({"x": "longitude"})

    # If no longitude, return (likely not spatial)
    if "longitude" not in ds.dims:
        return ds

    # Convert 0..360 to -180..180
    # Logic: (lon + 180) % 360 - 180
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

    # SORTING IS CRITICAL after wrapping
    ds = ds.sortby("longitude").sortby("latitude")

    return ds


def validate_alignment(ds1, ds2, name1="Dataset 1", name2="Dataset 2"):
    """
    Checks if two datasets are on the exact same spatial grid.
    Raises an error if they do not match.
    """
    print(f"Validating alignment between {name1} and {name2}...")

    # 1. Check Longitude Extents
    min1, max1 = ds1.longitude.min().item(), ds1.longitude.max().item()
    min2, max2 = ds2.longitude.min().item(), ds2.longitude.max().item()

    if not np.isclose(min1, min2, atol=1e-3) or not np.isclose(max1, max2, atol=1e-3):
        raise ValueError(
            f"❌ Longitude mismatch!\n  {name1}: {min1} to {max1}\n  {name2}: {min2} to {max2}"
        )

    # 2. Check Size
    if ds1.longitude.size != ds2.longitude.size:
        raise ValueError(
            f"❌ Longitude size mismatch! {name1}: {ds1.longitude.size}, {name2}: {ds2.longitude.size}"
        )

    print("✅ Grid alignment confirmed.")


def relocate_ocean_population(pop_data, era_template):
    """
    Detects population counts located in pixels where ERA5 is NaN (Ocean).
    Moves these counts to the NEAREST valid land pixel.
    """
    if "time" in era_template.dims:
        land_mask = era_template.isel(time=0).notnull().values
    else:
        land_mask = era_template.notnull().values

    lats = pop_data.latitude.values
    lons = pop_data.longitude.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    land_indices = np.where(land_mask.ravel())[0]

    if len(land_indices) == 0:
        print("Warning: ERA5 Template has no valid land pixels!")
        return pop_data

    pop_vals = pop_data.values.copy()
    is_3d = pop_vals.ndim == 3
    if not is_3d:
        pop_vals = pop_vals[np.newaxis, :, :]

    pop_sum_flat = pop_vals.sum(axis=0).ravel()
    stranded_mask_flat = (pop_sum_flat > 0) & (~land_mask.ravel())
    stranded_indices = np.where(stranded_mask_flat)[0]

    if len(stranded_indices) > 0:
        land_coords = np.column_stack(
            (lat_grid.ravel()[land_indices], lon_grid.ravel()[land_indices])
        )
        tree = cKDTree(land_coords)
        stranded_coords = np.column_stack(
            (lat_grid.ravel()[stranded_indices], lon_grid.ravel()[stranded_indices])
        )

        _, nearest_land_pos = tree.query(stranded_coords, k=1)
        target_indices_flat = land_indices[nearest_land_pos]

        total_moved = 0
        for t in range(pop_vals.shape[0]):
            year_data_flat = pop_vals[t].reshape(-1)
            values_to_move = year_data_flat[stranded_indices]
            np.add.at(year_data_flat, target_indices_flat, values_to_move)
            year_data_flat[stranded_indices] = 0
            pop_vals[t] = year_data_flat.reshape(pop_vals[t].shape)
            total_moved += values_to_move.sum()

        print(
            f"    - Relocated {total_moved / 1e6:.3f} M people from ocean to nearest coast."
        )

    if not is_3d:
        pop_vals = pop_vals[0]

    return pop_data.copy(data=pop_vals)


def upsample_population_with_mask(pop_data, era_template):
    """
    Upsamples Coarse Population to Fine ERA5 Grid.
    """
    # 0. STANDARDISATION
    # Ensure inputs are -180..180 and sorted
    pop_data = standardize_longitude(pop_data)
    era_template = standardize_longitude(era_template)

    # 1. Dimension Ordering (Time, Y, X)
    try:
        pop_data = pop_data.transpose("year", "latitude", "longitude")
    except ValueError:
        pop_data = pop_data.transpose(..., "latitude", "longitude")

    # 2. Strict Sorting
    pop_data = pop_data.sortby("latitude").sortby("longitude")
    era_template = era_template.sortby("latitude").sortby("longitude")

    pop_data = pop_data.rio.write_crs("EPSG:4326")

    # Compute original total for assertion
    original_total = pop_data.sum().values

    # 3. Upsample (Sum for conservation)
    pop_fine = pop_data.rio.reproject_match(era_template, resampling=Resampling.sum)

    if "x" in pop_fine.dims:
        pop_fine = pop_fine.rename({"y": "latitude", "x": "longitude"})

    # Assert total is conserved
    upsampled_total = pop_fine.sum().values
    assert np.isclose(original_total, upsampled_total, rtol=3), (
        f"Population total changed: {original_total} -> {upsampled_total}"
    )

    # 4. Coastal Fix
    pop_final = relocate_ocean_population(pop_fine, era_template)

    return pop_final


def load_population_data(age_group, years, suffix="era5_regridded.nc"):
    """Loads processed annual population files for the modern period."""
    datasets = []
    print(f"Loading {age_group} data for {len(years)} years...")

    for year in years:
        file_path = DirsLocal.pop_e5l_grid / f"t_{age_group}_{year}_{suffix}"
        if file_path.exists():
            ds = xr.open_dataset(file_path)

            # Standardization happens IMMEDIATELY upon loading
            ds = standardize_longitude(ds)

            if "time" in ds.dims:
                ds = ds.rename({"time": "year"})
            if "year" not in ds.coords:
                ds = ds.expand_dims(year=[year])
            datasets.append(ds)
        else:
            print(f"Warning: Missing file {file_path}")

    if not datasets:
        raise FileNotFoundError(f"No files found for {age_group}")

    ds = xr.concat(datasets, dim="year", data_vars="minimal", coords="minimal")

    # Sort after concat
    if "latitude" in ds.dims:
        ds = ds.sortby("latitude").sortby("longitude")

    return ds


def plot_pre_post_upsampling(pre_data, post_data, year, region_name="Med_Coast"):
    """Plots pre and post upsampling using xarray for safety."""
    pre_data = standardize_longitude(pre_data)
    post_data = standardize_longitude(post_data)

    lat_slice = slice(35, 45)  # Mediterranean
    lon_slice = slice(10, 20)

    pre_region = pre_data.sel(year=year, latitude=lat_slice, longitude=lon_slice)
    post_region = post_data.sel(year=year, latitude=lat_slice, longitude=lon_slice)

    if pre_region.sum() == 0 or post_region.sum() == 0:
        print(
            f"Warning: No population in pre-upsampling data for {region_name} in {year}. Skipping plot."
        )
        return

    fig, axs = plt.subplots(
        1, 2, figsize=(14, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    kwargs = {
        "transform": ccrs.PlateCarree(),
        "cmap": "viridis",
        "robust": True,
        "add_colorbar": True,
    }

    pre_region.plot(ax=axs[0], **kwargs)
    axs[0].set_title(f"Coarse (Pre) - {year}")
    axs[0].coastlines()

    post_region.plot(ax=axs[1], **kwargs)
    axs[1].set_title(f"Fine (Post) - {year}")
    axs[1].coastlines()

    plt.suptitle(f"Upsampling Check: {region_name}")
    plt.show()


def plot_final_data(dat):
    """Plots pre and post upsampling using xarray for safety."""
    fig, axs = plt.subplots(
        1, 1, figsize=(14, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    kwargs = {
        "transform": ccrs.PlateCarree(),
        "cmap": "viridis",
        "robust": True,
        "add_colorbar": True,
    }

    dat.plot(ax=axs, **kwargs)
    axs.coastlines()
    plt.tight_layout()

    plt.show()


def clean_and_align(ds_to_fix, ds_reference):
    """
    1. Drops extra coordinates (band, spatial_ref) that block concatenation.
    2. Overwrites lat/lon with the reference dataset to fix floating point mismatches.
    """
    # A. Convert to Dataset if needed
    if isinstance(ds_to_fix, xr.DataArray):
        ds_to_fix = ds_to_fix.to_dataset(name="pop")

    # B. Drop non-dimensional coordinates that might conflict
    # (e.g., 'band', 'spatial_ref', 'age_band_lower_bound')
    drop_vars = [
        v for v in ds_to_fix.coords if v not in ["year", "latitude", "longitude"]
    ]
    ds_to_fix = ds_to_fix.drop_vars(drop_vars)

    # C. FORCE ALIGNMENT (The Magic Step)
    # We assign the latitude/longitude arrays from the reference directly.
    # This guarantees they are bitwise identical.
    ds_to_fix = ds_to_fix.assign_coords(
        {"latitude": ds_reference.latitude, "longitude": ds_reference.longitude}
    )

    return ds_to_fix


def main():
    print("--- Starting Population Combination ---")

    modern_years = range(2000, 2026)  # todo this should not be hardcoded
    historical_years = range(1980, 2000)

    # 1. Load ERA5 Template & Standardize
    era_files = list(DirsLocal.e5l_d.glob("*.nc"))
    if not era_files:
        raise FileNotFoundError("No ERA5 files found.")

    with xr.open_dataset(era_files[0]) as era_ds:
        era_template = era_ds["t_max"].isel(time=0).drop_vars("time", errors="ignore")
        era_template = era_template.rio.write_crs("EPSG:4326")
        # STRICT STANDARDIZATION
        era_template = standardize_longitude(era_template)
        era_template = era_template.sortby("latitude").sortby("longitude")

    # 2. Modern Data
    infants_worldpop = load_population_data("under_1", modern_years)
    elderly_worldpop = load_population_data("65_over", modern_years)

    # calc average totals across all modern years for sanity check
    avg_infants = infants_worldpop.pop.mean(dim="year").sum().item()
    avg_elderly = elderly_worldpop.pop.mean(dim="year").sum().item()
    print(f"Average Infants (2000-2025): {avg_infants / 1e6:.2f} M")
    print(f"Average Elderly (2000-2025): {avg_elderly / 1e6:.2f} M")

    # Verify Modern Data matches ERA5 range
    validate_alignment(
        infants_worldpop, era_template, "WorldPop Infants", "ERA5 Template"
    )

    # 3. Historical Data (Pre-2000)
    print("\n[2/4] Processing Historical Data...")
    demographics_totals = xr.open_dataarray(FilesLocal.pop_before_2000)

    print("Plotting raw historical data slice (1990) to check for obvious issues...")
    mediterranean_slice = demographics_totals.sel(
        year=1990,
        age_band_lower_bound=0,
        latitude=slice(45, 35),
        longitude=slice(10, 20),
    )
    plot_final_data(dat=mediterranean_slice)

    # Standardize Historical Input BEFORE anything else
    demographics_totals = standardize_longitude(demographics_totals)

    # --- Infants ---
    infants_lancet = (
        demographics_totals.sel(
            year=slice(historical_years.start, historical_years.stop - 1),
            age_band_lower_bound=0,
        )
        / 5.0
    )

    # Upsample (will use standardized era_template)
    infants_lancet_fine = upsample_population_with_mask(infants_lancet, era_template)

    # 0-4y to <1y Conversion
    infants_lancet_fine = infants_lancet_fine

    # Verify Upsampled Data matches WorldPop range
    validate_alignment(
        infants_lancet_fine, infants_worldpop, "Upsampled Infants", "WorldPop Infants"
    )
    plot_pre_post_upsampling(
        pre_data=infants_lancet, post_data=infants_lancet_fine, year=1990
    )

    # --- Elderly ---
    elderly_lancet = demographics_totals.sel(
        year=slice(historical_years.start, historical_years.stop - 1),
        age_band_lower_bound=65,
    )
    elderly_lancet_fine = upsample_population_with_mask(elderly_lancet, era_template)
    validate_alignment(
        elderly_lancet_fine, elderly_worldpop, "Upsampled Elderly", "WorldPop Elderly"
    )

    # 4. Combine and Save
    print("\n[3/4] Combining and Saving...")

    # Prepare Reference (Modern Data is usually the 'truth' since it came from Script B)
    # Ensure modern data is clean too
    ds_inf_mod = clean_and_align(infants_worldpop, infants_worldpop)
    ds_eld_mod = clean_and_align(elderly_worldpop, elderly_worldpop)

    # Align Historical Data to Modern Grid
    ds_inf_hist = clean_and_align(infants_lancet_fine, infants_worldpop)
    ds_eld_hist = clean_and_align(elderly_lancet_fine, elderly_worldpop)

    # Combine
    infants_final = xr.concat([ds_inf_hist, ds_inf_mod], dim="year")
    elderly_final = xr.concat([ds_eld_hist, ds_eld_mod], dim="year")

    infants_final = infants_final.where(infants_final.pop > 0)
    elderly_final = elderly_final.where(elderly_final.pop > 0)

    plot_final_data(dat=infants_final.sel(year=1990).pop)
    plot_final_data(dat=infants_final.sel(year=2010).pop)
    plot_final_data(dat=elderly_final.sel(year=1990).pop)
    plot_final_data(dat=elderly_final.sel(year=2010).pop)

    def plot_worldpop_lancet(dat):
        fig, axs = plt.subplots(
            2, 1, figsize=(7, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        dat = dat.sortby("latitude").sortby("longitude")
        dat = dat.sel(latitude=slice(35, 45), longitude=slice(0, 20))

        kwargs = {
            "transform": ccrs.PlateCarree(),
            "cmap": "viridis",
            "robust": True,
            "add_colorbar": True,
        }

        dat.sel(year=2000).pop.plot(ax=axs[0], **kwargs)
        axs[0].set_title("WorldPop - 2000")
        axs[0].coastlines()

        dat.sel(year=1999).pop.plot(ax=axs[1], **kwargs)
        axs[1].set_title("Lancet Upsampled - 1999")
        axs[1].coastlines()

        plt.suptitle("Comparison of WorldPop and Lancet Upsampled (1990)")
        plt.show()

    plot_worldpop_lancet(dat=infants_final)

    # FINAL CHECK
    print(
        f"Final Infants Longitude Range: {infants_final.longitude.min().item():.2f} to {infants_final.longitude.max().item():.2f}"
    )

    if infants_final.longitude.max() > 181:
        raise ValueError("CRITICAL ERROR: Final dataset still has >180 longitudes!")

    DirsLocal.pop_e5l_grid_combined.mkdir(parents=True, exist_ok=True)

    comp = {"zlib": True, "complevel": 5}
    infants_final.to_netcdf(FilesLocal.pop_inf, encoding={"pop": comp})
    elderly_final.to_netcdf(FilesLocal.pop_over_65, encoding={"pop": comp})

    print("\n✅ Processing Complete.")


if __name__ == "__main__":
    main()
