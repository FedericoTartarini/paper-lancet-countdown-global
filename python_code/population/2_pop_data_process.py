import os
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import xarray as xr
from tqdm import tqdm

# Assuming these are your local config files
from my_config import VarsWorldPop, Dirs, Vars


def plot_comparison(original_da, regridded_da, title_prefix=""):
    """
    Helper function to visualize the population data before and after regridding.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- FIX: Sort coordinates for plotting ---
    # ERA5 latitudes are often descending, which crashes xarray plotting.
    original_for_plot = original_da.sortby(["latitude", "longitude"])
    regridded_for_plot = regridded_da.sortby(["latitude", "longitude"])

    # Handle 0 or negative values for Log plot
    valid_vals = original_for_plot.values[original_for_plot.values > 0]
    vmin = np.min(valid_vals) if valid_vals.size > 0 else 1

    # Plot 1: Original
    original_for_plot.plot(
        ax=axes[0], cmap="viridis", robust=True, cbar_kwargs={"label": "Pop Count"}
    )
    axes[0].set_title(f"{title_prefix} Intermediate (5km)")

    # Plot 2: Regridded
    regridded_for_plot.plot(
        ax=axes[1], cmap="viridis", robust=True, cbar_kwargs={"label": "Pop Count"}
    )
    axes[1].set_title(f"{title_prefix} Final ERA5 Grid (31km)")

    plt.tight_layout()
    plt.show()


def get_era5_target_grid(year):
    """
    Extracts the grid coordinates from an ERA5 file.
    """
    era5_path = Dirs.dir_era_daily / f"{year}_temperature_summary.nc"

    # decode_timedelta=False suppresses the warning
    era5_data = xr.open_dataset(era5_path, decode_timedelta=False)

    # Standardize longitude to -180 to 180
    era5_data = era5_data.assign_coords(
        longitude=(((era5_data.longitude + 180) % 360) - 180)
    )

    # Return just the coordinates
    # We strip 'time' or other vars to keep it lightweight
    return era5_data[["latitude", "longitude"]]


def sum_files(files, directory=""):
    """
    Sums data across a list of files safely.
    """
    total_sum = None
    for file in files:
        file_path = os.path.join(directory, file)

        # masked=True handles NoData (-3.4e38) automatically
        data_array = rioxarray.open_rasterio(file_path, masked=True).squeeze()

        # Replace NaNs with 0 and ensure no negatives
        data_array = data_array.fillna(0).where(data_array >= 0, 0)

        if total_sum is None:
            total_sum = data_array
        else:
            total_sum = total_sum + data_array

    return total_sum


def clean_and_coarsen(data_array):
    """
    Cleans data and coarsens it by a factor of 5.
    """
    # 1. Clean data
    data_array = data_array.where((data_array < 3.1e8) & (np.isfinite(data_array)), 0)

    # 2. Standardize names
    if "y" in data_array.dims:
        data_array = data_array.rename({"y": "latitude", "x": "longitude"})

    # 3. Coarsen
    data_array = data_array.coarsen(latitude=5, boundary="trim").sum()
    data_array = data_array.coarsen(longitude=5, boundary="trim").sum()

    return data_array


def regrid_to_era5_fast(pop_da, target_ds):
    """
    Aggregates population into ERA5 grid using numpy histogram2d.
    No Pandas, No Broadcasting -> Very Fast & Low Memory.
    """
    # 1. Prepare Source Data (WorldPop)
    src_lons, src_lats = np.meshgrid(pop_da.longitude.values, pop_da.latitude.values)
    weights = pop_da.values

    # Flatten arrays
    src_lons_flat = src_lons.ravel()
    src_lats_flat = src_lats.ravel()
    weights_flat = weights.ravel()

    # Optimization: Only process pixels with people
    mask = weights_flat > 0
    src_lons_flat = src_lons_flat[mask]
    src_lats_flat = src_lats_flat[mask]
    weights_flat = weights_flat[mask]

    if len(weights_flat) == 0:
        # Return empty grid if no population
        return xr.zeros_like(target_ds["latitude"] * target_ds["longitude"])

    # 2. Prepare Target Bins (ERA5)
    # Ensure bins are strictly increasing for np.histogram
    tgt_lats = np.sort(target_ds.latitude.values)
    tgt_lons = np.sort(target_ds.longitude.values)

    # Calculate resolution
    lat_res = np.abs(tgt_lats[1] - tgt_lats[0])
    lon_res = np.abs(tgt_lons[1] - tgt_lons[0])

    # Create edges: [center - half_res, ... , center + half_res]
    lat_edges = np.append(tgt_lats - lat_res / 2, tgt_lats[-1] + lat_res / 2)
    lon_edges = np.append(tgt_lons - lon_res / 2, tgt_lons[-1] + lon_res / 2)

    # 3. Aggregate (The Magic Step)
    H, _, _ = np.histogram2d(
        src_lats_flat, src_lons_flat, bins=[lat_edges, lon_edges], weights=weights_flat
    )

    # 4. Pack back into xarray
    # H shape corresponds to the SORTED latitudes/longitudes
    out_da = xr.DataArray(
        H,
        coords={"latitude": tgt_lats, "longitude": tgt_lons},
        dims=("latitude", "longitude"),
        name="pop",
    )

    # 5. Align back to original Target Grid
    # (Because ERA5 is usually descending, but our bins were ascending)
    out_da = out_da.reindex_like(target_ds, method=None, fill_value=0)

    return out_da


def process_and_combine_ages(ages, sex, year, directory, target_grid):
    """
    Loads raw files, sums ages, and aggregates to the ERA5 target grid.
    """
    # 1. Gather files
    combined_files = []
    for age in ages:
        age_files = [
            f
            for f in os.listdir(directory)
            if f"_{sex}_{age}_{year}" in f and f.endswith(".tif")
        ]
        combined_files.extend(age_files)

    if not combined_files:
        print(f"No files found for Sex: {sex}, Year: {year}, Ages: {ages}")
        return None

    # 2. Sum raw data -> Clean -> Coarsen
    summed_data = sum_files(combined_files, directory)
    pop_intermediate = clean_and_coarsen(summed_data)

    # --- SANITY CHECK START ---
    total_input_pop = float(pop_intermediate.sum(dtype="float64").item())
    # --------------------------

    # 3. Align to ERA5 (Using Fast Histogram)
    pop_regrided_da = regrid_to_era5_fast(pop_intermediate, target_grid)

    # --- SANITY CHECK END ---
    total_output_pop = float(pop_regrided_da.sum(dtype="float64").item())
    diff = total_output_pop - total_input_pop
    percent_diff = (diff / total_input_pop * 100) if total_input_pop != 0 else 0

    print(f"\n[Sanity Check] {sex}-{year} Ages:{ages}")
    print(f"  Input Pop:  {total_input_pop:,.0f}")
    print(f"  Output Pop: {total_output_pop:,.0f}")
    print(f"  Diff:       {diff:+,.0f} ({percent_diff:.5f}%)")

    # --- VISUAL CHECK ---
    # Only plot if year is 2000 (or whichever condition you prefer)
    if year == 2021 and sex == "t" and ages == [0]:
        plot_comparison(pop_intermediate, pop_regrided_da, title_prefix=f"{sex}-{year}")

    return pop_regrided_da


def process_and_save_population_data(ages, year, sex, target_grid):
    out_path = (
        Dirs.dir_pop_era_grid
        / f"{sex}_{'_'.join(map(str, ages))}_{year}_era5_compatible.nc"
    )

    if out_path.exists():
        # print(f"Skipping {out_path.name} (Exists)")
        return

    pop_regridded = process_and_combine_ages(
        ages=ages,
        sex=sex,
        year=year,
        directory=Dirs.dir_pop_raw,
        target_grid=target_grid,
    )

    if pop_regridded is None:
        return

    # Final formatting
    ds = pop_regridded.to_dataset(name="pop")
    ds = ds.expand_dims(time=[year])
    ds = ds.sortby(["latitude", "longitude"])

    # Save
    encoding = {"pop": {"zlib": True, "complevel": 5}}
    ds.to_netcdf(out_path, encoding=encoding)


def main(
    ages_array=[[0], [65, 70, 75, 80, 85, 90], [75, 80, 85, 90]],
    years_array=VarsWorldPop.get_years_range(),
    sex_array=["t"],
):
    print("Loading ERA5 Target Grid...")
    target_grid = get_era5_target_grid(year=Vars.year_min_analysis)

    total_iterations = len(ages_array) * len(sex_array) * len(years_array)

    with tqdm(total=total_iterations) as pbar:
        for age in ages_array:
            for year in years_array:
                for sex in sex_array:
                    process_and_save_population_data(
                        ages=age, year=year, sex=sex, target_grid=target_grid
                    )
                    pbar.update(1)


if __name__ == "__main__":
    main()
