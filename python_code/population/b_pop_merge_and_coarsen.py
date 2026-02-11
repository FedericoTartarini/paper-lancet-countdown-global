import os
import sys
import numpy as np
import rioxarray
import xarray as xr
from tqdm import tqdm
import argparse
import concurrent.futures
from rasterio.enums import Resampling
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path for imports
sys.path.append(os.getcwd())

# Assuming these are your local config files
from my_config import VarsWorldPop, DirsLocal, ensure_directories

"""
This script merges and regrids WorldPop population data to ERA5-Land resolution for the Lancet Countdown project.

It processes raw WorldPop TIFF files from the SSD directory, summing population across specified age groups.
For years 2000-2014, it combines male and female data into total population.
The data is cleaned and directly regridded to match the ERA5-Land grid (~9 km) using conservative summing,
ensuring proper alignment and avoiding intermediate coarsening steps. Results are saved as compressed NetCDF files
in the intermediate directory.

Usage:
    - Trial run (test with 2000 data only): python python_code/population/b_pop_merge_and_coarsen.py --trial
    - Full run (all years): python python_code/population/b_pop_merge_and_coarsen.py

Dependencies: rioxarray, xarray, numpy, tqdm, argparse, rasterio
"""


def _output_age_label(age_group):
    """Return the desired output age label for a given age_group list.

    Mapping rules:
    - [0] -> 'under_1'
    - startswith 65 and length > 1 -> '65_over'
    - startswith 75 and length > 1 -> '75_over'
    - otherwise fallback to the numeric joined string (e.g., '30_34')
    """
    # If it's a single-element list with 0
    try:
        if len(age_group) == 1 and age_group[0] == 0:
            return "under_1"

        if len(age_group) >= 1 and age_group[0] == 65:
            return "65_over"

    except Exception:
        # Fallback: if age_group isn't iterable as expected, fall back to string
        return str(age_group)

    # default
    return "_".join(map(str, age_group))


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


def clean_and_regrid_to_era5(data_array, era_template):
    """
    Cleans data and regrids it to match the ERA5-Land resolution using conservative summation.
    """
    # 1. Clean data
    data_array = data_array.where((data_array < 3.1e8) & (np.isfinite(data_array)), 0)

    # 2. Standardize names
    if "y" in data_array.dims:
        data_array = data_array.rename({"y": "latitude", "x": "longitude"})

    # 3. Shift ERA5 longitude to -180 to 180 to match WorldPop
    era_shifted = era_template.assign_coords(
        longitude=(((era_template.longitude + 180) % 360) - 180)
    )
    era_shifted = era_shifted.sortby("longitude")

    # Set CRS for reproject_match
    era_shifted = era_shifted.rio.write_crs("EPSG:4326")

    # 4. Regrid WorldPop to match ERA5 grid using sum resampling
    pop_regridded = data_array.rio.reproject_match(
        era_shifted, resampling=Resampling.sum
    )

    # 5. Rename dimensions if needed (reproject_match may use x/y)
    if "x" in pop_regridded.dims:
        pop_regridded = pop_regridded.rename({"x": "longitude", "y": "latitude"})

    # 6. Shift longitude back to 0-360 to match heatwave_days coordinates
    pop_regridded = pop_regridded.assign_coords(
        longitude=((pop_regridded.longitude + 360) % 360)
    )
    pop_regridded = pop_regridded.sortby("longitude")

    return pop_regridded


def process_and_save_regridded_population(ages, year, sex, directory, era_template):
    """
    Loads raw files, sums ages and sexes if needed, cleans, regrids, and saves intermediate NetCDF.
    """
    out_path = (
        DirsLocal.pop_e5l_grid
        / f"{sex}_{_output_age_label(ages)}_{year}_era5_regridded.nc"
    )

    if out_path.exists():
        print(f"Skipping {out_path.name} (Exists)")
        return

    # Assert required files exist
    if year <= 2014:
        expected_ages = [0, 65, 70, 75, 80]
        for age in expected_ages:
            f_file = f"global_f_{age}_{year}_1km.tif"
            m_file = f"global_m_{age}_{year}_1km.tif"
            assert os.path.exists(os.path.join(directory, f_file)), f"Missing {f_file}"
            assert os.path.exists(os.path.join(directory, m_file)), f"Missing {m_file}"
    else:
        expected_ages = [0, 65, 70, 75, 80, 85, 90]
        for age in expected_ages:
            t_file = f"global_t_{age}_{year}_1km.tif"
            assert os.path.exists(os.path.join(directory, t_file)), f"Missing {t_file}"

    # 1. Gather files
    combined_files = []
    for age in ages:
        if year <= 2014 and sex == "t":
            # Combine males and females
            for s in ["f", "m"]:
                age_files = [
                    f
                    for f in os.listdir(directory)
                    if f"_{s}_{age}_{year}" in f and f.endswith(".tif")
                ]
                combined_files.extend(age_files)
        else:
            age_files = [
                f
                for f in os.listdir(directory)
                if f"_{sex}_{age}_{year}" in f and f.endswith(".tif")
            ]
            combined_files.extend(age_files)

    if not combined_files:
        print(f"No files found for Sex: {sex}, Year: {year}, Ages: {ages}")
        return

    # 2. Sum raw data -> Clean -> Regrid
    summed_data = sum_files(combined_files, directory)
    total_raw = float(summed_data.sum(dtype="float64").item())
    pop_regridded = clean_and_regrid_to_era5(summed_data, era_template)
    total_regridded = float(pop_regridded.sum(dtype="float64").item())

    # Sanity check
    total_pop = total_regridded
    print(f"Total pop for {sex}-{year} Ages:{ages}: {total_pop:,.0f}")
    # Allow 2% tolerance for match due to floating point precision and cleaning
    tolerance = 0.02
    match = abs(total_raw - total_regridded) / max(total_raw, 1) < tolerance
    emoji = "✅" if match else "❌"
    print(f"Total raw: {total_raw:,.0f}, Regridded: {total_regridded:,.0f} {emoji}")

    # 3. Save as NetCDF
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = pop_regridded.to_dataset(name="pop")
    ds = ds.expand_dims(time=[year])
    ds = ds.sortby(["latitude", "longitude"])

    encoding = {"pop": {"zlib": True, "complevel": 5}}
    ds.to_netcdf(out_path, encoding=encoding)


def main(
    ages_array=VarsWorldPop.age_groups,
    years_array=VarsWorldPop.get_years_range(),
    sex_array=["t"],
):
    ensure_directories([DirsLocal.pop_e5l_grid])

    # Load ERA5 template
    era_sample_files = list(DirsLocal.e5l_d.glob("*.nc"))
    if not era_sample_files:
        raise FileNotFoundError("No ERA5 daily files found for template.")
    era_sample_path = era_sample_files[0]
    era_ds = xr.open_dataset(era_sample_path)
    era_template = era_ds["t_max"].isel(time=0)  # Use t_max as template

    total_iterations = len(ages_array) * len(sex_array) * len(years_array)

    with tqdm(total=total_iterations) as pbar:

        def update_progress(future):
            pbar.update(1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    process_and_save_regridded_population,
                    age,
                    year,
                    sex,
                    DirsLocal.pop_raw_ssd,
                    era_template,
                )
                for age in ages_array
                for year in years_array
                for sex in sex_array
            ]

            for future in concurrent.futures.as_completed(futures):
                future.result()  # Raise any exceptions
                update_progress(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and regrid WorldPop data.")
    parser.add_argument(
        "--trial", action="store_true", help="Run in trial mode with only 2000 data."
    )
    args = parser.parse_args()

    if args.trial:
        years_array = [2000]
    else:
        years_array = VarsWorldPop.get_years_range()

    main(years_array=years_array)
