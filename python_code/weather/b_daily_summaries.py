"""
This file contains the code which calculates daily summaries of weather data using the ERA5-Land dataset.
"""

import logging
import argparse
import sys
from pathlib import Path
import os
import warnings

# Add project root to sys.path to allow importing my_config
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path.cwd()

sys.path.append(str(project_root))

import xarray as xr
from dask.distributed import Client

from my_config import Dirs

# Create a logs directory
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "daily_summaries.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Write to file
        logging.StreamHandler(),  # Keep printing to PBS/Console
    ],
)
logger = logging.getLogger(__name__)


def process_and_save_data(ds, output_file, year_label):
    """
    Helper function to process the dataset and save the result.
    """
    # ERA5-Land 2t variable is usually named 't2m' or '2t'
    # We are identifying the variable name
    var_name = list(ds.data_vars)[0]  # Assuming single variable files

    # Resample to daily frequency
    # We want min, mean, and max for each day
    # Using explicit aggregation instead of .agg() for compatibility
    resampler = ds[var_name].resample(time="1D")

    daily_ds = xr.Dataset()
    daily_ds["t_min"] = resampler.min()
    daily_ds["t_mean"] = resampler.mean()
    daily_ds["t_max"] = resampler.max()

    # Add metadata
    daily_ds.attrs["description"] = (
        f"Daily summaries of 2-meter temperature from ERA5-Land for {year_label}"
    )
    daily_ds.attrs["source"] = "NCI project zz93 ERA5-Land"

    # Save to NetCDF
    # We use compute() here to trigger the dask computation and write to disk
    if isinstance(output_file, str):
        output_file = Path(output_file)

    # Ensure expansion if typically passed from user input like ~
    output_file = Path(os.path.expanduser(str(output_file)))

    # Define compression encoding
    encoding_params = {
        "zlib": True,
        "complevel": 1,
        "dtype": "float32",
        # "least_significant_digit": 1,
    }
    encoding = {var: encoding_params for var in daily_ds.data_vars}

    # Suppress "All-NaN slice encountered" warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        daily_ds.to_netcdf(output_file, encoding=encoding)

    logger.info(f"Saved: {output_file.name}")


def process_year_in_months(year, year_dir, output_file):
    """
    Robustly processes a year by calculating each month individually,
    saving to interim files, and then merging.
    Checks points at:
    1. If final file exists -> Skip
    2. If interim file exists -> Skip (Resume capability)
    """
    if output_file.exists():
        logger.info(f"File {output_file.name} already exists. Skipping.")
        return

    # Define intermediate directory
    interim_dir = output_file.parent / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    monthly_files = []

    # 1. Process each month
    for month in range(1, 13):
        month_str = f"{year}{month:02d}"  # e.g., 202001
        interim_file = interim_dir / f"{year}_{month:02d}_daily.nc"
        monthly_files.append(interim_file)

        if interim_file.exists():
            # Checkpoint recovery
            logger.info(f"   Checkpoint found: {interim_file.name}")
            continue

        # Find input files for this month
        # Pattern usually: 2t_era5-land_oper_sfc_YYYYMM01-YYYYMMDD.nc
        # We match *YYYYMM* to capture specific month files
        input_files = list(year_dir.glob(f"*{month_str}*.nc"))

        if not input_files:
            logger.warning(
                f"   ⚠️ No input files found for {year}-{month:02d}. Skipping month."
            )
            continue

        try:
            # Process this month specifically
            ds = xr.open_dataset(
                input_files,
                parallel=True,
                chunks={"time": -1, "latitude": 500, "longitude": 500},
            )
            process_and_save_data(ds, interim_file, f"{year}-{month:02d}")
            ds.close()
        except Exception as e:
            logger.error(f"Failed to process month {year}-{month:02d}: {e}")
            # If a month fails, we stop this year to avoid creating a broken incomplete file
            return

    # 2. Merge all months
    valid_months = [f for f in monthly_files if f.exists()]

    if len(valid_months) != 12:
        logger.warning(
            f"Found {len(valid_months)}/12 interim files for {year}. Skipping merge."
        )
        return

    logger.info(f"Merging {len(valid_months)} monthly files into {output_file.name}...")

    try:
        # Open all with dask
        ds_year = xr.open_mfdataset(valid_months, parallel=True, chunks={"time": -1})

        # Save final
        # Logic matches process_and_save_data regarding encoding via reusing logic?
        # No, simpler to just write here since data is already 'daily'
        encoding_params = {
            "zlib": True,
            "complevel": 1,
            "dtype": "float32",
            "least_significant_digit": 1,
        }
        encoding = {var: encoding_params for var in ds_year.data_vars}

        ds_year.to_netcdf(output_file, encoding=encoding)

        logger.info(f"✅ Successfully created year file: {output_file}")

        # 3. Cleanup
        logger.info("Cleaning up interim files...")
        for f in valid_months:
            f.unlink()

    except Exception as e:
        logger.error(f"Failed during merge of {year}: {e}")
        # Clean up partial output if it exists
        if output_file.exists():
            output_file.unlink()


def main():
    """
    Main function to calculate daily summaries (min, mean, max) from hourly ERA5-Land data.
    """
    parser = argparse.ArgumentParser(description="Process ERA5-Land daily summaries.")
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Run in trial mode: only process the first year (1980) to verify execution.",
    )
    parser.add_argument(
        "--local_file",
        type=str,
        help="Path to a local file to process for checks (bypasses Gadi logic).",
    )
    args = parser.parse_args()

    # Initialize Dask Client for distributed computing
    # Use n_workers based on PBS allocation if available, with 1 thread per worker
    # to avoid GIL contention during heavy numpy operations.
    n_workers = int(os.environ.get("PBS_NCPUS", 4))
    client = Client(n_workers=n_workers, threads_per_worker=1)
    logger.info(f"Dask Dashboard link: {client.dashboard_link}")

    if args.local_file:
        logger.info(f"Running in LOCAL FILE mode with: {args.local_file}")
        file_path = Path(args.local_file).expanduser()
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return

        ds = xr.open_dataset(file_path, chunks={})
        ds = ds.chunk({"time": -1, "latitude": 500, "longitude": 500})

        output_file = Path("~/Downloads/local_daily_summary_test.nc").expanduser()
        try:
            process_and_save_data(ds, output_file, "Local Test File")
        except Exception as e:
            logger.error(f"Failed to process local file: {e}")
            raise

        ds.close()
        return

    # Define the output directory for daily summaries
    output_dir = Dirs.dir_era_daily
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input directory: {Dirs.dir_era_land}")
    logger.info(f"Output directory: {output_dir}")

    # Iterate over the years of interest
    start_year = 1980
    end_year = 2025

    if args.trial:
        logger.info("=== TRIAL MODE ACTIVE ===")
        logger.info(f"Processing only start_year: {start_year}")
        end_year = start_year

    for year in range(start_year, end_year + 1):
        logger.info(f"=== Processing Year: {year} ===")

        output_file = output_dir / f"{year}_daily_summaries.nc"

        try:
            # Construct the path for the specific year in the zz93 structure
            # Structure: /g/data/zz93/era5-land/reanalysis/surface/2t/{year}/*.nc
            year_dir = Dirs.dir_era_land / str(year)

            if not year_dir.exists():
                logger.error(f"Directory not found: {year_dir}")
                continue

            # Check if this year folder has files
            if not any(year_dir.iterdir()):
                logger.error(f"Directory empty: {year_dir}")
                continue

            # Switch to Monthly Strategy
            process_year_in_months(year, year_dir, output_file)

        except Exception as e:
            logger.error(f"Failed to process year {year}: {e}")


if __name__ == "__main__":
    pass
    # ensure_directories([Dirs.dir_era_daily, Dirs.dir_results_heatwaves])
    main()


# python3 python_code/weather/b_daily_summaries.py --local_file ~/Downloads/2t_era5-land_oper_sfc_19500101-19500131.nc
