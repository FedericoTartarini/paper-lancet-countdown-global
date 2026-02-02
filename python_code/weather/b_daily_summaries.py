"""
This file contains the code which calculates daily summaries of weather data using the ERA5-Land dataset.
Structure: Monthly checkpoints (Robust) + Optimized Compression + Gadi Safety Fixes.
"""

import os

# --- GADI FIX 1: DISABLE FILE LOCKING ---
# MUST be set before importing xarray/netCDF4
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# ----------------------------------------

import logging
import argparse
import sys
from pathlib import Path
import warnings
import dask

# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path.cwd()

sys.path.append(str(project_root))

import xarray as xr
from dask.distributed import Client

from my_config import Dirs

# --- GADI FIX 2: INCREASE TIMEOUTS ---
# Prevents "Connection timed out" errors when the disk is busy
dask.config.set(
    {
        "distributed.comm.timeouts.connect": "90s",
        "distributed.comm.timeouts.tcp": "90s",
    }
)
# -------------------------------------

# Create a logs directory
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "daily_summaries.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# --- OPTIMIZATION SETTINGS ---
ENCODING_PARAMS = {
    "zlib": True,
    "complevel": 1,
    "shuffle": True,
    "dtype": "float32",
    "least_significant_digit": 1,
    "_FillValue": -9999.0,
}


def process_and_save_data(ds, output_file, year_label):
    """
    Helper function to process the dataset and save the result.
    """
    var_name = list(ds.data_vars)[0]

    # Resample to daily frequency
    # Note: If 'flox' is installed, this step is automatically optimized
    resampler = ds[var_name].resample(time="1D")

    daily_ds = xr.Dataset()
    daily_ds["t_min"] = resampler.min()
    daily_ds["t_mean"] = resampler.mean()
    daily_ds["t_max"] = resampler.max()

    daily_ds.attrs["description"] = (
        f"Daily summaries of 2-meter temperature from ERA5-Land for {year_label}"
    )
    daily_ds.attrs["source"] = "NCI project zz93 ERA5-Land"

    if isinstance(output_file, str):
        output_file = Path(output_file)
    output_file = Path(os.path.expanduser(str(output_file)))

    encoding = {var: ENCODING_PARAMS for var in daily_ds.data_vars}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")

        # --- GADI FIX 3: ROBUST WRITE ---
        # compute=True triggers the calculation.
        # We let the Dask Client handle the parallel compute, but the 'to_netcdf'
        # write itself is handled safely.
        daily_ds.to_netcdf(output_file, encoding=encoding)

    logger.info(f"   Saved interim: {output_file.name}")


def process_year_in_months(year, year_dir, output_file, target_chunks):
    """
    Robustly processes a year by calculating each month individually.
    """
    if output_file.exists():
        logger.info(f"File {output_file.name} already exists. Skipping.")
        return

    interim_dir = output_file.parent / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    monthly_files = []

    for month in range(1, 13):
        month_str = f"{year}{month:02d}"
        interim_file = interim_dir / f"{year}_{month:02d}_daily.nc"
        monthly_files.append(interim_file)

        if interim_file.exists():
            logger.info(f"   Checkpoint found: {interim_file.name}")
            continue

        input_files = sorted(list(year_dir.glob(f"*{month_str}*.nc")))

        if not input_files:
            logger.warning(
                f"   ⚠️ No input files found for {year}-{month:02d}. Skipping."
            )
            continue

        try:
            # Smart Opening
            # Use chunks={} first to load with native chunks (avoids warning),
            # then rechunk to our target size for processing.

            if len(input_files) == 1:
                ds = xr.open_dataset(
                    input_files[0],
                    chunks={},
                    engine="netcdf4",  # Force engine for safety
                )
            else:
                ds = xr.open_mfdataset(
                    input_files,
                    parallel=True,
                    chunks={},
                    coords="minimal",
                    compat="override",
                    engine="netcdf4",
                )

            # Explicit rechunking
            ds = ds.chunk(target_chunks)

            logger.info(f"   Processing {year}-{month:02d}...")
            process_and_save_data(ds, interim_file, f"{year}-{month:02d}")
            ds.close()

        except Exception as e:
            logger.error(f"Failed to process month {year}-{month:02d}: {e}")
            return

    # Merge Step
    valid_months = [f for f in monthly_files if f.exists()]

    if len(valid_months) != 12:
        logger.warning(
            f"Found {len(valid_months)}/12 interim files for {year}. Skipping merge."
        )
        return

    logger.info(f"Merging {len(valid_months)} monthly files into {output_file.name}...")

    try:
        # Chunk spatially during merge as well to keep memory low
        # Load native chunks first, then rechunk.
        ds_year = xr.open_mfdataset(
            valid_months, parallel=True, chunks={}, engine="netcdf4"
        )
        ds_year = ds_year.chunk(target_chunks)

        encoding = {var: ENCODING_PARAMS for var in ds_year.data_vars}
        ds_year.to_netcdf(output_file, encoding=encoding)

        logger.info(f"✅ Created year file: {output_file}")

        logger.info("Cleaning up interim files...")
        for f in valid_months:
            f.unlink()

    except Exception as e:
        logger.error(f"Failed during merge of {year}: {e}")
        if output_file.exists():
            output_file.unlink()


def main():
    parser = argparse.ArgumentParser(description="Process ERA5-Land daily summaries.")
    parser.add_argument("--trial", action="store_true", help="Run trial mode.")
    parser.add_argument(
        "--local", action="store_true", help="Run on local files instead of Gadi."
    )
    args = parser.parse_args()

    # Determine input and output directories based on mode
    if args.local:
        input_dir = Dirs.dir_era_land_hourly_local
        output_dir = Dirs.dir_era_land_daily_local
        n_workers = 1  # Use single worker on local to avoid communication overhead
        target_chunks = {
            "time": -1,
            "latitude": 800,
            "longitude": 800,
        }  # Larger chunks locally
        logger.info("Running in LOCAL MODE")
    else:
        input_dir = Dirs.dir_era_land
        output_dir = Dirs.dir_era_daily
        n_workers = int(os.environ.get("PBS_NCPUS", 4))
        target_chunks = {
            "time": -1,
            "latitude": 400,
            "longitude": 400,
        }  # Smaller chunks for Gadi
        logger.info("Running in GADI MODE")

    # Context manager for clean shutdown
    with Client(n_workers=n_workers, threads_per_worker=1) as client:
        logger.info(f"Dask Dashboard: {client.dashboard_link}")

        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_dir}")

        start_year = 1980
        end_year = 2025
        if args.trial:
            end_year = start_year

        for year in range(start_year, end_year + 1):
            logger.info(f"=== Processing Year: {year} ===")
            output_file = output_dir / f"{year}_daily_summaries.nc"

            try:
                year_dir = input_dir / str(year)
                if not year_dir.exists() or not any(year_dir.iterdir()):
                    logger.error(f"Directory missing/empty: {year_dir}")
                    continue

                process_year_in_months(year, year_dir, output_file, target_chunks)

            except Exception as e:
                logger.error(f"Failed to process year {year}: {e}")


if __name__ == "__main__":
    main()

"""
Local trial: python3 python_code/weather/b_daily_summaries.py --local --trial
Local full: python3 python_code/weather/b_daily_summaries.py --local
Gadi trial: python3 python_code/weather/b_daily_summaries.py --trial
Gadi full: python3 python_code/weather/b_daily_summaries.py
"""
