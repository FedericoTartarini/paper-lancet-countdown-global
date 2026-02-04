"""
Calculate daily summaries (min, mean, max) from ERA5-Land hourly data for a single year.

This script is optimized for NCI Gadi HPC:
- Processes one year at a time (submitted as separate PBS jobs)
- Reads directly from /g/data/zz93/era5-land/reanalysis/2t
- Writes output to /scratch/mn51/ft8695/era5-land/daily/2t for fast I/O
- Loads all monthly files at once and creates yearly output directly (no interim files)
- Uses Dask distributed for parallel processing

Usage:
    python python_code/weather/b_daily_summaries_gadi.py --year 1980
    python python_code/weather/b_daily_summaries_gadi.py --year 1980 --trial  # Process only January for testing

    --trial: Only processes January (for testing the pipeline)
    --year 1980: Processes the entire year 1980 (all 12 months)

Valid years: 1979-2025
"""

import argparse
import sys
import warnings
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import xarray as xr
from dask.distributed import Client

from my_config import DirsGadi, ensure_directories
from python_code.log_config import setup_logging

# Set up logging
logger = setup_logging(project_root)

# Optimization settings for output files
ENCODING_PARAMS = {
    "zlib": True,
    "complevel": 1,
    "shuffle": True,
    "dtype": "float32",
    "least_significant_digit": 1,
    "_FillValue": -9999.0,
}

# Chunking strategy for Gadi (optimized for transpose operations)
TARGET_CHUNKS = {"time": -1, "latitude": 200, "longitude": 200}


def process_year(year: int, trial: bool = False) -> None:
    """
    Process a single year of ERA5-Land hourly data to daily summaries.

    Args:
        year: The year to process (e.g., 1980). Valid range: 1979-2025
        trial: If True, only process January for testing

    The function:
    1. Locates all monthly files for the year
    2. Opens them with xarray's open_mfdataset (parallel, lazy loading)
    3. Resamples to daily min, mean, max
    4. Saves compressed output to scratch
    """
    # Validate year range
    if year < 1979 or year > 2025:
        logger.error(f"❌ Invalid year: {year}. Valid range is 1979-2025")
        sys.exit(1)

    input_dir = DirsGadi.dir_era_land_hourly / str(year)
    output_dir = DirsGadi.dir_era_daily
    ensure_directories([output_dir])

    output_file = output_dir / f"{year}_daily_summaries.nc"

    # Check if already processed
    if output_file.exists():
        logger.info(f"✅ Output already exists: {output_file}")
        logger.info("Skipping processing. Delete file to reprocess.")
        return

    # Check if input directory exists
    if not input_dir.exists():
        logger.error(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    # Find input files
    if trial:
        # Trial mode: only process first month (January)
        pattern = f"*{year}01*.nc"
        logger.info(f"🧪 TRIAL MODE: Processing only January {year}")
    else:
        # Full mode: all months
        pattern = "*.nc"
        logger.info(f"📅 Processing full year: {year}")

    input_files = sorted(list(input_dir.glob(pattern)))

    if not input_files:
        logger.error(f"❌ No input files found in {input_dir} with pattern {pattern}")
        sys.exit(1)

    logger.info(f"Found {len(input_files)} input file(s)")
    logger.debug(f"Files: {[f.name for f in input_files[:3]]}...")  # Show first 3

    # Verify we have the expected number of files
    if not trial:
        expected_files = 12  # One file per month for a full year
        if len(input_files) != expected_files:
            logger.warning(
                f"⚠️ Expected {expected_files} files but found {len(input_files)}. "
                f"This may indicate missing months."
            )
    else:
        logger.info(f"Trial mode: processing {len(input_files)} file(s)")

    # Open all files at once with xarray
    logger.info("Opening dataset with xarray.open_mfdataset...")
    try:
        ds = xr.open_mfdataset(
            input_files,
            parallel=True,
            chunks={},  # Load with native chunks first
            coords="minimal",
            compat="override",
            combine="by_coords",
        )
        logger.info(f"Dataset opened. Shape: {dict(ds.sizes)}")

        # Verify data spans the expected time range
        time_start = ds.time.min().values
        time_end = ds.time.max().values
        logger.info(f"Time range: {time_start} to {time_end}")

        # Check if data is hourly
        time_diff = ds.time.diff(dim="time")
        time_freq = time_diff.median().values
        logger.info(f"Median time step: {time_freq}")

        if not trial:
            # Full year checks
            import pandas as pd

            expected_start = pd.Timestamp(f"{year}-01-01")
            expected_end = pd.Timestamp(f"{year}-12-31 23:00:00")

            if pd.Timestamp(time_start) > expected_start + pd.Timedelta(days=1):
                logger.warning(
                    f"⚠️ Data starts at {time_start}, expected around {expected_start}"
                )

            if pd.Timestamp(time_end) < expected_end - pd.Timedelta(days=1):
                logger.warning(
                    f"⚠️ Data ends at {time_end}, expected around {expected_end}"
                )

            # Check hourly frequency (should be 1 hour or 3600 seconds)
            expected_freq = pd.Timedelta(hours=1)
            if abs(pd.Timedelta(time_freq) - expected_freq) > pd.Timedelta(minutes=1):
                logger.warning(
                    f"⚠️ Time frequency is {time_freq}, expected hourly (1 hour)"
                )
            else:
                logger.info("✅ Data confirmed as hourly")

        # Rechunk to optimal size for processing
        logger.info(f"Rechunking to: {TARGET_CHUNKS}")
        ds = ds.chunk(TARGET_CHUNKS)

    except Exception as e:
        logger.error(f"❌ Failed to open dataset: {e}")
        sys.exit(1)

    # Get variable name (should be 't2m' or similar)
    var_name = list(ds.data_vars)[0]
    logger.info(f"Processing variable: {var_name}")

    # Resample to daily frequency
    logger.info("Resampling to daily summaries (min, mean, max)...")
    resampler = ds[var_name].resample(time="1D")

    # Create output dataset with daily summaries
    daily_ds = xr.Dataset()
    daily_ds["t_min"] = resampler.min()
    daily_ds["t_mean"] = resampler.mean()
    daily_ds["t_max"] = resampler.max()

    # Add metadata
    daily_ds.attrs["description"] = (
        f"Daily summaries of 2-meter temperature from ERA5-Land for {year}"
    )
    daily_ds.attrs["source"] = "ERA5-Land hourly data from /g/data/zz93"
    daily_ds.attrs["processing"] = "Resampled to daily min, mean, max"
    daily_ds.attrs["created_by"] = "b_daily_summaries_gadi.py"

    # Rechunk for optimal write performance (larger chunks for output)
    # Daily data is much smaller, so we can use larger chunks for writing
    logger.info("Rechunking for optimal write performance...")
    # Use chunks that fit within actual dimensions (1801x3600)
    # Conservative chunks for stability
    output_chunks = {"time": -1, "latitude": 600, "longitude": 1200}
    daily_ds = daily_ds.chunk(output_chunks)

    # Prepare encoding for compression with chunking specification
    encoding = {
        var: {
            **ENCODING_PARAMS,
            "chunksizes": (
                365 if not trial else 31,
                600,
                1200,
            ),  # Optimize for yearly or monthly access
        }
        for var in daily_ds.data_vars
    }

    # Save to disk
    logger.info(f"💾 Saving to: {output_file}")
    logger.info("Computing and writing data (this may take a few minutes)...")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            # Use compute=True to force computation before writing (faster than lazy write)
            daily_ds.to_netcdf(
                output_file,
                encoding=encoding,
                compute=True,
            )

        logger.info(f"✅ Successfully created: {output_file.name}")
        logger.info(f"File size: {output_file.stat().st_size / 1e9:.2f} GB")

    except Exception as e:
        logger.error(f"❌ Failed to save output: {e}")
        if output_file.exists():
            output_file.unlink()
        sys.exit(1)

    finally:
        ds.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process ERA5-Land hourly data to daily summaries for a single year (Gadi-optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process (e.g., 1980)",
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Trial mode: only process January for testing",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of Dask workers (default: 8)",
    )
    parser.add_argument(
        "--memory_limit",
        type=str,
        default="4GB",
        help="Memory limit per worker (e.g., '4GB', default: '4GB')",
    )

    args = parser.parse_args()

    # Set up Dask distributed client with proper memory limits and spill-to-disk
    logger.info("🚀 Starting Dask distributed client...")
    client = Client(
        n_workers=args.n_workers,
        threads_per_worker=1,
        memory_limit=args.memory_limit,
        local_directory="/scratch/mn51/ft8695/dask-worker-space",
        memory_target_fraction=0.6,  # Target 60% memory usage before spilling
        memory_spill_fraction=0.7,  # Spill to disk at 70% memory usage
        memory_pause_fraction=0.8,  # Pause workers at 80% memory usage
    )
    logger.info(f"Dask Dashboard: {client.dashboard_link}")
    logger.info(f"Workers: {args.n_workers}, Memory per worker: {args.memory_limit}")
    logger.info("Memory management: target=60%, spill=70%, pause=80%")

    try:
        process_year(args.year, trial=args.trial)
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        client.close()

    logger.info("🎉 Done!")


if __name__ == "__main__":
    main()
