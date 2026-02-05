"""
Calculate daily summaries (min, mean, max) from ERA5-Land hourly data for a single year.

This script is optimized for NCI Gadi HPC:
- Processes one year at a time (submitted as separate PBS jobs)
- Reads directly from /g/data/zz93/era5-land/reanalysis/2t
- Writes output to /scratch/mn51/ft8695/era5-land/daily/2t for fast I/O
- Processes each month individually, saves interim results, then combines into yearly output
- Uses Dask distributed for parallel processing

Usage:
    python python_code/weather/a_daily_summaries_gadi.py --year 1980
    python python_code/weather/a_daily_summaries_gadi.py --year 1980 --trial  # Process only January for testing

    --trial: Only processes January (for testing the pipeline)
    --year 1980: Processes the entire year 1980 (all 12 months)

Valid years: 1979-2025
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

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


def process_month(
    year: int, month: int, input_dir: Path, interim_dir: Path
) -> Optional[Path]:
    """
    Process a single month of ERA5-Land hourly data to daily summaries.

    Args:
        year: The year to process
        month: The month to process (1-12)
        input_dir: Directory containing hourly data
        interim_dir: Directory to save interim monthly results

    Returns:
        Path to the saved interim file
    """
    month_str = f"{year}{month:02d}"
    interim_file = interim_dir / f"{year}_{month:02d}_daily.nc"

    if interim_file.exists():
        logger.info(f"   ✅ Interim file exists: {interim_file.name}")
        return interim_file

    # Find input files for this month
    input_files = sorted(list(input_dir.glob(f"*{month_str}*.nc")))

    if not input_files:
        logger.warning(f"   ⚠️ No input files found for {year}-{month:02d}")
        return None

    logger.info(f"   Processing {year}-{month:02d} ({len(input_files)} files)...")

    try:
        # Open dataset
        if len(input_files) == 1:
            ds = xr.open_dataset(input_files[0], chunks={})
        else:
            ds = xr.open_mfdataset(
                input_files,
                parallel=True,
                chunks={},
                coords="minimal",
                compat="override",
                combine="by_coords",
            )

        # Rechunk for processing
        ds = ds.chunk(TARGET_CHUNKS)

        # Get variable name
        var_name = list(ds.data_vars)[0]

        # Resample to daily
        resampler = ds[var_name].resample(time="1D")
        daily_ds = xr.Dataset()
        daily_ds["t_min"] = resampler.min()
        daily_ds["t_mean"] = resampler.mean()
        daily_ds["t_max"] = resampler.max()

        # Add metadata
        daily_ds.attrs["description"] = (
            f"Daily summaries of 2-meter temperature from ERA5-Land for {year}-{month:02d}"
        )
        daily_ds.attrs["source"] = "ERA5-Land hourly data from /g/data/zz93"
        daily_ds.attrs["processing"] = "Resampled to daily min, mean, max"
        daily_ds.attrs["created_by"] = "a_daily_summaries_gadi.py"

        # Rechunk for output
        output_chunks = {"time": -1, "latitude": 600, "longitude": 1200}
        daily_ds = daily_ds.chunk(output_chunks)

        # Encoding
        time_chunks = daily_ds.dims["time"]  # Use actual number of days in month
        encoding = {
            var: {
                **ENCODING_PARAMS,
                "chunksizes": (time_chunks, 600, 1200),
            }
            for var in daily_ds.data_vars
        }

        # Save
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            daily_ds.to_netcdf(interim_file, encoding=encoding, compute=True)

        logger.info(f"   ✅ Saved interim: {interim_file.name}")

        # Clean up memory
        ds.close()
        del ds
        del daily_ds

        return interim_file

    except Exception as e:
        logger.error(f"   ❌ Failed to process {year}-{month:02d}: {e}")
        return None


def process_year(year: int, trial: bool = False) -> None:
    """
    Process a single year of ERA5-Land hourly data to daily summaries.
    Now processes month by month and combines results.

    Args:
        year: The year to process (e.g., 1980). Valid range: 1979-2025
        trial: If True, only process January for testing

    The function:
    1. Processes each month individually, saving interim files
    2. Combines all monthly interim files into yearly output
    3. Cleans up interim files after successful merge
    """
    # Validate year range
    if year < 1979 or year > 2025:
        logger.error(f"❌ Invalid year: {year}. Valid range is 1979-2025")
        sys.exit(1)

    input_dir = DirsGadi.e5l_h / str(year)
    output_dir = DirsGadi.e5l_d
    ensure_directories([output_dir])

    output_file = output_dir / f"{year}_daily_summaries.nc"
    interim_dir = output_dir / "interim"
    ensure_directories([interim_dir])

    # Check if already processed
    if output_file.exists():
        logger.info(f"✅ Output already exists: {output_file}")
        logger.info("Skipping processing. Delete file to reprocess.")
        return

    # Check if input directory exists
    if not input_dir.exists():
        logger.error(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    # Determine months to process
    if trial:
        months_to_process = [1]  # Only January for trial
        logger.info(f"🧪 TRIAL MODE: Processing only January {year}")
    else:
        months_to_process = list(range(1, 13))  # All months
        logger.info(f"📅 Processing full year: {year}")

    # Process each month
    interim_files = []
    for month in months_to_process:
        interim_file = process_month(year, month, input_dir, interim_dir)
        if interim_file:
            interim_files.append(interim_file)

    # Check if we have all expected files
    expected_count = 1 if trial else 12
    if len(interim_files) != expected_count:
        logger.error(
            f"❌ Expected {expected_count} interim files, got {len(interim_files)}. "
            "Cannot proceed with merge."
        )
        # Clean up interim files on failure
        logger.info("🧹 Cleaning up interim files due to merge failure...")
        for f in interim_files:
            if f.exists():
                f.unlink()
        try:
            interim_dir.rmdir()
        except OSError:
            pass  # Directory not empty or doesn't exist
        return

    logger.info(f"📋 Merging {len(interim_files)} monthly files into yearly output...")

    # Open and combine all interim files
    ds_year = xr.open_mfdataset(
        interim_files,
        parallel=True,
        chunks={},
        coords="minimal",
        compat="override",
        combine="by_coords",
    )

    try:
        # Rechunk for final output
        output_chunks = {"time": -1, "latitude": 600, "longitude": 1200}
        ds_year = ds_year.chunk(output_chunks)

        # Update metadata for yearly file
        ds_year.attrs["description"] = (
            f"Daily summaries of 2-meter temperature from ERA5-Land for {year}"
        )
        ds_year.attrs["source"] = "ERA5-Land hourly data from /g/data/zz93"
        ds_year.attrs["processing"] = (
            "Resampled to daily min, mean, max (monthly processing)"
        )
        ds_year.attrs["created_by"] = "a_daily_summaries_gadi.py"

        # Encoding for yearly file
        encoding = {
            var: {
                **ENCODING_PARAMS,
                "chunksizes": (365 if not trial else 31, 600, 1200),
            }
            for var in ds_year.data_vars
        }

        # Save final output
        logger.info(f"💾 Saving yearly output: {output_file}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            ds_year.to_netcdf(output_file, encoding=encoding, compute=True)

        logger.info(f"✅ Successfully created: {output_file.name}")
        logger.info(f"File size: {output_file.stat().st_size / 1e9:.2f} GB")
        logger.info("💡 To copy this file back to local repository, run:")
        logger.info(f"   python python_code/copy_files_hpc.py --copy-daily {year}")

        # Clean up memory
        ds_year.close()
        del ds_year

        # Clean up interim files
        logger.info("🧹 Cleaning up interim files...")
        for f in interim_files:
            f.unlink()

        # Remove interim directory if empty
        try:
            interim_dir.rmdir()
        except OSError:
            pass  # Directory not empty or doesn't exist

    except Exception as e:
        logger.error(f"❌ Failed to merge and save: {e}")
        if output_file.exists():
            output_file.unlink()
        sys.exit(1)

    finally:
        # ds_year is already closed and deleted in the try block
        pass


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

    args = parser.parse_args()

    # Set up Dask distributed client with proper memory limits and spill-to-disk
    logger.info("🚀 Starting Dask distributed client...")
    client = Client(
        n_workers=8,
        threads_per_worker=1,
        local_directory="/scratch/mn51/ft8695/dask-worker-space",
        memory_target_fraction=0.6,  # Target 60% memory usage before spilling
        memory_spill_fraction=0.7,  # Spill to disk at 70% memory usage
        memory_pause_fraction=0.8,  # Pause workers at 80% memory usage
    )
    logger.info(f"Dask Dashboard: {client.dashboard_link}")
    logger.info("Workers: 8")
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
