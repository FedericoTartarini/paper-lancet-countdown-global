"""
Calculate heatwave occurrences and duration for a single year.

This script is optimized for NCI Gadi HPC:
- Processes one year at a time (submitted as separate PBS jobs)
- Processes data in LATITUDE CHUNKS to reduce memory usage
- Saves interim chunk results to allow resuming if job fails
- Reads daily summaries from /scratch/mn51/ft8695/era5-land/daily/2t
- Reads quantile thresholds from /scratch/mn51/ft8695/era5-land/quantiles
- Writes output to /scratch/mn51/ft8695/heatwaves

Heatwave Definition:
- A heatwave is defined as a period of at least 3 consecutive days
  where both daily minimum and maximum temperatures exceed their
  respective 95th percentile thresholds (climatology).

Usage:
    read the instructions on README_HEATWAVES.md in /gadi for how to submit this script as a job for each year.

Valid years: 1980-2024 (based on Vars.get_analysis_years())
"""

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path.cwd()

sys.path.append(str(project_root))

import numpy as np
import xarray as xr

from my_config import Vars, DirsGadi, ensure_directories
from python_code.log_config import setup_logging

# Set up logging
logger = setup_logging(project_root, log_filename="calculate_heatwaves.log")

# Processing configuration
LAT_CHUNK_SIZE = 200  # Process 200 latitude rows at a time (~11% of global grid)


# Keep attributes to preserve metadata/units
xr.set_options(keep_attrs=True)


def count_heatwave_days_vectorized(
    t_max: xr.DataArray,
    t_min: xr.DataArray,
    t_max_threshold: xr.DataArray,
    t_min_threshold: xr.DataArray,
    hw_min_length: int = 3,
) -> xr.DataArray:
    """
    Calculate heatwave days using vectorized rolling operations.

    A day is a heatwave day if it is part of a run of at least
    `hw_min_length` consecutive days where both t_max > t_max_threshold
    and t_min > t_min_threshold.

    Args:
        t_max: Daily maximum temperature (time, lat, lon)
        t_min: Daily minimum temperature (time, lat, lon)
        t_max_threshold: Threshold for t_max (lat, lon)
        t_min_threshold: Threshold for t_min (lat, lon)
        hw_min_length: Minimum consecutive days for heatwave (default 3)

    Returns:
        Boolean DataArray marking heatwave days
    """
    # 1. Identify hot days (both conditions met)
    hot_day = (t_max > t_max_threshold) & (t_min > t_min_threshold)

    # 2. Find starts of heatwaves (runs of hw_min_length consecutive hot days)
    # Rolling min == 1 means all days in the window are hot
    hw_start = (
        hot_day.astype(int).rolling(time=hw_min_length, min_periods=hw_min_length).min()
        == 1
    )

    # 3. Extend heatwave flag forward to cover all days in each heatwave
    # A day is a heatwave day if any of the next (hw_min_length-1) days
    # or itself is a heatwave start
    heatwave_days = (
        hw_start.rolling(time=hw_min_length, min_periods=1, center=False)
        .max()
        .shift(time=-(hw_min_length - 1))
        .fillna(0)
        == 1
    )

    # 4. Also mark trailing days of heatwaves that extend beyond minimum
    # A hot day following a heatwave day is also a heatwave day
    # We need to propagate forward through consecutive hot days
    heatwave_days = heatwave_days | (
        heatwave_days.shift(time=1).fillna(False) & hot_day
    )

    # Repeat to capture longer heatwaves (iterate a few times)
    for _ in range(10):  # Max 10 extra days beyond minimum
        extended = heatwave_days.shift(time=1).fillna(False) & hot_day
        if not extended.any():
            break
        heatwave_days = heatwave_days | extended

    heatwave_days.name = "heatwave_days"
    heatwave_days.attrs["units"] = "1"
    heatwave_days.attrs["long_name"] = "Heatwave day indicator"

    return heatwave_days


def calculate_heatwave_metrics_vectorized(
    t_max: xr.DataArray,
    t_min: xr.DataArray,
    t_max_threshold: xr.DataArray,
    t_min_threshold: xr.DataArray,
    hw_min_length: int = 3,
) -> xr.Dataset:
    """
    Calculate heatwave count and total days using vectorized operations.

    Args:
        t_max: Daily maximum temperature (time, lat, lon)
        t_min: Daily minimum temperature (time, lat, lon)
        t_max_threshold: Threshold for t_max (lat, lon)
        t_min_threshold: Threshold for t_min (lat, lon)
        hw_min_length: Minimum consecutive days for heatwave (default 3)

    Returns:
        Dataset with 'heatwave_count' and 'heatwave_days'
    """
    # Get heatwave day mask
    hw_days_mask = count_heatwave_days_vectorized(
        t_max, t_min, t_max_threshold, t_min_threshold, hw_min_length
    )

    # Total heatwave days
    total_hw_days = hw_days_mask.sum(dim="time")

    # Count heatwave events (count transitions from 0 to 1)
    # A new heatwave starts when current day is heatwave but previous wasn't
    hw_starts = hw_days_mask.astype(int).diff(dim="time") == 1
    # Also count if first day is a heatwave day
    first_day_hw = hw_days_mask.isel(time=0)
    hw_count = hw_starts.sum(dim="time") + first_day_hw.astype(int)

    # Create output dataset
    ds_out = xr.Dataset(
        {
            "heatwave_count": hw_count.astype(np.int16),
            "heatwave_days": total_hw_days.astype(np.int16),
        }
    )

    ds_out["heatwave_count"].attrs = {
        "units": "1",
        "long_name": "Number of heatwave events",
    }
    ds_out["heatwave_days"].attrs = {
        "units": "days",
        "long_name": "Total heatwave days",
    }

    return ds_out


def process_year_chunked(
    year: int,
    input_dir: Path,
    output_dir: Path,
    t_thresholds: list,
    var_names: list = None,
    lat_chunk_size: int = LAT_CHUNK_SIZE,
) -> str:
    """
    Process heatwave indicators for a single year in latitude chunks.

    This approach:
    1. Processes data in latitude chunks to reduce memory usage
    2. Saves interim chunk results to allow resuming
    3. Combines chunks at the end

    Args:
        year: Year to process
        input_dir: Directory containing daily summary files
        output_dir: Directory to save heatwave output files
        t_thresholds: List of [t_min_threshold, t_max_threshold] DataArrays
        var_names: Names of temperature variables [t_min_name, t_max_name]
        lat_chunk_size: Number of latitude rows to process at once

    Returns:
        Status message
    """
    if var_names is None:
        var_names = ["t_min", "t_max"]

    input_file = input_dir / f"{year}_daily_summaries.nc"
    output_file = output_dir / f"heatwave_indicators_{year}.nc"
    interim_dir = output_dir / "interim" / str(year)

    # Check if already processed
    if output_file.exists():
        logger.info(f"✅ Output already exists: {output_file.name}")
        return f"Skipped {year} (Exists)"

    # Check if input file exists
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        return f"Error {year}: Input file not found"

    # Create interim directory
    ensure_directories([interim_dir])

    logger.info(f"📂 Opening input file: {input_file.name}")
    start_time = time.time()

    try:
        # Open dataset lazily
        ds = xr.open_dataset(input_file)

        # Coordinate sanity checks (protect against lon/lat mismatch)
        if "longitude" in ds.coords and "longitude" in t_thresholds[0].coords:
            if ds.longitude.size != t_thresholds[0].longitude.size:
                logger.warning(
                    "⚠️ Longitude size mismatch: data=%s, thresholds=%s",
                    ds.longitude.size,
                    t_thresholds[0].longitude.size,
                )
            if not np.allclose(
                ds.longitude.values, t_thresholds[0].longitude.values, equal_nan=True
            ):
                logger.warning(
                    "⚠️ Longitude values differ between data and thresholds. "
                    "Data lon range: %.2f to %.2f | Threshold lon range: %.2f to %.2f",
                    float(ds.longitude.values.min()),
                    float(ds.longitude.values.max()),
                    float(t_thresholds[0].longitude.values.min()),
                    float(t_thresholds[0].longitude.values.max()),
                )
        if "latitude" in ds.coords and "latitude" in t_thresholds[0].coords:
            if ds.latitude.size != t_thresholds[0].latitude.size:
                logger.warning(
                    "⚠️ Latitude size mismatch: data=%s, thresholds=%s",
                    ds.latitude.size,
                    t_thresholds[0].latitude.size,
                )

        lat_size = ds.dims["latitude"]

        # Calculate chunk boundaries
        lat_starts = list(range(0, lat_size, lat_chunk_size))
        n_chunks = len(lat_starts)

        logger.info(
            f"🔧 Processing in {n_chunks} latitude chunks (size={lat_chunk_size})"
        )

        chunk_files = []
        chunks_completed = 0

        for i, lat_start in enumerate(lat_starts):
            lat_end = min(lat_start + lat_chunk_size, lat_size)
            chunk_file = interim_dir / f"chunk_{i:03d}.nc"
            chunk_files.append(chunk_file)

            # Check if chunk already processed (resume capability)
            if chunk_file.exists():
                logger.info(f"   ⏭️  Chunk {i + 1}/{n_chunks} already exists, skipping")
                chunks_completed += 1
                continue

            chunk_start_time = time.time()

            # Load chunk data into memory
            logger.info(
                f"   📊 Chunk {i + 1}/{n_chunks}: lat[{lat_start}:{lat_end}] "
                f"({lat_end - lat_start} rows)"
            )

            # Select latitude chunk from input data and load to memory
            ds_chunk = ds.isel(latitude=slice(lat_start, lat_end))
            t_min_chunk = ds_chunk[var_names[0]].values  # Load as numpy array
            t_max_chunk = ds_chunk[var_names[1]].values

            # Select matching chunk from thresholds
            t_min_thresh = (
                t_thresholds[0].isel(latitude=slice(lat_start, lat_end)).values
            )
            t_max_thresh = (
                t_thresholds[1].isel(latitude=slice(lat_start, lat_end)).values
            )

            # Calculate heatwave metrics using numpy (faster than xarray for loaded data)
            hw_count, hw_days = calculate_heatwave_metrics_numpy(
                t_max=t_max_chunk,
                t_min=t_min_chunk,
                t_max_threshold=t_max_thresh,
                t_min_threshold=t_min_thresh,
                hw_min_length=3,
            )

            # Create xarray dataset for this chunk
            chunk_ds = xr.Dataset(
                {
                    "heatwave_count": (
                        ["latitude", "longitude"],
                        hw_count.astype(np.float32),
                    ),
                    "heatwave_days": (
                        ["latitude", "longitude"],
                        hw_days.astype(np.float32),
                    ),
                },
                coords={
                    "latitude": ds_chunk.latitude.values,
                    "longitude": ds_chunk.longitude.values,
                },
            )

            # Save chunk
            chunk_ds.to_netcdf(chunk_file)

            chunk_time = time.time() - chunk_start_time
            chunks_completed += 1
            elapsed = time.time() - start_time
            remaining = (elapsed / chunks_completed) * (n_chunks - chunks_completed)

            logger.info(
                f"      ✅ Chunk {i + 1}/{n_chunks} done in {chunk_time:.1f}s | "
                f"Elapsed: {elapsed / 60:.1f}min | ETA: {remaining / 60:.1f}min"
            )

            # Free memory
            del ds_chunk, t_min_chunk, t_max_chunk, t_min_thresh, t_max_thresh
            del hw_count, hw_days, chunk_ds
            gc.collect()

        # Close the input dataset
        ds.close()

        # Combine all chunks
        logger.info(f"🔗 Combining {n_chunks} chunks...")
        combine_start = time.time()

        # Load and concatenate all chunk files
        chunk_datasets = [xr.open_dataset(f) for f in chunk_files]
        combined = xr.concat(chunk_datasets, dim="latitude")

        # Close chunk datasets
        for ds_chunk in chunk_datasets:
            ds_chunk.close()

        # Add year dimension
        combined = combined.expand_dims(dim={"year": [year]})

        # Add metadata
        combined.attrs["description"] = f"Heatwave indicators for {year}"
        combined.attrs["source"] = "ERA5-Land daily summaries"
        combined.attrs["heatwave_definition"] = (
            "3+ consecutive days with both t_max and t_min exceeding 95th percentile"
        )
        combined.attrs["reference_period"] = (
            f"{Vars.year_reference_start}-{Vars.year_reference_end}"
        )
        combined.attrs["created_by"] = "d_calculate_heatwaves_gadi.py"

        combined["heatwave_count"].attrs = {
            "units": "1",
            "long_name": "Number of heatwave events",
        }
        combined["heatwave_days"].attrs = {
            "units": "days",
            "long_name": "Total heatwave days",
        }

        # Save final output
        encoding = {
            v: {
                "zlib": True,
                "complevel": 5,
                "dtype": "float32",
                "_FillValue": np.nan,
            }
            for v in combined.data_vars
        }
        combined.to_netcdf(output_file, encoding=encoding)
        combined.close()

        logger.info(f"   Combine time: {(time.time() - combine_start):.1f}s")

        # Clean up interim files
        logger.info("🧹 Cleaning up interim files...")
        shutil.rmtree(interim_dir)

        total_time = time.time() - start_time
        logger.info(f"✅ Successfully created: {output_file.name}")
        logger.info(f"   File size: {output_file.stat().st_size / 1e6:.2f} MB")
        logger.info(f"   Total time: {total_time / 60:.1f} minutes")

        return f"Processed {year}"

    except Exception as e:
        logger.error(f"❌ Error processing {year}: {e}", exc_info=True)
        # Don't clean up interim files on error - allows resume
        logger.info("   Interim files preserved for resume capability")
        return f"Error {year}: {e}"


def calculate_heatwave_metrics_numpy(
    t_max: np.ndarray,
    t_min: np.ndarray,
    t_max_threshold: np.ndarray,
    t_min_threshold: np.ndarray,
    hw_min_length: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate heatwave count and days using numpy (faster than xarray for loaded data).

    Args:
        t_max: Daily max temperature array (time, lat, lon)
        t_min: Daily min temperature array (time, lat, lon)
        t_max_threshold: Threshold array (lat, lon)
        t_min_threshold: Threshold array (lat, lon)
        hw_min_length: Minimum consecutive days for heatwave

    Returns:
        Tuple of (heatwave_count, heatwave_days) arrays with shape (lat, lon)
    """
    n_time, n_lat, n_lon = t_max.shape

    # Identify cells with no valid data (oceans or missing thresholds)
    ocean_mask = (
        np.isnan(t_max).all(axis=0)
        | np.isnan(t_min).all(axis=0)
        | np.isnan(t_max_threshold)
        | np.isnan(t_min_threshold)
    )

    # Initialize output arrays as float to allow NaN for oceans
    heatwave_days = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    heatwave_count = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    # 1. Identify hot days (both conditions met)
    hot_day = (t_max > t_max_threshold) & (t_min > t_min_threshold)

    # 2. Find consecutive hot day runs using convolution
    kernel = np.ones(hw_min_length)

    # Process each grid cell
    for lat_idx in range(n_lat):
        for lon_idx in range(n_lon):
            if ocean_mask[lat_idx, lon_idx]:
                continue

            hot_series = hot_day[:, lat_idx, lon_idx].astype(np.int8)

            if hot_series.sum() == 0:
                heatwave_days[lat_idx, lon_idx] = 0.0
                heatwave_count[lat_idx, lon_idx] = 0.0
                continue

            # Find runs of consecutive hot days
            conv = np.convolve(hot_series, kernel, mode="valid")
            hw_starts = np.where(conv == hw_min_length)[0]

            if len(hw_starts) == 0:
                heatwave_days[lat_idx, lon_idx] = 0.0
                heatwave_count[lat_idx, lon_idx] = 0.0
                continue

            # Mark all heatwave days
            is_hw_day = np.zeros(n_time, dtype=bool)
            for start in hw_starts:
                # Mark the initial 3 days
                is_hw_day[start : start + hw_min_length] = True
                # Extend forward while still hot
                end = start + hw_min_length
                while end < n_time and hot_series[end]:
                    is_hw_day[end] = True
                    end += 1

            heatwave_days[lat_idx, lon_idx] = float(is_hw_day.sum())

            # Count distinct heatwave events (transitions from 0 to 1)
            transitions = np.diff(is_hw_day.astype(np.int8))
            heatwave_count[lat_idx, lon_idx] = float(
                (transitions == 1).sum() + (1 if is_hw_day[0] else 0)
            )

    return heatwave_count, heatwave_days


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Calculate heatwave indicators from daily temperature data (Gadi-optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process (e.g., 2020)",
    )

    args = parser.parse_args()

    # Validate year range
    valid_years = Vars.get_analysis_years()
    if args.year not in valid_years:
        logger.error(
            f"❌ Invalid year: {args.year}. Valid range is {valid_years[0]}-{valid_years[-1]}"
        )
        sys.exit(1)

    logger.info("🚀 Starting heatwave calculation...")
    logger.info(f"   Latitude chunk size: {LAT_CHUNK_SIZE}")

    try:
        # Load thresholds (climatology)
        logger.info("📂 Loading temperature thresholds...")
        t_thresholds = []
        for var in ["t_min", "t_max"]:
            clim_file = (
                DirsGadi.e5l_q
                / f"daily_{var}_quantiles_{Vars.quantiles}_{Vars.year_reference_start}-{Vars.year_reference_end}.nc"
            )
            if not clim_file.exists():
                logger.error(f"❌ Threshold file not found: {clim_file}")
                sys.exit(1)
            ds = xr.open_dataset(clim_file)
            thresh = ds[var].load()
            t_thresholds.append(thresh)
            ds.close()
            logger.info(f"   Loaded: {clim_file.name}")

        # Ensure output directory exists
        output_dir = DirsGadi.hw_min_max
        ensure_directories([output_dir])
        logger.info(f"📁 Output directory: {output_dir}")

        # Process the year using chunked approach
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing year {args.year}")
        logger.info(f"{'=' * 60}")

        result = process_year_chunked(
            year=args.year,
            input_dir=DirsGadi.e5l_d,
            output_dir=output_dir,
            t_thresholds=t_thresholds,
            var_names=["t_min", "t_max"],
            lat_chunk_size=LAT_CHUNK_SIZE,
        )
        logger.info(f"Result: {result}")

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("🎉 Done!")


if __name__ == "__main__":
    main()
