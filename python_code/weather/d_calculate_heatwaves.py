"""
Calculate heatwave occurrences and duration.
Refactored for Lancet Countdown analysis.

Heatwave Definition:
- A heatwave is defined as a period of at least 3 consecutive days
  where both daily minimum and maximum temperatures exceed their
  respective 95th percentile thresholds (climatology).

This version uses vectorized operations for speed while correctly
counting all heatwave days (not just the minimum length).
"""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from my_config import Vars, DirsLocal

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


def process_year_and_save(year, input_dir, output_dir, t_thresholds, var_names=None):
    """
    Worker function for parallel processing.
    """
    if var_names is None:
        var_names = ["t_min", "t_max"]

    input_file = input_dir / f"{year}_daily_summaries.nc"
    output_file = output_dir / f"heatwave_indicators_{year}.nc"

    if output_file.exists():
        return f"Skipped {year} (Exists)"

    try:
        ds = xr.open_dataset(input_file)

        # Use vectorized version for speed
        results = calculate_heatwave_metrics_vectorized(
            t_max=ds[var_names[1]],  # t_max
            t_min=ds[var_names[0]],  # t_min
            t_max_threshold=t_thresholds[1],
            t_min_threshold=t_thresholds[0],
            hw_min_length=3,
        )

        # Expand dims for concatenation later if needed
        results = results.expand_dims(dim={"year": [year]})

        # Save with compression
        encoding = {v: {"zlib": True, "complevel": 5} for v in results.data_vars}
        results.to_netcdf(output_file, encoding=encoding)

        ds.close()
        del ds, results

        return f"Processed {year}"

    except Exception as e:
        return f"Error {year}: {e}"


def main():
    # 1. Load thresholds (climatology)
    t_thresholds = []
    for var in ["t_min", "t_max"]:
        clim_file = (
            DirsLocal.e5l_q
            / f"daily_{var}_quantiles_{Vars.quantiles}_{Vars.year_reference_start}-{Vars.year_reference_end}.nc"
        )
        ds = xr.open_dataset(clim_file)
        # Load threshold into memory
        thresh = ds[var].load()
        t_thresholds.append(thresh)
        ds.close()

    # 2. Prepare Analysis Years
    years = Vars.get_analysis_years()

    # Ensure output directory exists
    output_dir = DirsLocal.dir_results_heatwaves
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Run Parallel Processing
    results = Parallel(n_jobs=4, verbose=10)(
        delayed(process_year_and_save)(
            year, DirsLocal.e5l_d, output_dir, t_thresholds, ["t_min", "t_max"]
        )
        for year in years
    )

    print("\n".join(results))


if __name__ == "__main__":
    main()
