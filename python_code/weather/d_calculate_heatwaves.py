"""
Calculate heatwave occurrences and duration in a single pass.
Refactored for Lancet Countdown analysis.

Heatwave Definition:
- A heatwave is defined as a period of at least 3 consecutive days
  where both daily minimum and maximum temperatures exceed their
  respective 95th percentile thresholds (climatology).

- Heatwaves crossing year boundaries are truncated. To fully capture these, consider loading Dec 29-31 of the previous year.

Improvements over previous versions:
- Single pass calculation for both count and duration.
- Vectorized Boolean mask creation for efficiency.
- Clearer handling of end-of-year boundary conditions.
- Memory-efficient accumulators using int16.
"""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from my_config import Vars, Dirs

# Keep attributes to preserve metadata/units
xr.set_options(keep_attrs=True)


def calculate_heatwave_metrics(datasets, thresholds, days_threshold=2):
    """
    Core logic: Calculates both count and duration of heatwaves.

    Args:
        datasets (list): [tmin_array, tmax_array] (xarray DataArrays or numpy arrays)
        thresholds (list): [tmin_threshold, tmax_threshold]
        days_threshold (int): Minimum consecutive days to qualify as heatwave (> threshold).
                              Default 2 means 3 days or more are required.

    Returns:
        xr.Dataset: Contains 'heatwave_count' and 'heatwave_days'
    """
    # 1. Create the Boolean Mask (Vectorized)
    # Using np.nan > x yields False, so we often don't need fillna(-9999)
    # unless you have specific needs.

    # Initialize with the first condition
    exceeded = datasets[0] > thresholds[0]

    # Combine with remaining conditions (e.g. tmax > thresh)
    for data, thresh in zip(datasets[1:], thresholds[1:]):
        exceeded = np.logical_and(exceeded, data > thresh)

    # Extract values for fast looping.
    # Shape: (Time, Lat, Lon)
    mask_values = exceeded.values

    # 2. Initialize Accumulators
    # We use int16 to save memory (counts won't exceed 32,000)
    out_shape = mask_values.shape[1:]

    hw_counts = np.zeros(out_shape, dtype=np.int16)
    hw_days = np.zeros(out_shape, dtype=np.int16)

    current_run = np.zeros(out_shape, dtype=np.int16)

    # 3. Iterate over Time (The "Scan Line" approach)
    # We iterate t from 0 to N.
    # If mask[t] is True: increment run.
    # If mask[t] is False: check if run was long enough, add to stats, reset run.

    n_time = mask_values.shape[0]

    for t in range(n_time):
        is_hot = mask_values[t, :, :]

        # Increment active runs
        current_run[is_hot] += 1

        # Handle ended runs (where current_run > 0 BUT is_hot is False)
        # Note: We need to handle the case where a run ends OR it is the last timestep
        if t < n_time - 1:
            # Normal case: run ends if it was active (accum > 0) and now is not hot
            ended_mask = (current_run > 0) & (~is_hot)

            # Reset accumulator for non-hot cells
            # We must process the 'ended' stats before resetting 0s,
            # but actually we can just process ended_mask.

            # Apply heatwave logic to ended runs
            valid_hw = ended_mask & (current_run > days_threshold)

            if np.any(valid_hw):
                hw_counts[valid_hw] += 1
                hw_days[valid_hw] += current_run[valid_hw]

            # Reset counter where heatwave broke
            current_run[~is_hot] = 0

        else:
            # Last timestep boundary case
            # Any runs still active are checked
            valid_hw = current_run > days_threshold
            hw_counts[valid_hw] += 1
            hw_days[valid_hw] += current_run[valid_hw]

    # 4. Wrap result in Dataset
    coords = {"latitude": datasets[0].latitude, "longitude": datasets[0].longitude}

    ds_out = xr.Dataset(
        {
            "heatwave_count": (("latitude", "longitude"), hw_counts),
            "heatwave_days": (("latitude", "longitude"), hw_days),
        },
        coords=coords,
    )

    return ds_out


def process_year_and_save(
    year, input_dir, output_dir, t_thresholds, var_names=["t_min", "t_max"]
):
    """
    Worker function for parallel processing.
    """
    input_file = input_dir / f"{year}_temperature_summary.nc"
    output_file = output_dir / f"heatwave_indicators_{year}.nc"

    if output_file.exists():
        return f"Skipped {year} (Exists)"

    # Load Data
    try:
        ds = xr.open_dataset(input_file)

        # Optional: Load Dec 29-31 of previous year here and concat
        # to fix Jan 1st boundary issues.

        data_arrays = [ds[v] for v in var_names]

        # Calculate
        results = calculate_heatwave_metrics(data_arrays, t_thresholds)

        # Expand dims for concatenation later if needed
        results = results.expand_dims(dim={"year": [year]})

        # Save (Compresion recommended for float/int maps)
        encoding = {v: {"zlib": True, "complevel": 5} for v in results.data_vars}
        results.to_netcdf(output_file, encoding=encoding)

        return f"Processed {year}"

    except Exception as e:
        return f"Error {year}: {e}"


def main():
    # 1. Setup paths and quantiles
    quantile_val = Vars.quantiles[0]  # e.g., 0.95

    # Load thresholds (climatology)
    # Ideally load this once before the loop if small enough,
    # or pass paths to workers if very large.
    t_thresholds = []
    for var in ["t_min", "t_max"]:
        q_str = "_".join([str(int(100 * q)) for q in Vars.quantiles])
        clim_file = (
            Dirs.dir_era_quantiles
            / f"daily_{var}_quantiles_{q_str}_{Vars.year_reference_start}-{Vars.year_reference_end}.nc"
        )
        with xr.open_dataset(clim_file) as ds:
            # Squeeze to remove quantile dim after selection
            thresh = (
                ds.sel(quantile=quantile_val, method="nearest").to_array().squeeze()
            )
            t_thresholds.append(thresh.load())  # Load into memory for speed

    # 2. Prepare Analysis Years
    years = Vars.get_analysis_years()

    # Ensure output directory exists
    # Using one folder for both indicators keeps things cleaner
    output_dir = Dirs.dir_results_heatwaves  # e.g. "results/heatwaves"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Run Parallel Processing
    # Single pass for both indicators
    results = Parallel(n_jobs=6, verbose=10)(
        delayed(process_year_and_save)(
            year, Dirs.dir_era_daily, output_dir, t_thresholds, ["t_min", "t_max"]
        )
        for year in years
    )

    print("\n".join(results))


if __name__ == "__main__":
    # pass
    main()
