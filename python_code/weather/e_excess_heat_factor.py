"""
Calculate Excess Heat Factor (EHF) with Year-Boundary Padding.

Definition (Nairn & Fawcett 2015):
    EHF = EHI_sig * max(1, EHI_accl)
    Where:
      T_mean = (T_min + T_max) / 2
      T_3day = Rolling 3-day mean of T_mean
      T_95   = Historical 95th percentile of T_mean

      EHI_sig  = T_3day - T_95
      EHI_accl = T_3day - (Rolling 30-day mean of T_mean)

Requirement:
    - To calculate Jan 1st correctly, we MUST load Dec of the previous year.
    - We load [Year-1 (Dec)] + [Year (Jan-Dec)] + [Year+1 (Jan 1-2)]
      to ensure full valid coverage.
"""

import xarray as xr
from joblib import Parallel, delayed
from my_config import Vars, Dirs

# Preserve attributes
xr.set_options(keep_attrs=True)


def load_padded_data(year, input_dir):
    """
    Loads the target year PLUS:
    - Last 32 days of Previous Year (for 30-day acclimatization + 3-day rolling window)
    - First 2 days of Next Year (for 3-day forward rolling window overlap, if needed)

    Returns: A continuous DataArray of T_mean.
    """
    files_to_load = []

    # 1. Target Year
    files_to_load.append(input_dir / f"{year}_temperature_summary.nc")

    # 2. Previous Year (Check existence)
    prev_file = input_dir / f"{year - 1}_temperature_summary.nc"
    if prev_file.exists():
        files_to_load.append(prev_file)

    # 3. Next Year (Check existence)
    next_file = input_dir / f"{year + 1}_temperature_summary.nc"
    if next_file.exists():
        files_to_load.append(next_file)

    # Load and Combine
    # We load 't_min' and 't_max' to calculate 't_mean'
    try:
        ds = xr.open_mfdataset(files_to_load, combine="by_coords")

        # Calculate Daily Mean Temperature
        t_mean = (ds["t_min"] + ds["t_max"]) / 2
        t_mean.name = "t_mean"

        # Slice to relevant buffer:
        # Start: Dec 1 of Prev Year
        # End: Jan 5 of Next Year
        start_date = f"{year - 1}-12-01"
        end_date = f"{year + 1}-01-05"

        t_mean_padded = t_mean.sel(time=slice(start_date, end_date))
        return t_mean_padded.load()  # Load into memory for fast rolling ops

    except Exception as e:
        print(f"Error loading padding for {year}: {e}")
        return None


def calculate_ehf_for_year(year, input_dir, output_dir, t95_threshold):
    """
    Calculates EHF for a specific year, handling the time boundaries.
    """
    output_file = output_dir / f"ehf_{year}.nc"
    if output_file.exists():
        return f"Skipped {year} (Exists)"

    # 1. Load Data with Padding
    t_mean = load_padded_data(year, input_dir)

    if t_mean is None:
        return f"Failed {year}: Could not load data"

    # 2. Calculate Rolling Windows
    # Nairn & Fawcett use a 3-day average.
    # Convention: Is 'Day T' the start, middle, or end of the 3 days?
    # Usually: (T_i + T_i-1 + T_i-2)/3. We use center=False (default looks back).
    t_3day = t_mean.rolling(time=3, center=False).mean()

    # Acclimatization: Average of previous 30 days
    # We assume 'previous' means relative to the current day T
    # shift(1) ensures we don't include 'today' in the 'previous' history if desired
    # Standard EHF often uses: (T_3day) - (T_30day_lagged)
    t_30day = t_mean.rolling(time=30, center=False).mean()

    # 3. Calculate EHI Components

    # EHI_sig = T_3day - T_95
    # T95 must be aligned? Usually T95 is a single map (lat/lon).
    ehi_sig = t_3day - t95_threshold

    # EHI_accl = T_3day - T_30day
    ehi_accl = t_3day - t_30day

    # 4. Calculate EHF
    # Formula: EHI_sig * max(1, EHI_accl)
    # We use .clip(min=1) for the max(1, ...) part
    ehf_index = ehi_sig * ehi_accl.clip(min=1)

    # Negative EHF is treated as 0 (No heatwave)
    # Some definitions keep negatives to show "Cool waves", but typically:
    ehf_index = ehf_index.where(ehf_index > 0, 0)

    # 5. Trim back to the Target Year
    # We remove the Dec-Prev and Jan-Next padding
    ehf_final = ehf_index.sel(time=str(year))

    # 6. Save
    ehf_ds = ehf_final.to_dataset(name="ehf")
    encoding = {"ehf": {"zlib": True, "complevel": 5}}
    ehf_ds.to_netcdf(output_file, encoding=encoding)

    return f"Processed EHF {year}"


def main():
    # 1. Load the T_MEAN 95th Percentile Threshold
    # You need a quantile file specifically for T_MEAN, not just T_MIN/T_MAX
    # Assuming you have one named 'daily_t_mean_quantiles...'
    print("Loading Climatology...")

    quantile_val = Vars.quantiles[0]  # 0.95
    q_str = "_".join([str(int(100 * q)) for q in Vars.quantiles])

    # Note: I adjusted the filename to look for 't_mean'
    clim_file = (
        Dirs.dir_era_quantiles
        / f"daily_t_mean_quantiles_{q_str}_{Vars.year_reference_start}-{Vars.year_reference_end}.nc"
    )

    if not clim_file.exists():
        print(f"❌ CRITICAL ERROR: Climatology file not found:\n   {clim_file}")
        print("   EHF requires the 95th percentile of T_MEAN.")
        print(
            "   Please generate this file or adjust the code to approximate it from t_min/t_max."
        )
        return

    with xr.open_dataset(clim_file) as ds:
        t95_threshold = (
            ds.sel(quantile=quantile_val, method="nearest").to_array().squeeze().load()
        )

    # 2. Output Directory
    output_dir = Dirs.dir_results / "ehf"
    output_dir.mkdir(parents=True, exist_ok=True)

    years = Vars.get_analysis_years()

    # 3. Parallel Execution
    results = Parallel(n_jobs=1, verbose=10)(
        delayed(calculate_ehf_for_year)(
            year, Dirs.dir_era_daily, output_dir, t95_threshold
        )
        for year in years
    )

    print("\n".join(results))


if __name__ == "__main__":
    main()
