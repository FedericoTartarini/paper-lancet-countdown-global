"""
Calculate Heatwave Severity Categories (BoM Methodology).

Steps:
1. Generate Climatology Threshold (EHF_85):
   - Load EHF data for the reference period (e.g., 1986-2005).
   - Filter for POSITIVE EHF values only (EHF > 0).
   - Calculate the 85th percentile of these positive values per grid cell.

2. Classify Daily Severity:
   - Low-Intensity: 0 < EHF <= EHF_85
   - Severe:        EHF_85 < EHF <= 3 * EHF_85
   - Extreme:       EHF > 3 * EHF_85
"""

import xarray as xr
import numpy as np
from joblib import Parallel, delayed
from my_config import Vars, Dirs

# Define Output Codes
CAT_NONE = 0
CAT_LOW = 1
CAT_SEVERE = 2
CAT_EXTREME = 3


def calculate_severity_threshold(reference_years):
    """
    Calculates the EHF_85 threshold map from the baseline period.
    """
    output_file = Dirs.dir_era_quantiles / "ehf_severity_threshold_85.nc"

    if output_file.exists():
        print("Loading existing severity threshold...")
        return xr.open_dataset(output_file)["ehf_85"]

    print(
        f"Calculating EHF_85 Threshold from baseline: {reference_years.start}-{reference_years.stop - 1}..."
    )

    # 1. Load all EHF files for the reference period
    ehf_files = [
        Dirs.dir_results / "ehf" / f"ehf_{year}.nc" for year in reference_years
    ]

    # Verify files exist
    ehf_files = [f for f in ehf_files if f.exists()]
    if not ehf_files:
        raise FileNotFoundError("No EHF files found for the reference period.")

    # 2. Open Dataset (Lazy loading)
    ds = xr.open_mfdataset(ehf_files, combine="by_coords", parallel=True)
    ehf = ds["ehf"]

    # 3. Filter for Positive Values Only
    # We only care about days that were actually heatwaves (EHF > 0)
    # where() replaces False with NaN.
    ehf_positive = ehf.where(ehf > 0)

    # 4. Calculate 85th Percentile
    # This reduces the Time dimension, leaving a Lat/Lon map
    # skipna=True is default, which is what we want.
    ehf_85 = ehf_positive.quantile(0.85, dim="time", skipna=True)

    # Handle pixels that NEVER had a heatwave in the baseline (NaNs)
    # We set their threshold to infinity so they can't trigger severe/extreme?
    # Or set to 0? BoM doesn't strictly specify, but Infinity is safer to avoid false positives.
    # For now, we leave as NaN or fill with a high value.
    ehf_85 = ehf_85.fillna(99999)

    ehf_85.name = "ehf_85"

    # Save
    ehf_85.to_netcdf(output_file)
    print(f"Saved threshold to {output_file}")


def classify_year(year, ehf_85_path, output_dir):
    """
    Classifies EHF into severity categories 0, 1, 2, 3.
    """
    input_file = Dirs.dir_results / "ehf" / f"ehf_{year}.nc"
    output_file = output_dir / f"ehf_severity_{year}.nc"

    if output_file.exists():
        return f"Skipped {year}"

    if not input_file.exists():
        return f"Missing EHF input for {year}"

    # Load Data
    ehf = xr.open_dataset(input_file)["ehf"]
    ehf_85 = xr.open_dataset(ehf_85_path)["ehf_85"]

    # Initialize Output Array (Int8 to save space)
    # Start with 0 (None)
    severity = xr.zeros_like(ehf, dtype=np.int8)
    severity.name = "severity"

    # --- Apply Classification Logic ---

    # 1. Low Intensity: EHF > 0
    # We start by setting everything > 0 to CAT_LOW (1)
    severity = severity.where(ehf <= 0, CAT_LOW)

    # 2. Severe: EHF > EHF_85
    # Overwrite 1s with 2s where condition meets
    severity = severity.where(ehf <= ehf_85, CAT_SEVERE)

    # 3. Extreme: EHF > 3 * EHF_85
    # Overwrite 2s with 3s where condition meets
    severity = severity.where(ehf <= (3 * ehf_85), CAT_EXTREME)

    # Re-enforce 0 for non-heatwaves (in case NaNs messed up logic)
    severity = severity.where(ehf > 0, CAT_NONE)

    # Save
    encoding = {"severity": {"zlib": True, "complevel": 5, "dtype": "int8"}}
    severity.to_netcdf(output_file, encoding=encoding)

    return f"Classified {year}"


def main():
    # 1. Calculate or Load the Threshold (The "Ruler")
    # Uses the reference period defined in your config
    ref_years = range(Vars.year_reference_start, Vars.year_reference_end + 1)

    # This function calculates the threshold map and saves it
    # We just need the path to pass to workers
    calculate_severity_threshold(reference_years=ref_years)
    threshold_path = Dirs.dir_era_quantiles / "ehf_severity_threshold_85.nc"

    # 2. Prepare Output Directory
    output_dir = Dirs.dir_results / "ehf_severity"
    output_dir.mkdir(parents=True, exist_ok=True)

    years = Vars.get_analysis_years()

    # 3. Parallel Classification
    # We pass the PATH to the threshold file, not the object,
    # to avoid pickling large arrays across processes.
    results = Parallel(n_jobs=6, verbose=10)(
        delayed(classify_year)(year, threshold_path, output_dir) for year in years
    )

    print("\n".join(results))


if __name__ == "__main__":
    pass
    # main()
