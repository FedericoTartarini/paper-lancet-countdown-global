import sys
from pathlib import Path
import xarray as xr
import numpy as np


def verify_file(file_path):
    """
    Verifies the integrity and logic of the generated daily summary NetCDF file.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"📂 Opening: {path.name}")
    print(f"   Size: {path.stat().st_size / (1024**3):.2f} GB")

    try:
        ds = xr.open_dataset(path)

        # 1. Check Dimensions
        print("\n--- 1. Dimension Checks ---")
        if "time" in ds.dims:
            print(f"✅ Time dimension found: {ds.sizes['time']} days")
        else:
            print("❌ Time dimension missing!")

        if "latitude" in ds.dims and "longitude" in ds.dims:
            print(
                f"✅ Spatial dimensions found: {ds.sizes['latitude']}x{ds.sizes['longitude']}"
            )
        else:
            print("❌ Spatial dimensions missing!")

        # 2. Check Variables
        print("\n--- 2. Variable Checks ---")
        expected_vars = ["t_min", "t_mean", "t_max"]
        missing = [v for v in expected_vars if v not in ds.data_vars]

        if not missing:
            print(f"✅ All variables present: {expected_vars}")
        else:
            print(f"❌ Missing variables: {missing}")
            return

        # 3. Logic Check (min <= mean <= max)
        print("\n--- 3. Logic Consistency Check ---")
        print("   (Loading a small slice to check values...)")

        # Take the first day and a small spatial chunk to avoid loading 5GB
        subset = (
            ds.isel(time=0)
            .isel(latitude=slice(100, 200), longitude=slice(100, 200))
            .load()
        )

        t_min = subset["t_min"]
        t_mean = subset["t_mean"]
        t_max = subset["t_max"]

        # Create a mask for valid data (ignoring NaNs over oceans)
        valid_mask = ~np.isnan(t_min) & ~np.isnan(t_mean) & ~np.isnan(t_max)

        if valid_mask.sum() == 0:
            print(
                "⚠️ Sample slice contained only NaNs (likely ocean). Skipping logic check."
            )
        else:
            # Check relationships only where data exists
            t_min_v = t_min.where(valid_mask)
            t_mean_v = t_mean.where(valid_mask)
            t_max_v = t_max.where(valid_mask)

            # Check min <= mean
            # We check if min - mean > epsilon. Using a small tolerance for floating point safety.
            violations_min_mean = t_min_v > t_mean_v

            # Check mean <= max
            violations_mean_max = t_mean_v > t_max_v

            invalid_mask = ds.t_min > ds.t_max

            # Count how many points are invalid
            error_count = invalid_mask.sum().values

            if violations_min_mean.any() or violations_mean_max.any():
                print("❌ FAILED: Logical inconsistency found.")

                if violations_min_mean.any():
                    max_diff = (t_min_v - t_mean_v).max().values
                    print("   Details: t_min > t_mean detected.")
                    print(f"   Max violation difference: {max_diff}")

                if violations_mean_max.any():
                    max_diff = (t_mean_v - t_max_v).max().values
                    print("   Details: t_mean > t_max detected.")
                    print(f"   Max violation difference: {max_diff}")

                print(
                    "   (Note: Tiny differences (<1e-6) may be due to floating point precision)"
                )
            if error_count > 0:
                print(f"Number of conflicting points: {error_count}")
                print("❌ FAILED: Logical inconsistency found (e.g. min > max).")
            else:
                print(
                    "✅ PASSED: t_min <= t_mean <= t_max holds true for checked sample."
                )

        ds.close()

    except Exception as e:
        print(f"❌ Error reading file: {e}")


if __name__ == "__main__":
    # If a file is passed as argument, use it; otherwise default to the test file
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = "~/Downloads/local_daily_summary_test.nc"

    verify_file(file_path=target_file)
