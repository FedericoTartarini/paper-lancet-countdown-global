from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from my_config import DirsLocal


def plot_global_mean_temperature(ds):
    """
    Plot the global mean temperature (average over time) on a map.
    """
    print("\n--- 4. Global Mean Temperature Map ---")
    mean_temp = ds.t_mean.mean(dim="time") - 273.15  # Convert from K to °C

    plt.figure(figsize=(12, 6))
    mean_temp.plot(cmap="coolwarm", vmin=-50, vmax=50)
    plt.title("Global Mean 2m Temperature (°C) - 1980")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def plot_sydney_temperature(ds):
    """
    Extract Sydney data and plot monthly mean, min, max temperatures.
    """
    print("\n--- 5. Sydney Temperature Time Series ---")
    # Sydney coordinates: approx -33.8688, 151.2093
    sydney_lat = -33.8688
    sydney_lon = 151.2093

    # Find nearest point
    sydney_data = ds.sel(latitude=sydney_lat, longitude=sydney_lon, method="nearest")

    # Resample to monthly
    monthly_min = sydney_data.t_min.resample(time="1D").mean() - 273.15  # °C
    monthly_mean = sydney_data.t_mean.resample(time="1D").mean() - 273.15  # °C
    monthly_max = sydney_data.t_max.resample(time="1D").mean() - 273.15  # °C

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_min.time, monthly_min.values, label="Min", color="blue")
    plt.plot(monthly_mean.time, monthly_mean.values, label="Mean", color="green")
    plt.plot(monthly_max.time, monthly_max.values, label="Max", color="red")
    plt.title("Monthly Temperature in Sydney (°C)")
    plt.ylim(0, 42)
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()


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
            ds.close()
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

        # 4. Plot global mean
        plot_global_mean_temperature(ds)

        # 5. Plot Sydney
        plot_sydney_temperature(ds)

        ds.close()

    except Exception as e:
        print(f"❌ Error reading file: {e}")


if __name__ == "__main__":
    verify_file(file_path=DirsLocal.e5l_d / "2025_daily_summaries.nc")
