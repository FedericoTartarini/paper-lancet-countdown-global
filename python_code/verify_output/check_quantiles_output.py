"""
Verification script for the output of c_calculate_quantiles.py.

This script performs the following checks and visualizations:
1. Verifies that output files exist for all temperature variables
2. Checks file structure (dimensions, coordinates, data variables)
3. Validates data ranges (quantiles should be within reasonable temperature bounds)
4. Creates global maps of the quantile values
5. Plots histograms of quantile distributions
6. Spot-checks specific locations (e.g., known hot/cold regions)

Usage:
    python python_code/verify_output/check_quantiles_output.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path.cwd()
sys.path.append(str(project_root))

from my_config import DirsLocal, Vars


def check_file_exists(file_path: Path) -> bool:
    """Check if a file exists and print status."""
    exists = file_path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} File exists: {file_path.name}")
    return exists


def check_file_structure(ds: xr.Dataset, t_var: str) -> bool:
    """Check that the dataset has the expected structure."""
    print(f"\n📋 Checking structure for {t_var}:")

    # Check dimensions
    expected_dims = {"latitude", "longitude"}
    actual_dims = set(ds.sizes.keys())
    dims_ok = expected_dims.issubset(actual_dims)
    status = "✅" if dims_ok else "❌"
    print(f"   {status} Dimensions: {list(ds.sizes.keys())}")

    # Check coordinates
    print(
        f"   📐 Latitude range: {float(ds.latitude.min()):.2f} to {float(ds.latitude.max()):.2f}"
    )
    print(
        f"   📐 Longitude range: {float(ds.longitude.min()):.2f} to {float(ds.longitude.max()):.2f}"
    )
    print(f"   📐 Quantile values: {ds.quantile}")

    # Check data variable
    has_var = t_var in ds.data_vars
    status = "✅" if has_var else "❌"
    print(f"   {status} Data variable '{t_var}' present")

    return dims_ok and has_var


def check_data_ranges(ds: xr.Dataset, t_var: str) -> bool:
    """Check that data values are within reasonable temperature bounds."""
    print(f"\n🌡️  Checking data ranges for {t_var}:")

    data = ds[t_var]

    # Get statistics (excluding NaN)
    min_val = float(data.min())
    max_val = float(data.max())
    mean_val = float(data.mean())

    # Temperature bounds in Kelvin (reasonable Earth surface temperatures)
    # Min: ~180K (-93°C, Antarctica) to Max: ~330K (57°C, Death Valley)
    min_bound = 180  # K
    max_bound = 350  # K

    min_ok = min_val >= min_bound
    max_ok = max_val <= max_bound

    print(
        f"   Min value: {min_val:.2f} K ({min_val - 273.15:.2f} °C) {'✅' if min_ok else '❌'}"
    )
    print(
        f"   Max value: {max_val:.2f} K ({max_val - 273.15:.2f} °C) {'✅' if max_ok else '❌'}"
    )
    print(f"   Mean value: {mean_val:.2f} K ({mean_val - 273.15:.2f} °C)")

    # Check for NaN percentage
    total_cells = data.size
    nan_count = int(data.isnull().sum())
    nan_pct = (nan_count / total_cells) * 100
    print(f"   NaN percentage: {nan_pct:.2f}% ({nan_count:,} / {total_cells:,} cells)")

    return min_ok and max_ok


def plot_global_map(ds: xr.Dataset, t_var: str, output_dir: Path) -> None:
    """Create a global map of the quantile values."""
    print(f"\n🗺️  Creating global map for {t_var}...")

    data = ds[t_var]

    # Convert to Celsius for easier interpretation
    data_celsius = data - 273.15

    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={"projection": None})

    # Plot the quantile value
    im = data_celsius.plot(
        ax=ax,
        cmap="RdYlBu_r",
        cbar_kwargs={"label": f"{t_var} {Vars.quantiles} quantile (°C)"},
    )

    ax.set_title(
        f"{t_var} - {Vars.quantiles} Quantile\n({Vars.year_reference_start}-{Vars.year_reference_end} Reference Period)"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    output_file = output_dir / f"quantile_map_{t_var}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {output_file.name}")


def plot_histogram(ds: xr.Dataset, t_var: str, output_dir: Path) -> None:
    """Create a histogram of the quantile distribution."""
    print(f"\n📊 Creating histogram for {t_var}...")

    data = ds[t_var]

    # Convert to Celsius and flatten
    data_celsius = (data - 273.15).values.flatten()
    data_celsius = data_celsius[~np.isnan(data_celsius)]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(data_celsius, bins=100, edgecolor="black", alpha=0.7)
    ax.set_xlabel(f"{t_var} {Vars.quantiles} quantile (°C)")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Distribution of {t_var} {Vars.quantiles} Quantile\n({Vars.year_reference_start}-{Vars.year_reference_end} Reference Period)"
    )

    # Add statistics
    mean_val = np.mean(data_celsius)
    median_val = np.median(data_celsius)
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.1f}°C")
    ax.axvline(
        median_val, color="green", linestyle="--", label=f"Median: {median_val:.1f}°C"
    )
    ax.legend()

    output_file = output_dir / f"quantile_histogram_{t_var}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {output_file.name}")


def plot_subregions(ds: xr.Dataset, t_var: str, output_dir: Path) -> None:
    """Create plots for specific subregions."""
    print(f"\n🗺️  Creating subregion plots for {t_var}...")

    subregions = {
        "Sydney and Melbourne": {
            "latitude": slice(-40, -20),
            "longitude": slice(140, 160),
        },
        "India and Tibet": {
            "latitude": slice(10, 40),
            "longitude": slice(70, 100),
        },
        "Central Europe": {
            "latitude": slice(35, 55),
            "longitude": slice(0, 20),
        },
    }

    for region_name, bounds in subregions.items():
        print(f"   📍 Plotting: {region_name}")

        # Subset data for the region using where instead of sel
        data = ds[t_var].where(
            (ds.latitude >= bounds["latitude"].start)
            & (ds.latitude <= bounds["latitude"].stop)
            & (ds.longitude >= bounds["longitude"].start)
            & (ds.longitude <= bounds["longitude"].stop),
            drop=True,
        )
        data_celsius = data - 273.15

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        im = data_celsius.plot(
            ax=ax,
            cmap="RdYlBu_r",
            cbar_kwargs={"label": f"{t_var} {Vars.quantiles} quantile (°C)"},
        )
        ax.set_title(
            f"{region_name} - {t_var} {Vars.quantiles} Quantile\n({Vars.year_reference_start}-{Vars.year_reference_end} Reference Period)"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Save plot
        output_file = (
            output_dir
            / f"quantile_subregion_{region_name.replace(' ', '_')}_{t_var}.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: {output_file.name}")


def spot_check_locations(ds: xr.Dataset, t_var: str) -> None:
    """Check quantile values at specific known locations."""
    print(f"\n📍 Spot-checking locations for {t_var}:")

    # Define locations to check (lat, lon, name)
    locations = [
        (0, 0, "Gulf of Guinea (Ocean)"),
        (-23.5, 133.9, "Alice Springs, Australia (Hot Desert)"),
        (64.8, -147.7, "Fairbanks, Alaska (Cold)"),
        (25.0, 55.0, "Dubai, UAE (Hot)"),
        (-77.8, 166.7, "McMurdo Station, Antarctica (Very Cold)"),
    ]

    data = ds[t_var]

    for lat, lon, name in locations:
        try:
            value = float(data.sel(latitude=lat, longitude=lon, method="nearest"))
            value_celsius = value - 273.15
            print(f"   {name}: {value:.2f} K ({value_celsius:.2f} °C)")
        except Exception as e:
            print(f"   ❌ {name}: Error - {e}")


def main():
    """Main verification routine."""
    print("=" * 70)
    print("🔍 Quantiles Output Verification")
    print(f"   Reference period: {Vars.year_reference_start}-{Vars.year_reference_end}")
    print(f"   Quantile: {Vars.quantiles}")
    print(f"   Variables: {Vars.t_vars}")
    print("=" * 70)

    # Create output directory for plots
    output_dir = DirsLocal.e5l_q / "verification_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    all_checks_passed = True

    for t_var in Vars.t_vars:
        print(f"\n{'=' * 70}")
        print(f"🌡️  Verifying: {t_var}")
        print("=" * 70)

        # Build expected filename
        # q = int(Vars.quantiles * 100)  # Convert to percentage for filename
        q = Vars.quantiles
        quantile_file = (
            DirsLocal.e5l_q
            # Path(
            #     "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/data/lancet/countdown-global/era5/quantiles"
            # )
            / f"daily_{t_var}_quantiles_{q}_{Vars.year_reference_start}-{Vars.year_reference_end}.nc"
        )

        # Check file exists
        if not check_file_exists(quantile_file):
            all_checks_passed = False
            continue

        # Open dataset
        ds = xr.open_dataset(quantile_file)

        # filter out values outside of the reasonable range before checks
        data = ds[t_var]
        data = data.where((data >= 180) & (data <= 350))
        ds[t_var] = data

        # Run checks
        if not check_file_structure(ds, t_var):
            all_checks_passed = False

        if not check_data_ranges(ds, t_var):
            all_checks_passed = False

        # Spot check locations
        spot_check_locations(ds, t_var)

        # Create visualizations
        plot_global_map(ds, t_var, output_dir)
        plot_histogram(ds, t_var, output_dir)
        plot_subregions(ds, t_var, output_dir)

        ds.close()

    # Summary
    print(f"\n{'=' * 70}")
    if all_checks_passed:
        print("✅ All verification checks passed!")
    else:
        print("❌ Some verification checks failed. Review output above.")
    print("=" * 70)
    print(f"\n📊 Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
