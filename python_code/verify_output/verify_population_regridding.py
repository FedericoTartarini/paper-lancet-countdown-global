import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from my_config import VarsWorldPop, DirsLocal

"""
Script to verify and plot the regridded WorldPop data from b_pop_merge_and_coarsen.py.

Compares with old files, plots maps, cumulative by latitude, and checks totals.
"""


def load_population_file(file_path):
    """Load a population NetCDF file."""
    ds = xr.open_dataset(file_path)
    # Assume single time, squeeze
    if "time" in ds.dims:
        ds = ds.isel(time=0)
    return ds["pop"]


def plot_global_maps(new_data, old_data, age_label, year, plot_dir):
    """Plot global maps for new and old data in subplots."""
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 12), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for i, (data, title) in enumerate(
        [
            (new_data, f"New {age_label} {year}"),
            (old_data, f"Old {age_label} {year}"),
        ]
    ):
        ax = axs[i]
        lon, lat = np.meshgrid(data.longitude, data.latitude)
        vmax = float(data.quantile(0.99).values)
        im = ax.pcolormesh(
            lon,
            lat,
            data.values,
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.5, label="Population")

    plt.tight_layout()
    save_path = plot_dir / f"maps_{age_label}_{year}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cumulative_by_lat(new_data, old_data, age_label, year, plot_dir):
    """Plot cumulative population by latitude for new and old data on the same plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # New data
    cum_pop_new = new_data.sum(dim="longitude").cumsum(dim="latitude")
    ax.plot(cum_pop_new.latitude, cum_pop_new.values, label="New", color="blue")

    # Old data
    cum_pop_old = old_data.sum(dim="longitude").cumsum(dim="latitude")
    ax.plot(cum_pop_old.latitude, cum_pop_old.values, label="Old", color="red")

    ax.set_xlabel("Latitude")
    ax.set_ylabel("Cumulative Population")
    ax.set_title(f"{age_label} {year} Cumulative by Lat")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    save_path = plot_dir / f"cum_lat_{age_label}_{year}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compare_sums(new_data, old_data, label):
    """Compare sums between new and old data."""
    new_sum = float(new_data.sum().values)
    old_sum = float(old_data.sum().values)
    diff = abs(new_sum - old_sum)
    rel_diff = diff / max(new_sum, old_sum) * 100
    print(
        f"{label}: New sum: {new_sum:,.0f}, Old sum: {old_sum:,.0f}, Diff: {diff:,.0f} ({rel_diff:.2f}%)"
    )
    return new_sum, old_sum


def main():
    # Paths
    new_dir = DirsLocal.pop_e5l_grid
    old_dir = Path(
        "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/data/lancet/countdown-global/results/worldpop_era5_grid"
    )

    # Create plot subfolder
    plot_dir = Path("compare_pop_data")
    plot_dir.mkdir(exist_ok=True)

    # Age groups
    age_groups = ["under_1", "65_over"]

    # Sample years for plotting
    sample_years = [2000, 2015]

    # Collect all new files
    new_files = list(new_dir.glob("*.nc"))
    print(f"Found {len(new_files)} new files.")

    # Load and compare
    for age_label in age_groups:
        print(f"\nProcessing {age_label}")
        for year in sample_years:
            new_file = new_dir / f"t_{age_label}_{year}_era5_regridded.nc"
            old_file = old_dir / f"t_{age_label}_{year}_era5_compatible.nc"

            if new_file.exists() and old_file.exists():
                new_data = load_population_file(new_file)
                old_data = load_population_file(old_file)

                # Check quantiles and outliers
                q95_new = new_data.quantile(0.95)
                q99_new = new_data.quantile(0.99)
                outliers_new = int((new_data > q99_new).sum().values)
                print(
                    f"New {age_label} {year}: 95th: {q95_new.values:,.0f}, 99th: {q99_new.values:,.0f}, Cells >99th: {outliers_new}"
                )

                q95_old = old_data.quantile(0.95)
                q99_old = old_data.quantile(0.99)
                outliers_old = int((old_data > q99_old).sum().values)
                print(
                    f"Old {age_label} {year}: 95th: {q95_old.values:,.0f}, 99th: {q99_old.values:,.0f}, Cells >99th: {outliers_old}"
                )

                # Compare sums
                compare_sums(new_data, old_data, f"{age_label} {year}")

                # Plot maps
                plot_global_maps(new_data, old_data, age_label, year, plot_dir)

                # Plot cumulative by lat
                plot_cumulative_by_lat(new_data, old_data, age_label, year, plot_dir)

            else:
                print(
                    f"Files not found for {age_label} {year}: New exists: {new_file.exists()}, Old exists: {old_file.exists()}"
                )

    # Check world totals for all new files
    print("\nWorld totals for new files:")
    for file_path in new_files:
        data = load_population_file(file_path)
        total = float(data.sum().values)
        print(f"{file_path.name}: {total:,.0f}")


if __name__ == "__main__":
    main()
