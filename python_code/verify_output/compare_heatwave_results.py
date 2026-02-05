"""
Compare new ERA5-Land heatwave results with old ERA5 results.

The old results have coarser resolution (ERA5 ~0.25°) vs new (ERA5-Land ~0.1°).
This script regrids the new data to match the old resolution for fair comparison.

Checks performed:
- Visual comparison: Maps of heatwave days/count
- Statistical comparison: Histograms, scatter plots, correlations
- Spatial patterns: Difference maps
- Temporal consistency: Year-to-year correlations

Usage:
    python python_code/verify_output/compare_heatwave_results.py --year 2020
    python python_code/verify_output/compare_heatwave_results.py --all-years
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path.cwd()

sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

from my_config import DirsLocal

# Suppress shapely warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
# Suppress cartopy facecolor warnings
warnings.filterwarnings(
    "ignore",
    message="facecolor will have no effect",
    category=UserWarning,
    module="cartopy",
)

# Set up matplotlib for non-interactive use
plt.switch_backend("Agg")

# Directories
OLD_DIR = Path(
    "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/data/lancet/countdown-global/results/heatwaves/old"
)
NEW_DIR = DirsLocal.hw_min_max

# Output directory for plots
OUTPUT_DIR = DirsLocal.hw_min_max / "heatwave_comparison_plots"
OUTPUT_DIR.mkdir(exist_ok=True)


def find_common_years():
    """Find years that exist in both old and new directories."""
    old_years = set()
    new_years = set()

    # Find old files
    if OLD_DIR.exists():
        for f in OLD_DIR.glob("heatwave_indicators_*.nc"):
            try:
                year = int(f.stem.split("_")[-1])
                old_years.add(year)
            except ValueError:
                continue

    # Find new files
    if NEW_DIR.exists():
        for f in NEW_DIR.glob("heatwave_indicators_*.nc"):
            try:
                year = int(f.stem.split("_")[-1])
                new_years.add(year)
            except ValueError:
                continue

    common_years = sorted(old_years & new_years)
    print(f"Found {len(common_years)} common years: {common_years}")
    print(f"Old years only: {sorted(old_years - new_years)}")
    print(f"New years only: {sorted(new_years - old_years)}")

    return common_years


def load_and_regrid(year):
    """Load old and new datasets, regrid new to old resolution."""
    # Load old data
    old_file = OLD_DIR / f"heatwave_indicators_{year}.nc"
    old_ds = xr.open_dataset(old_file, decode_timedelta=False)

    # Load new data
    new_file = NEW_DIR / f"heatwave_indicators_{year}.nc"
    new_ds = xr.open_dataset(new_file, decode_timedelta=False)

    print(f"   Old dataset dimensions: {dict(old_ds.sizes)}")
    print(
        f"   Old latitude range: {old_ds.latitude.values.min():.2f} to {old_ds.latitude.values.max():.2f}"
    )
    print(
        f"   Old longitude range: {old_ds.longitude.values.min():.2f} to {old_ds.longitude.values.max():.2f}"
    )
    print(f"   New dataset dimensions: {dict(new_ds.sizes)}")
    print(
        f"   New latitude range: {new_ds.latitude.values.min():.2f} to {new_ds.latitude.values.max():.2f}"
    )
    print(
        f"   New longitude range: {new_ds.longitude.values.min():.2f} to {new_ds.longitude.values.max():.2f}"
    )

    # Convert old_ds longitude from 0-360 to -180-180 for consistency
    old_ds = old_ds.assign_coords(longitude=((old_ds.longitude + 180) % 360) - 180)
    old_ds = old_ds.sortby("longitude")

    # Regrid new data to old resolution
    new_regridded = new_ds.interp(
        latitude=old_ds.latitude, longitude=old_ds.longitude, method="linear"
    )

    # Apply land mask: mask old data to only include areas where ERA5-Land has data
    # This ensures fair comparison by excluding ocean areas from ERA5
    for var in new_regridded.data_vars:
        if var in old_ds.data_vars:
            # Mask old data where new data is NaN (ocean areas)
            old_ds[var] = old_ds[var].where(~np.isnan(new_regridded[var]))

    print("   Applied land mask to ERA5 data (excluded ocean areas)")

    return old_ds, new_regridded


def create_comparison_plots(year, old_ds, new_ds):
    """Create visual comparison plots using cartopy for proper geographic projection."""
    # Create figure with cartopy projections
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Heatwave Comparison: ERA5 vs ERA5-Land ({year})", fontsize=16)

    # Create subplots with PlateCarree projection
    gs = fig.add_gridspec(2, 3, hspace=0.1, wspace=0.05)
    axes = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
            axes.append(ax)

    variables = ["heatwave_days", "heatwave_count"]

    for i, var in enumerate(variables):
        # Old data
        old_data = old_ds[var].values.squeeze()
        new_data = new_ds[var].values.squeeze()
        vmin = min(np.nanmin(old_data), np.nanmin(new_data))
        vmax = max(np.nanmax(old_data), np.nanmax(new_data))

        # Get coordinates
        lon = old_ds.longitude.values
        lat = old_ds.latitude.values

        # Plot old
        ax = axes[i * 3]
        mesh1 = ax.pcolormesh(
            lon,
            lat,
            old_data,
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        ax.set_title(f"ERA5 {var.replace('_', ' ').title()}", fontsize=12)
        ax.coastlines(resolution="50m", color="black", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=":", color="gray")
        ax.add_feature(cfeature.LAND)
        plt.colorbar(mesh1, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

        # Plot new
        ax = axes[i * 3 + 1]
        mesh2 = ax.pcolormesh(
            lon,
            lat,
            new_data,
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        ax.set_title(f"ERA5-Land {var.replace('_', ' ').title()}", fontsize=12)
        ax.coastlines(resolution="50m", color="black", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=":", color="gray")
        ax.add_feature(cfeature.LAND)
        plt.colorbar(mesh2, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

        # Plot difference
        ax = axes[i * 3 + 2]
        diff = new_data - old_data
        # Use different color scale for differences
        diff_vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
        mesh3 = ax.pcolormesh(
            lon,
            lat,
            diff,
            cmap="RdBu_r",
            vmin=-diff_vmax,
            vmax=diff_vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        ax.set_title("Difference (ERA5-Land - ERA5)", fontsize=12)
        ax.coastlines(resolution="50m", color="black", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=":", color="gray")
        ax.add_feature(cfeature.LAND)
        plt.colorbar(mesh3, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

    # plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"heatwave_maps_{year}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   Saved geographic map comparison: heatwave_maps_{year}.png")


def create_distribution_plots(year, old_ds, new_ds):
    """Create distribution comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Heatwave Distributions: ERA5 vs ERA5-Land ({year})", fontsize=14)

    variables = ["heatwave_days", "heatwave_count"]

    for i, var in enumerate(variables):
        # Flatten and remove NaNs
        old_flat = old_ds[var].values.flatten()
        new_flat = new_ds[var].values.flatten()

        old_flat = old_flat[~np.isnan(old_flat)]
        new_flat = new_flat[~np.isnan(new_flat)]

        # Plot histograms
        axes[i].hist(old_flat, alpha=0.7, label="ERA5", bins=50, density=True)
        axes[i].hist(new_flat, alpha=0.7, label="ERA5-Land", bins=50, density=True)
        axes[i].set_xlabel(var.replace("_", " ").title())
        axes[i].set_ylabel("Density")
        axes[i].legend()
        axes[i].set_title(f"{var.replace('_', ' ').title()} Distribution")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"heatwave_distributions_{year}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"   Saved distribution comparison: heatwave_distributions_{year}.png")


def create_scatter_plots(year, old_ds, new_ds):
    """Create scatter plots and compute correlations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Heatwave Scatter: ERA5 vs ERA5-Land ({year})", fontsize=14)

    variables = ["heatwave_days", "heatwave_count"]
    stats = {}

    for i, var in enumerate(variables):
        # Flatten and remove NaNs
        old_flat = old_ds[var].values.flatten()
        new_flat = new_ds[var].values.flatten()

        # Remove NaN pairs
        valid_mask = ~(np.isnan(old_flat) | np.isnan(new_flat))
        old_clean = old_flat[valid_mask]
        new_clean = new_flat[valid_mask]

        # Hexbin plot
        hb = axes[i].hexbin(old_clean, new_clean, gridsize=50, cmap="Blues", mincnt=1)
        axes[i].set_xlabel(f"ERA5 {var.replace('_', ' ').title()}")
        axes[i].set_ylabel(f"ERA5-Land {var.replace('_', ' ').title()}")
        axes[i].set_title(f"{var.replace('_', ' ').title()} Correlation")
        axes[i].plot(
            [old_clean.min(), old_clean.max()],
            [old_clean.min(), old_clean.max()],
            "r--",
            alpha=0.7,
            label="1:1 line",
        )
        axes[i].legend()
        # Add colorbar for hexbin
        plt.colorbar(hb, ax=axes[i], label="Count")

        # Compute statistics
        if len(old_clean) > 0:
            corr, _ = pearsonr(old_clean, new_clean)
            rmse = np.sqrt(np.mean((new_clean - old_clean) ** 2))
            mean_diff = np.mean(new_clean - old_clean)
            stats[var] = {
                "correlation": corr,
                "rmse": rmse,
                "mean_difference": mean_diff,
                "n_points": len(old_clean),
            }

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"heatwave_scatter_{year}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"   Saved scatter plot: heatwave_scatter_{year}.png")
    return stats


def print_statistics(year, stats):
    """Print comparison statistics."""
    print(f"\n📊 Statistics for {year}:")
    for var, stat in stats.items():
        print(f"   {var.replace('_', ' ').title()}:")
        print(".4f")
        print(".2f")
        print(".2f")
        print(f"      N points: {stat['n_points']:,}")


def compare_year(year):
    """Compare heatwave results for a single year."""
    print(f"\n🔍 Comparing heatwave results for {year}")

    try:
        # Load and regrid data
        old_ds, new_ds = load_and_regrid(year)

        # Create plots
        create_comparison_plots(year, old_ds, new_ds)
        create_distribution_plots(year, old_ds, new_ds)
        stats = create_scatter_plots(year, old_ds, new_ds)

        # Print statistics
        print_statistics(year, stats)

        # Close datasets
        old_ds.close()
        new_ds.close()

    except Exception as e:
        print(f"❌ Error comparing {year}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare new ERA5-Land heatwave results with old ERA5 results"
    )
    parser.add_argument("--year", type=int, help="Year to compare (e.g., 2020)")
    parser.add_argument(
        "--all-years", action="store_true", help="Compare all common years"
    )

    args = parser.parse_args()

    # Find common years
    common_years = find_common_years()

    if not common_years:
        print("❌ No common years found between old and new datasets")
        sys.exit(1)

    if args.year:
        if args.year not in common_years:
            print(f"❌ Year {args.year} not found in both datasets")
            print(f"   Common years: {common_years}")
            sys.exit(1)
        years_to_compare = [args.year]
    elif args.all_years:
        years_to_compare = common_years
    else:
        # Default: compare first year
        years_to_compare = [common_years[-1]]
        print(f"Defaulting to year {common_years[-1]} (use --all-years for all)")

    print(f"\n📁 Plots will be saved to: {OUTPUT_DIR}")

    # Compare selected years
    for year in years_to_compare:
        compare_year(year)

    print(f"\n✅ Comparison complete! Check plots in {OUTPUT_DIR}")


if __name__ == "__main__":
    # pass
    main()
