"""
How to interpret the Map
If the red dots are mostly in the Southern Hemisphere (Australia, Brazil, Southern Africa):

Confirmed: The error was the "End of Year" bug.
It's summer there in December, so heatwaves were getting cut off by the New Year.

If you see red dots in Europe/USA (Northern Hemisphere):

Unexpected: It is winter there in December. Unless it is a tropical region (near Equator), there shouldn't be heatwaves in Dec.
If you see this, the difference might be due to threshold precision (float vs double) rather than the date cutoff.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from my_config import Dirs  # Assuming these exist in your project
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- CONFIGURATION ---
# Adjust these variable names if they differ in your actual files
VAR_NAME_COUNT_OLD = "heatwave_count"
VAR_NAME_DAYS_OLD = "heatwaves_days"  # Note: old code named it 'heatwaves_days'
VAR_NAME_COUNT_NEW = "heatwave_count"
VAR_NAME_DAYS_NEW = "heatwave_days"


def load_old_data(year: float = 2000):
    """
    Loads old data which was split into two separate directories/files.
    """
    # Adjust filename pattern based on your previous code: "indicator_{year}.nc"
    dir_results_heatwaves = Dirs.dir_results / "heatwaves" / "results_2025"
    path_count = (
        dir_results_heatwaves / "heatwaves_counts_era5" / f"indicator_{year}.nc"
    )
    path_days = dir_results_heatwaves / "heatwaves_days_era5" / f"indicator_{year}.nc"

    if not path_count.exists() or not path_days.exists():
        print(f"⚠️ Missing old data for {year}")
        return None, None

    ds_count = xr.open_dataset(path_count)
    ds_days = xr.open_dataset(path_days)

    return ds_count[VAR_NAME_COUNT_OLD], ds_days[VAR_NAME_DAYS_OLD]


def load_new_data(year):
    """
    Loads new data which is combined in one file.
    """
    path_new = Dirs.dir_results_heatwaves / f"heatwave_indicators_{year}.nc"

    if not path_new.exists():
        print(f"⚠️ Missing new data for {year}")
        return None, None

    ds = xr.open_dataset(path_new)
    return ds[VAR_NAME_COUNT_NEW], ds[VAR_NAME_DAYS_NEW]


def plot_comparison(year, da_old, da_new, title_metric):
    """
    Plots Old vs New vs Difference side-by-side.
    """
    # Calculate Difference
    # Ensure coordinates match exactly (sometimes floating point errors in lat/lon prevent subtraction)
    # We use .values to force subtraction if shapes are identical, avoiding alignment issues
    if da_old.shape == da_new.shape:
        diff = da_new.values - da_old.values
        # Wrap back into DataArray for easy plotting
        da_diff = xr.DataArray(diff, coords=da_new.coords, dims=da_new.dims)
    else:
        print(f"❌ Shape mismatch for {year}: Old {da_old.shape} vs New {da_new.shape}")
        return

    # Check if they are identical
    is_identical = np.allclose(diff, 0, equal_nan=True)
    status_emoji = "✅ MATCH" if is_identical else "⚠️ DIFF"

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(7, 13), subplot_kw={"projection": None})

    # Common colorbar limits for the data (not the diff)
    vmin = min(da_old.min(), da_new.min()).item()
    vmax = max(da_old.max(), da_new.max()).item()

    # 1. Old
    da_old.plot(ax=axes[0], vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=True)
    axes[0].set_title(f"OLD: {title_metric}")

    # 2. New
    da_new.plot(ax=axes[1], vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=True)
    axes[1].set_title(f"NEW: {title_metric}")

    # 3. Difference (New - Old)
    # Use diverging colormap to see positive/negative shifts
    # Robust=True helps handle outliers if there are huge edge-case differences
    diff_plot = da_diff.plot(ax=axes[2], cmap="RdBu_r", robust=True, add_colorbar=True)
    axes[2].set_title(f"DIFF (New - Old)\n{status_emoji}")

    plt.suptitle(f"Comparison: {title_metric} ({year})", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Print stats if different
    if not is_identical:
        print(f"   Stats for {title_metric} Diff:")
        print(f"   Max Diff: {np.nanmax(diff)}")
        print(f"   Min Diff: {np.nanmin(diff)}")
        print(f"   Total Discrepancy (Sum): {np.nansum(np.abs(diff))}")


def check_december_bias(year: int = 2000):
    # 1. Load Data
    ds_old = load_old_data(year)[1]  # Get days
    ds_new = load_new_data(year)[1]  # Get days

    # 2. Calculate Difference
    diff = ds_new - ds_old

    # Filter: Only look at pixels where there is a Discrepancy
    diff_mask = diff > 0

    # 3. Analyze Geography of Errors
    # Get latitudes where errors occurred
    error_lats = diff.where(diff_mask, drop=True).latitude

    pct_south = (error_lats < 0).sum() / error_lats.count() * 100
    pct_north = (error_lats >= 0).sum() / error_lats.count() * 100

    print(f"--- Analysis of Discrepancies for {year} ---")
    print(f"Total Pixels with Errors: {diff_mask.sum().item()}")
    print(f"Errors in Southern Hemisphere: {pct_south.item():.1f}%")
    print(f"Errors in Northern Hemisphere: {pct_north.item():.1f}%")

    # 4. Analyze "Missing 3-Day Event" Signature
    # If the difference is exactly 3 days, it's highly likely a specific
    # short heatwave at the end of the year was missed.
    count_3_day_diff = (diff == 3).sum().item()
    print(
        f"Pixels with exactly +3 days diff (Missed minimal heatwaves): {count_3_day_diff}"
    )

    # 5. Plot the Map of Errors
    fig, ax = plt.subplots(
        figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Plot only the differences
    diff.where(diff > 0).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Reds",
        vmin=0,
        vmax=5,  # Focus on small counts
        cbar_kwargs={"label": "Difference in Days (New - Old)"},
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.set_title(
        f"Locations of Discrepancies ({year})\n(Expect these in Southern Hemisphere / Tropics)"
    )

    plt.show()


def main():
    check_december_bias(2003)

    # Pick a few sample years to test, or loop through all
    # years_to_test = Vars.get_analysis_years()
    years_to_test = [2003, 2020]  # Start with known hot years

    for year in years_to_test:
        print(f"\n--- Analyzing {year} ---")

        # 1. Load Data
        old_count, old_days = load_old_data(year)
        new_count, new_days = load_new_data(year)

        if old_count is None or new_count is None:
            continue

        # 2. Compare Counts
        plot_comparison(year, old_count, new_count, "Heatwave Events (Count)")

        # 3. Compare Days
        plot_comparison(year, old_days, new_days, "Heatwave Duration (Days)")


if __name__ == "__main__":
    main()
