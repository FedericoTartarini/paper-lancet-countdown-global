"""
Analyze and plot population trends for renamed ERA5-compatible total files.

This script looks for files named like:
  t_under_1_2021_era5_compatible.nc
  t_65_over_2022_era5_compatible.nc

It aggregates global totals and produces diagnostic plots and Hovmöller (latitude vs year)
visualizations to help detect discontinuities between years.
"""

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import re
import numpy as np

# Geo plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import box

# Assuming your config is accessible
from my_config import DirsLocal, Vars, FilesLocal


def save_fig(fig, name: str):
    """Save figure to the interim directory with a consistent name."""
    out_dir = Path(DirsLocal.figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False)
    fig.savefig(out_path, bbox_inches="tight")


# --- ADVANCED PLOT: Zonal Statistics ---
def plot_zonal_hovmoller(directory, age_label="under_1"):
    """
    Plots Latitude vs Time (Hovmöller) for a given filename age label.
    """
    directory = Path(directory)
    age_str = age_label

    files = sorted(list(directory.glob(f"t_{age_str}_*_era5_compatible.nc")))

    if not files:
        print("No files for Hovmoller plot.")
        return

    datasets = []

    print("Building Zonal Means (Hovmöller)...")
    for f in tqdm(files):
        m = re.search(r"(19|20)\d{2}", f.name)
        if not m:
            continue
        year = int(m.group(0))

        try:
            ds = xr.open_dataset(f)
            if "pop" not in ds:
                print(f"  'pop' variable not in {f.name}; skipping")
                continue

            pop = ds["pop"]
            dims = list(pop.dims)

            # Detect latitude and longitude dimension names
            lat_dim = next(
                (d for d in dims if "lat" in str(d).lower() or d in ("latitude", "y")),
                None,
            )
            lon_dim = next(
                (d for d in dims if "lon" in str(d).lower() or d in ("longitude", "x")),
                None,
            )

            # If longitude detected, sum over it first to get zonal totals
            if lon_dim is not None and lon_dim in pop.dims:
                zonal = pop.sum(dim=lon_dim)
            else:
                zonal = pop

            # Determine the latitude dim if not found earlier
            if lat_dim is None:
                lat_dim = next(
                    (
                        d
                        for d in zonal.dims
                        if "lat" in str(d).lower() or d in ("latitude", "y")
                    ),
                    None,
                )

            # Sum over any remaining dims except the latitude dim
            sum_dims = [d for d in zonal.dims if d != lat_dim]
            if sum_dims:
                zonal = zonal.sum(dim=sum_dims)

            # Rename latitude dim to a consistent name
            if lat_dim is not None and lat_dim != "latitude":
                try:
                    zonal = zonal.rename({lat_dim: "latitude"})
                except Exception:
                    pass

            # Final squeeze and ensure 1D over latitude
            zonal = zonal.squeeze()
            if zonal.ndim != 1:
                other_dims = [d for d in zonal.dims if d != "latitude"]
                if other_dims:
                    zonal = zonal.sum(dim=other_dims)

            if "latitude" not in zonal.dims:
                raise ValueError(
                    f"Could not derive latitude dimension for file {f.name}; dims={zonal.dims}"
                )

            # Assign coords consistently and add year dim
            zonal = zonal.assign_coords({"latitude": zonal["latitude"]})
            zonal_sum = zonal.expand_dims(year=[year])
            datasets.append(zonal_sum)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not datasets:
        print("No valid datasets to concatenate for Hovmoller.")
        return

    # Concat along the explicit 'year' dim; use join='outer' to handle slight coord differences
    combined = xr.concat(datasets, dim="year", join="outer").sortby("year")

    # If combined is a Dataset (unlikely), pick 'pop' or convert to DataArray
    if isinstance(combined, xr.Dataset):
        if "pop" in combined.data_vars:
            combined = combined["pop"]
        else:
            # collapse variables into one DataArray
            combined = combined.to_array().sum(dim="variable")

    # Collapse any extra dims so combined has dims ('year', 'latitude')
    extra_dims = [d for d in combined.dims if d not in ("year", "latitude")]
    if extra_dims:
        combined = combined.sum(dim=extra_dims)

    # Attempt to ensure proper ordering
    try:
        combined = combined.transpose("year", "latitude")
        combined = combined.sortby("latitude", ascending=False)
    except Exception:
        pass

    plt.figure(figsize=(12, 8))
    # Use matplotlib imshow for a stable 2D Hovmöller plot (years x latitude).
    # combined.values has shape (n_years, n_latitudes)
    data2d = combined.values

    # If still higher-dimensional, reduce extra axes (sum over middle axes)
    if data2d.ndim > 2:
        # sum over axes 1..n-2
        axes = tuple(range(1, data2d.ndim - 1))
        data2d = data2d.sum(axis=axes)

    # Ensure we have a 2D array (time x latitude)
    if data2d.ndim != 2:
        data2d = np.squeeze(data2d)
        if data2d.ndim != 2:
            raise ValueError(
                f"Unexpected data shape for Hovmoller: {combined.values.shape}"
            )

    years = np.asarray(combined["year"].values)
    lats = np.asarray(combined["latitude"].values)

    # Guarantee years are increasing and data is aligned
    sort_idx = np.argsort(years)
    if not np.all(sort_idx == np.arange(len(years))):
        years = years[sort_idx]
        data2d = data2d[sort_idx, :]

    # extent = (xmin, xmax, ymin, ymax)
    extent = (
        float(years.min()),
        float(years.max()),
        float(lats.max()),
        float(lats.min()),
    )

    im = plt.imshow(
        data2d.T,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    plt.xlabel("Year")
    plt.ylabel("Latitude")
    plt.title(f"Population Density by Latitude over Time (Age: {age_str})")
    plt.colorbar(im, label="Population")
    plt.tight_layout()
    plt.show()


# --- COMBINED DATASET PLOTTING ---
def plot_population_trends_from_combined(infants_da, elderly_da):
    """Produce trend plots (millions, growth rate, absolute) and save them."""

    # Helper to get summed series
    def summed_series(da):
        if isinstance(da, xr.Dataset):
            if "pop" in da:
                da = da["pop"]
            else:
                da = da.to_array().sum(dim="variable")
        s = da.sum(dim=["latitude", "longitude"])  # DataArray indexed by year
        return s

    inf_sum = summed_series(infants_da)
    eld_sum = summed_series(elderly_da)

    # Build DataFrame for plotting
    def series_to_df(s, label):
        yrs = np.asarray(s["year"].values)
        vals = np.asarray(s.values)
        return pd.DataFrame({"Year": yrs, "Age_Group": label, "Total_Population": vals})

    df = pd.concat(
        [
            series_to_df(inf_sum, Vars.infants),
            series_to_df(eld_sum, Vars.over_65),
        ],
        ignore_index=True,
    )
    df = df.sort_values(["Age_Group", "Year"]).reset_index(drop=True)

    # Plot 1: Total (millions)
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="Year",
        y=df["Total_Population"] / 1e6,
        hue="Age_Group",
        marker="o",
        ax=ax,
    )
    ax.set_title("Global Population Trend (millions)")
    ax.set_ylabel("Total People (millions)")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    # ax.axvline(x=2015, color="r", linestyle=":", alpha=0.5)
    save_fig(fig, "global_population_trend_millions.pdf")
    plt.show()

    # Plot 2: Growth rate
    df["Growth_Rate"] = df.groupby("Age_Group")["Total_Population"].pct_change() * 100
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Year", y="Growth_Rate", hue="Age_Group", marker="o", ax=ax)
    ax.set_title("Year-over-Year Growth Rate (%)")
    ax.set_ylabel("Growth Rate (%)")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhspan(-0.5, 2.0, color="green", alpha=0.1)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    save_fig(fig, "global_population_growth_rate.pdf")
    plt.show()


def plot_hovmoller_from_da(da, age_label="under_1"):
    """Create and save a Hovmöller (latitude x year) from a combined DataArray."""
    if isinstance(da, xr.Dataset):
        if "pop" in da:
            da = da["pop"]
        else:
            da = da.to_array().sum(dim="variable")

    # Sum across longitude to get zonal totals per latitude
    if "longitude" in da.dims:
        zonal = da.sum(dim="longitude")
    elif "lon" in da.dims:
        zonal = da.sum(dim="lon")
    else:
        # try to infer
        dims = list(da.dims)
        if "latitude" in dims and len(dims) == 2:
            zonal = da
        else:
            # sum all dims except latitude
            sum_dims = [d for d in da.dims if d != "latitude"]
            zonal = da.sum(dim=sum_dims)

    # Ensure latitude dim is named 'latitude'
    if "latitude" not in zonal.dims:
        # attempt to rename common variants
        for d in zonal.dims:
            if "lat" in str(d).lower():
                zonal = zonal.rename({d: "latitude"})
                break

    # Collapse any leftover dims except year/latitude
    extra = [d for d in zonal.dims if d not in ("year", "latitude")]
    if extra:
        zonal = zonal.sum(dim=extra)

    # Sort latitude descending
    zonal = zonal.sortby("latitude", ascending=False)

    # Sort years
    yrs = np.asarray(zonal["year"].values)
    sort_idx = np.argsort(yrs)

    data2d = np.asarray(zonal.values)
    if data2d.ndim == 1:
        # only latitude or only year - nothing to plot
        print(f"Not enough dimensions to plot Hovmoller for {age_label}")
        return

    if data2d.shape[0] == len(yrs):
        data2d_sorted = data2d[sort_idx, :]
    else:
        # try transpose
        data2d_sorted = data2d.T
        if data2d_sorted.shape[0] != len(yrs):
            data2d_sorted = data2d_sorted

    years = yrs[sort_idx]
    lats = np.asarray(zonal["latitude"].values)

    extent = (
        float(years.min()),
        float(years.max()),
        float(lats.max()),
        float(lats.min()),
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        data2d_sorted.T, aspect="auto", origin="upper", extent=extent, cmap="viridis"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Population by Latitude over Time ({age_label})")
    fig.colorbar(im, ax=ax, label="Population")
    save_fig(fig, f"hovmoller_{age_label}.png")


def plot_population_trends(infants_da, elderly_da):
    fig, axs = plt.subplots(1, 1, sharey=True)
    inf_sum = infants_da.sum(dim=["latitude", "longitude"])
    eld_sum = elderly_da.sum(dim=["latitude", "longitude"])
    axs.scatter(inf_sum.year, inf_sum / 10**6, label="Infants")
    axs.scatter(eld_sum.year, eld_sum / 10**6, label="Above 65")
    axs.legend()
    axs.set(xlabel="Year", ylabel="Population (millions)")
    plt.tight_layout()
    plt.savefig(DirsLocal.figures / "pop_data_trends.png")
    plt.show()

    for data, label in zip([eld_sum, inf_sum], ["65_over", "infants"]):
        fig, axs = plt.subplots(1, 1, sharey=True)
        axs.scatter(data.year, data / 10**6, label=label)
        axs.legend()
        axs.set(xlabel="Year", ylabel="Population (millions)")
        plt.tight_layout()
        plt.savefig(DirsLocal.figures / f"pop_data_trends_{label}.png")
        plt.show()


def main():
    """Load combined files and produce all plots, saving them into the interim folder."""

    # Load combined datasets
    infants_da = xr.open_dataset(FilesLocal.pop_infant)
    elderly_da = xr.open_dataset(FilesLocal.pop_over_65)

    # Trend plots (global totals, growth rate, absolute counts)
    plot_population_trends_from_combined(infants_da=infants_da, elderly_da=elderly_da)

    # Hovmöller per group
    plot_hovmoller_from_da(infants_da, age_label="under_1")
    plot_hovmoller_from_da(elderly_da, age_label="65_over")


if __name__ == "__main__":
    main()
