"""
Analyze and plot population trends for renamed ERA5-compatible total files.

This script looks for files named like:
  t_under_1_2021_era5_compatible.nc
  t_65_over_2022_era5_compatible.nc
  t_75_over_2023_era5_compatible.nc

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
from my_config import DirsLocal, Vars


def robust_open_pop(path: Path):
    """Open a combined NetCDF file and return an xarray DataArray for population.

    - Accepts Path or string.
    - Renames 'time' dim to 'year' if needed.
    - Returns a DataArray (pop) or raises FileNotFoundError.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ds = xr.open_dataset(path)

    # Normalize time dimension name
    if "time" in ds.dims and "year" not in ds.dims:
        ds = ds.rename({"time": "year"})

    # If dataset contains 'pop' variable, return it; otherwise try to pick the first data var
    if "pop" in ds.data_vars:
        da = ds["pop"]
    else:
        # pick first variable
        varname = list(ds.data_vars)[0]
        da = ds[varname]

    return da


def save_fig(fig, name: str):
    """Save figure to the interim directory with a consistent name."""
    out_dir = Path(DirsLocal.dir_figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_population_map(data, label, year=2001, bounds=(0, 35, 20, 47), v_max=20000):
    # Create a figure and axis with a cartopy projection centered on longitude 0
    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)},
        constrained_layout=True,
    )
    # Add features to the map
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.OCEAN)

    # Select the data for the specified year
    data = data.sel(year=year)

    # Set the CRS for the data
    data = data.rio.write_crs("EPSG:4326")

    if bounds:
        # Create a GeoDataFrame with the specified bounds
        gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:4326")

        # Clip the data using the GeoDataFrame
        data = data.rio.clip(gdf.geometry.values, gdf.crs, drop=True)

    # Plot the population data
    data.plot(
        vmax=v_max,
        cmap="viridis",
        ax=ax,
        cbar_kwargs={"orientation": "horizontal", "pad": 0.05, "label": "Population"},
    )

    plt.title(f"{label.capitalize()} population in {year}")
    # Show the plot
    plt.savefig(DirsLocal.dir_figures / f"pop_data_{label}_{year}.png")
    plt.show()


def analyze_population_time_series(directory, ages_array):
    """
    Scans a directory for t_... files, aggregates global totals,
    and generates diagnostic plots to find discontinuities.

    ages_array: list of filename age labels, e.g. ['under_1','65_over','75_over']
    """
    directory = Path(directory)
    records = []

    age_groups = list(ages_array)

    print(f"Scanning directory: {directory}")

    for age_str in age_groups:
        files = sorted(list(directory.glob(f"t_{age_str}_*_era5_compatible.nc")))

        if not files:
            print(f"No files found for age group: {age_str}")
            continue

        for file in tqdm(files, desc=f"Reading {age_str}"):
            try:
                m = re.search(r"(19|20)\d{2}", file.name)
                if not m:
                    print(f"  Could not find year in filename: {file.name}")
                    continue
                year = int(m.group(0))

                with xr.open_dataset(file) as ds:
                    if "pop" not in ds:
                        print(f"  'pop' variable not in {file.name}; skipping")
                        continue
                    total_pop = float(ds["pop"].sum().item())

                    records.append(
                        {
                            "Year": year,
                            "Age_Group": age_str,
                            "Total_Population": total_pop,
                            "Path": file,
                        }
                    )
            except Exception as e:
                print(f"Error reading {file.name}: {e}")

    if not records:
        print("No data found to plot.")
        return

    df = pd.DataFrame(records).sort_values(["Age_Group", "Year"]).reset_index(drop=True)

    # Print absolute values table (people) for quick inspection
    pivot = df.pivot(
        index="Year", columns="Age_Group", values="Total_Population"
    ).sort_index()
    pd.options.display.float_format = "{:,.0f}".format
    print("\nAbsolute totals (people) by year and age group:")
    print(pivot.to_string())

    # Add a helper column for plotting in millions (readable axis)
    df_plot = df.copy()
    df_plot["Total_Pop_Millions"] = df_plot["Total_Population"] / 1e6

    # 2. Plot 1: Total Global Population (Absolute Numbers)
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df_plot, x="Year", y="Total_Pop_Millions", hue="Age_Group", marker="o"
    )
    plt.title("Global Population Trend (Check for Steps/Jumps)")
    plt.ylabel("Total People (millions)")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.axvline(
        x=2015, color="r", linestyle=":", alpha=0.5, label="Potential Splice (2015)"
    )
    plt.legend()
    plt.show()

    # 3. Plot 2: Year-over-Year Growth Rate (The "Jump" Detector)
    df["Growth_Rate"] = df.groupby("Age_Group")["Total_Population"].pct_change() * 100

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Year", y="Growth_Rate", hue="Age_Group", marker="o")
    plt.title("Year-over-Year Growth Rate (Artifact Detector)")
    plt.ylabel("Growth Rate (%)")
    plt.axhline(0, color="k", linewidth=0.8)
    # Highlight normal range (approx -0.5% to +2.0%)
    plt.axhspan(-0.5, 2.0, color="green", alpha=0.1, label="Expected Normal Growth")
    plt.axvline(x=2015, color="r", linestyle=":", alpha=0.5)
    plt.legend()
    plt.show()

    return df


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
        float(lats.min()),
        float(lats.max()),
    )

    im = plt.imshow(
        data2d.T,
        aspect="auto",
        origin="lower",
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
def plot_population_trends_from_combined(infants_da, elderly_da, elderly75_da):
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
    eld75_sum = summed_series(elderly75_da)

    # Build DataFrame for plotting
    def series_to_df(s, label):
        yrs = np.asarray(s["year"].values)
        vals = np.asarray(s.values)
        return pd.DataFrame({"Year": yrs, "Age_Group": label, "Total_Population": vals})

    df = pd.concat(
        [
            series_to_df(inf_sum, "under_1"),
            series_to_df(eld_sum, "65_over"),
            series_to_df(eld75_sum, "75_over"),
        ],
        ignore_index=True,
    )
    df = df.sort_values(["Age_Group", "Year"]).reset_index(drop=True)

    # Plot 1: Total (millions)
    fig, ax = plt.subplots(figsize=(12, 6))
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
    ax.axvline(x=2015, color="r", linestyle=":", alpha=0.5)
    save_fig(fig, "global_population_trend_millions.png")

    # Plot 2: Growth rate
    df["Growth_Rate"] = df.groupby("Age_Group")["Total_Population"].pct_change() * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x="Year", y="Growth_Rate", hue="Age_Group", marker="o", ax=ax)
    ax.set_title("Year-over-Year Growth Rate (%)")
    ax.set_ylabel("Growth Rate (%)")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhspan(-0.5, 2.0, color="green", alpha=0.1)
    ax.axvline(x=2015, color="r", linestyle=":", alpha=0.5)
    save_fig(fig, "global_population_growth_rate.png")

    # Plot 3: Absolute totals (raw counts)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=df, x="Year", y="Total_Population", hue="Age_Group", marker="o", ax=ax
    )
    ax.set_title("Global Population Trend — Absolute Counts")
    ax.set_ylabel("Total People")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.axvline(x=2015, color="r", linestyle=":", alpha=0.5)
    save_fig(fig, "global_population_trend_absolute.png")


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
        float(lats.min()),
        float(lats.max()),
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        data2d_sorted.T, aspect="auto", origin="lower", extent=extent, cmap="viridis"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Population by Latitude over Time ({age_label})")
    fig.colorbar(im, ax=ax, label="Population")
    save_fig(fig, f"hovmoller_{age_label}.png")


def plot_population_trends(infants_da, elderly_da, elderly75_da):
    fig, axs = plt.subplots(1, 1, sharey=True)
    inf_sum = infants_da.sum(dim=["latitude", "longitude"])
    eld_sum = elderly_da.sum(dim=["latitude", "longitude"])
    eld_75 = elderly75_da.sum(dim=["latitude", "longitude"])
    axs.scatter(inf_sum.year, inf_sum / 10**6, label="Infants")
    axs.scatter(eld_sum.year, eld_sum / 10**6, label="Above 65")
    axs.scatter(eld_75.year, eld_75 / 10**6, label="Above 75")
    axs.legend()
    axs.set(xlabel="Year", ylabel="Population (millions)")
    plt.tight_layout()
    plt.savefig(DirsLocal.dir_figures / "pop_data_trends.png")
    plt.show()

    for data, label in zip(
        [eld_sum, eld_75, inf_sum], ["65_over", "75_over", "infants"]
    ):
        fig, axs = plt.subplots(1, 1, sharey=True)
        axs.scatter(data.year, data / 10**6, label=label)
        axs.legend()
        axs.set(xlabel="Year", ylabel="Population (millions)")
        plt.tight_layout()
        plt.savefig(DirsLocal.dir_figures / f"pop_data_trends_{label}.png")
        plt.show()


def main():
    """Load combined files and produce all plots, saving them into the interim folder."""
    # Paths (these should be created by c_pop_data_combine.py)
    path_inf = Path(DirsLocal.dir_pop_infants_file)
    path_eld = Path(DirsLocal.dir_pop_elderly_file)
    path_75 = Path(DirsLocal.dir_pop_above_75_file)

    # Load combined datasets
    infants_da = robust_open_pop(path_inf)
    elderly_da = robust_open_pop(path_eld)
    elderly75_da = robust_open_pop(path_75)

    # Trend plots (global totals, growth rate, absolute counts)
    plot_population_trends_from_combined(infants_da, elderly_da, elderly75_da)

    # Hovmöller per group
    plot_hovmoller_from_da(infants_da, age_label="under_1")
    plot_hovmoller_from_da(elderly_da, age_label="65_over")
    plot_hovmoller_from_da(elderly75_da, age_label="75_over")

    # Sample year maps (latest)
    sample_year = getattr(Vars, "year_max_analysis", None)

    plot_population_map(
        infants_da, label="infants", year=sample_year, bounds=None, v_max=20000
    )
    plot_population_map(
        elderly_da, label="elderly", year=sample_year, bounds=None, v_max=20000
    )
    plot_population_map(
        elderly75_da, label="75_over", year=sample_year, bounds=None, v_max=20000
    )

    # Also show interactive quick plots if requested
    df = analyze_population_time_series(
        directory=DirsLocal.dir_pop_era_grid,
        ages_array=["under_1", "65_over", "75_over"],
    )  # legacy per-year t_ files
    print("Interactive summary (legacy t_ files):")
    print(df.head())

    plot_population_trends(infants_da, elderly_da, elderly75_da)


if __name__ == "__main__":
    # pass
    main()
