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

# Assuming your config is accessible
from my_config import Dirs


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
        x=2020, color="r", linestyle=":", alpha=0.5, label="Potential Splice (2020)"
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
    plt.axvline(x=2020, color="r", linestyle=":", alpha=0.5)
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
                (d for d in dims if "lat" in d.lower() or d in ("latitude", "y")), None
            )
            lon_dim = next(
                (d for d in dims if "lon" in d.lower() or d in ("longitude", "x")), None
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
                        if "lat" in d.lower() or d in ("latitude", "y")
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


if __name__ == "__main__":
    # Directly use filename labels for the three groups
    ages = ["under_1", "65_over", "75_over"]

    # Run Global Trends
    df_results = analyze_population_time_series(
        directory=Dirs.dir_pop_era_grid, ages_array=ages
    )

    # # Run Zonal Map for each group
    # for a in ages:
    #     plot_zonal_hovmoller(Dirs.dir_pop_era_grid, age_label=a)
