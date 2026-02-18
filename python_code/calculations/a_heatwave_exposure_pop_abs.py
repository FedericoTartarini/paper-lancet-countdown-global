"""
Calculate heatwave exposure for vulnerable groups (infants and over_65).

This script multiplies yearly heatwave metrics by population on the same ERA5-Land grid:
- heatwave_days * population
- heatwave_counts * population

Output contains only: age_band, latitude, longitude, year.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from my_config import DirsLocal, FilesLocal, Vars, ensure_directories


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def normalize_coords(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Ensure standard coordinate names (latitude/longitude) are used."""
    rename_map = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def standardize_grid(
    da: xr.Dataset | xr.DataArray,
    decimals: int = 3,
) -> xr.Dataset | xr.DataArray:
    """Round coordinates and sort lat/lon to a consistent order."""
    da = normalize_coords(da)

    if "latitude" in da.coords:
        da = da.assign_coords(latitude=da["latitude"].round(decimals))
    if "longitude" in da.coords:
        da = da.assign_coords(longitude=da["longitude"].round(decimals))

    if "latitude" in da.coords:
        da = da.sortby("latitude")
    if "longitude" in da.coords:
        da = da.sortby("longitude")

    return da


def get_lat_lon(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Return latitude and longitude arrays from a DataArray."""
    if "latitude" not in da.coords or "longitude" not in da.coords:
        raise ValueError("Missing latitude/longitude coordinates in dataset.")
    return da["latitude"].values, da["longitude"].values


def assert_matching_grid(pop_da: xr.DataArray, hw_da: xr.DataArray) -> None:
    """Assert that population and heatwave grids use matching lat/lon coordinates."""
    pop_da = standardize_grid(pop_da)
    hw_da = standardize_grid(hw_da)

    pop_lat, pop_lon = get_lat_lon(pop_da)
    hw_lat, hw_lon = get_lat_lon(hw_da)

    if pop_lat.shape != hw_lat.shape or pop_lon.shape != hw_lon.shape:
        raise ValueError(
            "Population and heatwave grids have different coordinate sizes: "
            f"pop_lat={pop_lat.shape}, hw_lat={hw_lat.shape}, "
            f"pop_lon={pop_lon.shape}, hw_lon={hw_lon.shape}."
        )

    if not np.allclose(pop_lat, hw_lat, rtol=0, atol=1e-3):
        max_diff = float(np.max(np.abs(pop_lat - hw_lat)))
        raise ValueError(
            "Latitude coordinates do not match between population and heatwave data. "
            f"pop_lat range=({pop_lat.min()}, {pop_lat.max()}), "
            f"hw_lat range=({hw_lat.min()}, {hw_lat.max()}), "
            f"max_diff={max_diff}."
        )

    if not np.allclose(pop_lon, hw_lon, rtol=0, atol=1e-3):
        max_diff = float(np.max(np.abs(pop_lon - hw_lon)))
        raise ValueError(
            "Longitude coordinates do not match between population and heatwave data. "
            f"pop_lon range=({pop_lon.min()}, {pop_lon.max()}), "
            f"hw_lon range=({hw_lon.min()}, {hw_lon.max()}), "
            f"max_diff={max_diff}."
        )


def select_single_var(ds: xr.Dataset, name_hint: str) -> xr.DataArray:
    """Select a single variable from a Dataset, using a name hint if available."""
    if name_hint in ds.data_vars:
        return ds[name_hint]
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars.values()))
    raise ValueError(
        f"Population dataset has multiple variables. Provide a named variable. "
        f"Available: {list(ds.data_vars)}"
    )


def load_population(path: Path, name_hint: str, years: slice | None) -> xr.DataArray:
    """Load population data as a DataArray with standardized coords."""
    if not path.exists():
        raise FileNotFoundError(f"Population file not found: {path}")

    ds = xr.open_dataset(path)
    ds = standardize_grid(ds)

    pop_da = select_single_var(ds, name_hint)
    pop_da = standardize_grid(pop_da)

    # Drop known auxiliary coordinates
    for coord in ["band", "spatial_ref", "age_band_lower_bound", "variable"]:
        if coord in pop_da.coords:
            pop_da = pop_da.drop_vars(coord)

    if years is not None:
        pop_da = pop_da.sel(year=years)

    return pop_da


def resolve_hw_var(hw_ds: xr.Dataset, var_name: str) -> xr.DataArray:
    """Resolve a heatwave variable, supporting common naming variants."""
    if var_name in hw_ds.data_vars:
        return hw_ds[var_name]

    aliases = {
        Vars.hw_count: ["heatwave_count", "heatwave_counts"],
        Vars.hw_days: ["heatwave_day", "heatwave_days"],
    }

    for alias in aliases.get(var_name, []):
        if alias in hw_ds.data_vars:
            logging.warning("Using heatwave variable alias: %s -> %s", alias, var_name)
            return hw_ds[alias]

    raise ValueError(
        f"Heatwave variable not found: {var_name}. Available: {list(hw_ds.data_vars)}"
    )


def load_heatwave_metrics(years: slice | None) -> xr.Dataset:
    """Load heatwave days and counts as a Dataset."""
    hw_files = sorted(DirsLocal.hw_q_min_max.glob("*.nc"))
    if not hw_files:
        raise FileNotFoundError(f"No heatwave files found in {DirsLocal.hw_q_min_max}")

    hw_ds = xr.open_mfdataset(
        hw_files,
        combine="by_coords",
        parallel=False,
        decode_timedelta=True,
    )
    hw_ds = standardize_grid(hw_ds)

    if years is not None:
        hw_ds = hw_ds.sel(year=years)

    return hw_ds


def calculate_exposure(pop_da: xr.DataArray, hw_da: xr.DataArray) -> xr.DataArray:
    """Multiply population by a heatwave metric after alignment on common years."""
    pop_da = standardize_grid(pop_da)
    hw_da = standardize_grid(hw_da)

    if "year" in pop_da.coords and "year" in hw_da.coords:
        common_years = np.intersect1d(pop_da["year"].values, hw_da["year"].values)
        if common_years.size == 0:
            raise ValueError(
                "No overlapping years between population and heatwave data."
            )
        pop_da = pop_da.sel(year=common_years)
        hw_da = hw_da.sel(year=common_years)

    if np.issubdtype(hw_da.dtype, np.timedelta64):
        hw_da = hw_da / np.timedelta64(1, "D")

    pop_da, hw_da = xr.align(pop_da, hw_da, join="exact")
    exposure = (pop_da * hw_da).astype("float32")
    return exposure


def build_combined_dataset(
    pop_inf: xr.DataArray,
    pop_old: xr.DataArray,
    hw_ds: xr.Dataset,
) -> xr.Dataset:
    """Build a combined dataset with exposures for both age bands."""
    hw_days = resolve_hw_var(hw_ds, Vars.hw_days)
    hw_counts = resolve_hw_var(hw_ds, Vars.hw_count)

    pop_inf = standardize_grid(pop_inf)
    pop_old = standardize_grid(pop_old)
    hw_days = standardize_grid(hw_days)
    hw_counts = standardize_grid(hw_counts)

    assert_matching_grid(pop_inf, hw_days)
    assert_matching_grid(pop_old, hw_days)

    logging.info("Calculating exposure for %s...", Vars.infants)
    days_inf = calculate_exposure(pop_inf, hw_days)
    counts_inf = calculate_exposure(pop_inf, hw_counts)

    logging.info("Calculating exposure for %s...", Vars.over_65)
    days_old = calculate_exposure(pop_old, hw_days)
    counts_old = calculate_exposure(pop_old, hw_counts)

    days_combined = xr.concat(
        [days_inf, days_old],
        dim=pd.Index([Vars.infants, Vars.over_65], name=Vars.age_band),
    )
    counts_combined = xr.concat(
        [counts_inf, counts_old],
        dim=pd.Index([Vars.infants, Vars.over_65], name=Vars.age_band),
    )

    combined = xr.Dataset(
        {
            Vars.hw_days: days_combined,
            Vars.hw_count: counts_combined,
        }
    )

    combined = combined.transpose(Vars.age_band, "latitude", "longitude", "year")

    return combined


def coerce_numeric_for_plotting(da: xr.DataArray) -> xr.DataArray:
    """Convert timedelta data to days for plotting, keep numeric data as-is."""
    if np.issubdtype(da.dtype, np.timedelta64):
        return da / np.timedelta64(1, "D")
    return da


def plot_mediterranean_panels(
    pop_old: xr.DataArray,
    pop_inf: xr.DataArray,
    hw_ds: xr.Dataset,
    year: int = 2020,
) -> None:
    """Plot population and heatwave metrics for the Mediterranean region."""
    lat_slice = slice(35, 45)
    lon_slice = slice(10, 20)

    hw_days = resolve_hw_var(hw_ds, Vars.hw_days)
    hw_counts = resolve_hw_var(hw_ds, Vars.hw_count)

    hw_days = standardize_grid(hw_days)
    hw_counts = standardize_grid(hw_counts)

    pop_old = standardize_grid(pop_old)
    pop_inf = standardize_grid(pop_inf)

    pop_old_sel = pop_old.sel(year=year, latitude=lat_slice, longitude=lon_slice)
    pop_inf_sel = pop_inf.sel(year=year, latitude=lat_slice, longitude=lon_slice)
    hw_days_sel = hw_days.sel(year=year, latitude=lat_slice, longitude=lon_slice)
    hw_counts_sel = hw_counts.sel(year=year, latitude=lat_slice, longitude=lon_slice)

    hw_days_sel = coerce_numeric_for_plotting(hw_days_sel)
    hw_counts_sel = coerce_numeric_for_plotting(hw_counts_sel)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    pop_old_sel.plot(ax=axs[0, 0], cmap="viridis", robust=True, add_colorbar=True)
    axs[0, 0].set_title(f"Over 65 Population ({year})")

    pop_inf_sel.plot(ax=axs[0, 1], cmap="viridis", robust=True, add_colorbar=True)
    axs[0, 1].set_title(f"Infants Population ({year})")

    hw_days_sel.plot(ax=axs[1, 0], cmap="magma", robust=True, add_colorbar=True)
    axs[1, 0].set_title(f"Heatwave Days ({year})")

    hw_counts_sel.plot(ax=axs[1, 1], cmap="magma", robust=True, add_colorbar=True)
    axs[1, 1].set_title(f"Heatwave Counts ({year})")

    plt.show()


def load_combined_results() -> xr.Dataset:
    """Load the combined exposure dataset from disk."""
    if not FilesLocal.hw_combined_q.exists():
        raise FileNotFoundError(
            f"Combined output not found: {FilesLocal.hw_combined_q}"
        )
    return xr.open_dataset(FilesLocal.hw_combined_q)


def plot_global_trends(ds: xr.Dataset, population: str) -> None:
    """Plot global person-days and person-events for a population."""
    days = ds[Vars.hw_days].sel(age_band=population).sum(dim=["latitude", "longitude"])
    counts = (
        ds[Vars.hw_count].sel(age_band=population).sum(dim=["latitude", "longitude"])
    )

    years = days["year"].values

    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(years, days / 1e9, color="tab:red", marker="o", label="Person-Days")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Person-Days (Billions)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(
        years,
        counts / 1e9,
        color="tab:blue",
        marker="s",
        linestyle="--",
        label="Person-Events",
    )
    ax2.set_ylabel("Total Person-Events (Billions)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.grid(False)

    ax1.set_title(f"Global Heatwave Exposure: {population}")
    fig.tight_layout()
    plt.savefig(DirsLocal.figures / f"hw_exposure_global_trends_{population}.pdf")
    plt.show()


def plot_global_trends_combined(ds: xr.Dataset) -> None:
    """Plot global person-days and person-events for both populations."""
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    colors_days = {Vars.over_65: "tab:red", Vars.infants: "tab:green"}
    colors_counts = {Vars.over_65: "tab:blue", Vars.infants: "tab:orange"}

    for population in [Vars.over_65, Vars.infants]:
        days = (
            ds[Vars.hw_days].sel(age_band=population).sum(dim=["latitude", "longitude"])
        )
        counts = (
            ds[Vars.hw_count]
            .sel(age_band=population)
            .sum(dim=["latitude", "longitude"])
        )
        years = days["year"].values

        ax1.plot(
            years,
            days / 1e9,
            color=colors_days[population],
            marker="o",
            label=f"Days ({population})",
        )
        ax2.plot(
            years,
            counts / 1e9,
            color=colors_counts[population],
            marker="s",
            linestyle="--",
            label=f"Counts ({population})",
        )

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Person-Days (Billions)")
    ax2.set_ylabel("Total Person-Events (Billions)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True, alpha=0.3)
    ax2.grid(False)
    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)
    ax2.set(yticks=np.arange(0, 3.1, 0.5))

    ax1.set_title("Global Heatwave Exposure: Burden vs Frequency")
    fig.tight_layout()
    sns.despine()
    plt.savefig(DirsLocal.figures / "hw_exposure_global_trends_combined.pdf")
    plt.show()


def plot_severity_ratio(ds: xr.Dataset, population: str) -> None:
    """Plot ratio of person-days to person-events for a population."""
    days = ds[Vars.hw_days].sel(age_band=population).sum(dim=["latitude", "longitude"])
    counts = (
        ds[Vars.hw_count].sel(age_band=population).sum(dim=["latitude", "longitude"])
    )

    ratio = days / counts.where(counts != 0)

    plt.figure(figsize=(7, 4))
    plt.plot(ratio["year"].values, ratio, marker="o", color="tab:purple")
    plt.xlabel("Year")
    plt.ylabel("Avg Days per Person-Event")
    plt.title(f"Heatwave Severity Ratio: {population}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DirsLocal.figures / f"hw_exposure_severity_ratio_{population}.pdf")
    plt.show()


def plot_severity_ratio_combined(ds: xr.Dataset) -> None:
    """Plot severity ratio for both populations in one chart."""
    plt.figure(figsize=(7, 4))

    for population in [Vars.over_65, Vars.infants]:
        days = (
            ds[Vars.hw_days].sel(age_band=population).sum(dim=["latitude", "longitude"])
        )
        counts = (
            ds[Vars.hw_count]
            .sel(age_band=population)
            .sum(dim=["latitude", "longitude"])
        )
        ratio = days / counts.where(counts != 0)

        plt.plot(ratio["year"].values, ratio, marker="o", label=population)

    plt.xlabel("Year")
    plt.ylabel("Avg Days per Person-Event")
    plt.title("Heatwave Severity Ratio: Infants vs Over 65")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DirsLocal.figures / "hw_exposure_severity_ratio_combined.pdf")
    plt.show()


def plot_exposure_map(
    ds: xr.Dataset,
    year: int,
    population: str,
    metric: str,
) -> None:
    """Plot a global exposure map for a given population and metric."""
    data = ds[metric].sel(age_band=population, year=year)
    data.plot(cmap="inferno", robust=True, add_colorbar=True)
    plt.title(f"{metric} Exposure ({population}) - {year}")
    plt.tight_layout()
    plt.savefig(DirsLocal.figures / f"hw_exposure_map_{metric}_{population}_{year}.pdf")
    plt.show()


def plot_mediterranean_exposure(ds: xr.Dataset, year: int = 2020) -> None:
    """Plot exposure results for the Mediterranean region."""
    lat_slice = slice(35, 45)
    lon_slice = slice(10, 20)

    days_inf = ds[Vars.hw_days].sel(
        age_band=Vars.infants, year=year, latitude=lat_slice, longitude=lon_slice
    )
    days_old = ds[Vars.hw_days].sel(
        age_band=Vars.over_65, year=year, latitude=lat_slice, longitude=lon_slice
    )
    counts_inf = ds[Vars.hw_count].sel(
        age_band=Vars.infants, year=year, latitude=lat_slice, longitude=lon_slice
    )
    counts_old = ds[Vars.hw_count].sel(
        age_band=Vars.over_65, year=year, latitude=lat_slice, longitude=lon_slice
    )

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    days_inf.plot(ax=axs[0, 0], cmap="magma", robust=True, add_colorbar=True)
    axs[0, 0].set_title(f"Infants Person-Days ({year})")

    days_old.plot(ax=axs[0, 1], cmap="magma", robust=True, add_colorbar=True)
    axs[0, 1].set_title(f"Over 65 Person-Days ({year})")

    counts_inf.plot(ax=axs[1, 0], cmap="viridis", robust=True, add_colorbar=True)
    axs[1, 0].set_title(f"Infants Person-Events ({year})")

    counts_old.plot(ax=axs[1, 1], cmap="viridis", robust=True, add_colorbar=True)
    axs[1, 1].set_title(f"Over 65 Person-Events ({year})")
    sns.despine()

    plt.savefig(DirsLocal.figures / f"hw_exposure_mediterranean_{year}.pdf")
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate heatwave exposure for infants and over_65."
    )
    parser.add_argument("--year", type=int, default=None, help="Process a single year")
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Process a single early year for a quick test",
    )
    return parser.parse_args()


def main() -> None:
    """Run heatwave exposure calculation and save combined output."""
    setup_logging()

    years = slice(Vars.year_min_analysis, Vars.year_max_analysis)
    logging.info(
        "Processing full analysis period: %s-%s",
        Vars.year_min_analysis,
        Vars.year_max_analysis,
    )

    logging.info("Loading population data...")
    pop_inf = load_population(FilesLocal.pop_inf, "pop", years)
    pop_old = load_population(FilesLocal.pop_over_65, "pop", years)

    logging.info("Loading heatwave metrics...")
    hw_ds = load_heatwave_metrics(years)

    plot_mediterranean_panels(pop_old=pop_old, pop_inf=pop_inf, hw_ds=hw_ds, year=1999)
    plot_mediterranean_panels(pop_old=pop_old, pop_inf=pop_inf, hw_ds=hw_ds, year=2020)

    logging.info("Building combined exposure dataset...")
    combined = build_combined_dataset(pop_inf, pop_old, hw_ds)

    ensure_directories([FilesLocal.hw_combined_q.parent])

    logging.info("Saving output: %s", FilesLocal.hw_combined_q)
    combined.to_netcdf(
        FilesLocal.hw_combined_q,
        encoding={
            Vars.hw_days: {"zlib": True},  # "complevel": 5},
            Vars.hw_count: {"zlib": True},  # "complevel": 5},
        },
    )


def plot_avg_hw_days_per_person(
    ds: xr.Dataset,
    pop_inf: xr.DataArray,
    pop_old: xr.DataArray,
) -> None:
    """Plot global average heatwave days per person for each population by year."""
    days_inf = (
        ds[Vars.hw_days].sel(age_band=Vars.infants).sum(dim=["latitude", "longitude"])
    )
    days_old = (
        ds[Vars.hw_days].sel(age_band=Vars.over_65).sum(dim=["latitude", "longitude"])
    )

    pop_inf_total = pop_inf.sum(dim=["latitude", "longitude"])
    pop_old_total = pop_old.sum(dim=["latitude", "longitude"])

    common_years = np.intersect1d(days_inf["year"].values, pop_inf_total["year"].values)
    common_years = np.intersect1d(common_years, pop_old_total["year"].values)
    if common_years.size == 0:
        raise ValueError("No overlapping years between exposure and population totals.")

    days_inf = days_inf.sel(year=common_years)
    days_old = days_old.sel(year=common_years)
    pop_inf_total = pop_inf_total.sel(year=common_years)
    pop_old_total = pop_old_total.sel(year=common_years)

    avg_inf = days_inf / pop_inf_total
    avg_old = days_old / pop_old_total

    plt.figure(figsize=(7, 4))
    plt.plot(avg_inf["year"].values, avg_inf, marker="o", label=Vars.infants)
    plt.plot(avg_old["year"].values, avg_old, marker="o", label=Vars.over_65)
    plt.xlabel("Year")
    plt.ylabel("Average Heatwave Days per Person")
    plt.title("Average Heatwave Days per Person (Global)")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    sns.despine()
    plt.savefig(DirsLocal.figures / "hw_exposure_avg_days_per_person.pdf")
    plt.show()


def plot() -> None:
    combined = xr.open_dataset(FilesLocal.hw_combined_q)

    pop_inf = load_population(FilesLocal.pop_inf, "pop", None)
    pop_old = load_population(FilesLocal.pop_over_65, "pop", None)

    logging.info("Generating plots from combined results...")
    plot_global_trends_combined(ds=combined)
    plot_global_trends(combined, Vars.over_65)
    plot_global_trends(combined, Vars.infants)
    plot_severity_ratio(combined, Vars.over_65)
    plot_severity_ratio(combined, Vars.infants)
    plot_severity_ratio_combined(combined)
    plot_avg_hw_days_per_person(ds=combined, pop_inf=pop_inf, pop_old=pop_old)
    # plot_exposure_map(combined, year=2020, population=Vars.over_65, metric=Vars.hw_days)
    # plot_exposure_map(combined, year=2020, population=Vars.infants, metric=Vars.hw_days)
    # plot_exposure_map(
    #     combined, year=2020, population=Vars.over_65, metric=Vars.hw_count
    # )
    # plot_exposure_map(
    #     combined, year=2020, population=Vars.infants, metric=Vars.hw_count
    # )
    plot_mediterranean_exposure(combined, year=2020)

    logging.info("✅ Done")


if __name__ == "__main__":
    pass
    main()
    plot()
