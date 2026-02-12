"""
Calculate exposure to change in heatwave occurrence relative to the reference period.

Outputs a single NetCDF with dimensions:
- age_band, latitude, longitude, year
and variables:
- heatwave_days, heatwave_counts
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from my_config import DirsLocal, FilesLocal, Vars, ensure_directories
from python_code.calculations.a_heatwave_exposure_pop_abs import (
    assert_matching_grid,
    calculate_exposure,
    load_heatwave_metrics,
    load_population,
    resolve_hw_var,
    standardize_grid,
)


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def assert_reference_years_available(hw_da: xr.DataArray) -> None:
    """Ensure the reference period is fully present in the heatwave data."""
    ref_years = np.arange(Vars.year_reference_start, Vars.year_reference_end + 1)
    available_years = hw_da["year"].values
    missing = sorted(set(ref_years) - set(available_years))
    if missing:
        raise ValueError(f"Missing reference years in heatwave data: {missing}")


def compute_reference_mean(hw_da: xr.DataArray) -> xr.DataArray:
    """Compute reference-period mean for a heatwave metric."""
    assert_reference_years_available(hw_da)
    return hw_da.sel(
        year=slice(Vars.year_reference_start, Vars.year_reference_end)
    ).mean(dim="year")


def compute_delta(hw_da: xr.DataArray) -> xr.DataArray:
    """Compute anomaly relative to reference-period mean."""
    ref_mean = compute_reference_mean(hw_da)
    return hw_da - ref_mean


def build_exposure_change_dataset(
    pop_inf: xr.DataArray,
    pop_old: xr.DataArray,
    hw_ds: xr.Dataset,
) -> xr.Dataset:
    """Build exposure-to-change dataset for infants and over_65."""
    hw_days = resolve_hw_var(hw_ds, Vars.hw_days)
    hw_counts = resolve_hw_var(hw_ds, Vars.hw_count)

    pop_inf = standardize_grid(pop_inf)
    pop_old = standardize_grid(pop_old)
    hw_days = standardize_grid(hw_days)
    hw_counts = standardize_grid(hw_counts)

    assert_matching_grid(pop_inf, hw_days)
    assert_matching_grid(pop_old, hw_days)
    assert_matching_grid(pop_inf, hw_counts)
    assert_matching_grid(pop_old, hw_counts)

    days_delta = compute_delta(hw_days)
    counts_delta = compute_delta(hw_counts)

    logging.info("Calculating exposure change for %s...", Vars.infants)
    days_inf = calculate_exposure(pop_inf, days_delta)
    counts_inf = calculate_exposure(pop_inf, counts_delta)

    logging.info("Calculating exposure change for %s...", Vars.over_65)
    days_old = calculate_exposure(pop_old, days_delta)
    counts_old = calculate_exposure(pop_old, counts_delta)

    days_combined = xr.concat(
        [days_inf, days_old],
        dim=xr.DataArray([Vars.infants, Vars.over_65], dims=[Vars.age_band]),
    )
    counts_combined = xr.concat(
        [counts_inf, counts_old],
        dim=xr.DataArray([Vars.infants, Vars.over_65], dims=[Vars.age_band]),
    )

    combined = xr.Dataset(
        {
            Vars.hw_days: days_combined,
            Vars.hw_count: counts_combined,
        }
    )

    combined = combined.transpose(Vars.age_band, "latitude", "longitude", "year")
    return combined


def weighted_mean(exposure: xr.DataArray, population: xr.DataArray) -> xr.DataArray:
    """Compute population-weighted mean change for an exposure metric."""
    total_exposure = exposure.sum(dim=["latitude", "longitude"], skipna=True)
    total_population = population.sum(dim=["latitude", "longitude"], skipna=True)
    return total_exposure / total_population


def plot_weighted_mean_change(
    combined: xr.Dataset,
    pop_inf: xr.DataArray,
    pop_old: xr.DataArray,
) -> None:
    """Plot weighted mean change for days and counts."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    for idx, metric in enumerate([Vars.hw_days, Vars.hw_count]):
        days_inf = weighted_mean(combined[metric].sel(age_band=Vars.infants), pop_inf)
        days_old = weighted_mean(combined[metric].sel(age_band=Vars.over_65), pop_old)

        axs[idx].plot(
            days_inf["year"].values,
            days_inf,
            label="Infants",
            linewidth=2,
        )
        axs[idx].plot(
            days_old["year"].values,
            days_old,
            label="Over 65",
            linewidth=2,
        )
        axs[idx].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axs[idx].set_ylabel(f"Weighted mean change ({metric})")
        axs[idx].legend()

    axs[-1].set_xlabel("Year")
    fig.suptitle(
        f"Population-Weighted Change vs {Vars.year_reference_start}-{Vars.year_reference_end}"
    )
    plt.savefig(DirsLocal.figures / "hw_change_weighted_mean.pdf")
    sns.despine()
    plt.show()


def plot_total_exposure_change(combined: xr.Dataset) -> None:
    """Plot total exposure change over time for days and counts."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    for idx, metric in enumerate([Vars.hw_days, Vars.hw_count]):
        data_inf = (
            combined[metric]
            .sel(age_band=Vars.infants)
            .sum(dim=["latitude", "longitude"], skipna=True)
        )
        data_old = (
            combined[metric]
            .sel(age_band=Vars.over_65)
            .sum(dim=["latitude", "longitude"], skipna=True)
        )

        axs[idx].plot(
            data_inf["year"].values,
            data_inf / 1e9,
            label="Infants",
            linewidth=2,
        )
        axs[idx].plot(
            data_old["year"].values,
            data_old / 1e9,
            label="Over 65",
            linewidth=2,
        )
        axs[idx].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axs[idx].set_ylabel(f"Total change ({metric}) in billions")
        axs[idx].legend()

    axs[-1].set_xlabel("Year")
    fig.suptitle("Total Exposure Change (Global)")
    plt.savefig(DirsLocal.figures / "hw_change_total_exposure.pdf")
    sns.despine()
    plt.show()


def plot_mediterranean_change(
    combined: xr.Dataset,
    pop_inf: xr.DataArray,
    pop_old: xr.DataArray,
    year: int = 2020,
) -> None:
    """Plot Mediterranean regional change maps and time series."""
    lat_slice = slice(35, 45)
    lon_slice = slice(10, 20)

    pop_inf_reg = pop_inf.sel(latitude=lat_slice, longitude=lon_slice)
    pop_old_reg = pop_old.sel(latitude=lat_slice, longitude=lon_slice)

    combined_reg = combined.sel(latitude=lat_slice, longitude=lon_slice)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    combined_reg[Vars.hw_days].sel(age_band=Vars.infants, year=year).plot(
        ax=axs[0, 0], cmap="magma", robust=True, add_colorbar=True
    )
    axs[0, 0].set_title(f"Infants Days Change ({year})")

    combined_reg[Vars.hw_days].sel(age_band=Vars.over_65, year=year).plot(
        ax=axs[0, 1], cmap="magma", robust=True, add_colorbar=True
    )
    axs[0, 1].set_title(f"Over 65 Days Change ({year})")

    combined_reg[Vars.hw_count].sel(age_band=Vars.infants, year=year).plot(
        ax=axs[1, 0], cmap="viridis", robust=True, add_colorbar=True
    )
    axs[1, 0].set_title(f"Infants Events Change ({year})")

    combined_reg[Vars.hw_count].sel(age_band=Vars.over_65, year=year).plot(
        ax=axs[1, 1], cmap="viridis", robust=True, add_colorbar=True
    )
    axs[1, 1].set_title(f"Over 65 Events Change ({year})")

    plt.savefig(DirsLocal.figures / f"hw_change_mediterranean_maps_{year}.pdf")
    sns.despine()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    for idx, metric in enumerate([Vars.hw_days, Vars.hw_count]):
        inf_mean = weighted_mean(
            combined_reg[metric].sel(age_band=Vars.infants), pop_inf_reg
        )
        old_mean = weighted_mean(
            combined_reg[metric].sel(age_band=Vars.over_65), pop_old_reg
        )

        axs[idx].plot(inf_mean["year"].values, inf_mean, label="Infants")
        axs[idx].plot(old_mean["year"].values, old_mean, label="Over 65")
        axs[idx].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axs[idx].set_ylabel(f"Weighted mean change ({metric})")
        axs[idx].legend()

    axs[-1].set_xlabel("Year")
    fig.suptitle("Mediterranean Weighted Mean Change")
    plt.savefig(DirsLocal.figures / "hw_change_mediterranean_weighted_mean.pdf")
    sns.despine()
    plt.show()


def assert_expected_output(ds: xr.Dataset) -> None:
    """Validate output structure for exposure change."""
    expected_vars = {Vars.hw_days, Vars.hw_count}
    missing = expected_vars - set(ds.data_vars)
    if missing:
        raise ValueError(f"Missing expected variables in output: {missing}")

    expected_dims = {Vars.age_band, "latitude", "longitude", "year"}
    if set(ds.dims) != expected_dims:
        raise ValueError(
            f"Unexpected dims in output: {set(ds.dims)} (expected {expected_dims})"
        )


def plot_global_histograms(combined: xr.Dataset, year: int = 2020) -> None:
    """Plot distributions of exposure change for a given year."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    for row, metric in enumerate([Vars.hw_days, Vars.hw_count]):
        for col, population in enumerate([Vars.infants, Vars.over_65]):
            data = combined[metric].sel(age_band=population, year=year)
            data = data.where(np.isfinite(data)).values.ravel()
            data = data[~np.isnan(data)]

            axs[row, col].hist(data, bins=50, color="tab:blue", alpha=0.7)
            axs[row, col].set_title(f"{metric} - {population} ({year})")
            axs[row, col].set_xlabel("Change")
            axs[row, col].set_ylabel("Grid cells")

    plt.savefig(DirsLocal.figures / f"hw_change_histograms_{year}.pdf")
    sns.despine()
    plt.show()


def main() -> None:
    """Run exposure-to-change calculation and plotting locally."""
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

    logging.info("Building exposure change dataset...")
    combined = build_exposure_change_dataset(pop_inf, pop_old, hw_ds)

    ensure_directories([FilesLocal.hw_change_combined.parent, DirsLocal.figures])

    logging.info("Saving output: %s", FilesLocal.hw_change_combined)
    combined.to_netcdf(
        FilesLocal.hw_change_combined,
        encoding={
            Vars.hw_days: {"zlib": True, "complevel": 5},
            Vars.hw_count: {"zlib": True, "complevel": 5},
        },
    )

    combined = xr.open_dataset(FilesLocal.hw_change_combined)

    logging.info("Generating plots...")
    plot_weighted_mean_change(combined, pop_inf, pop_old)
    plot_total_exposure_change(combined)
    plot_mediterranean_change(combined, pop_inf, pop_old, year=2020)
    # plot_global_histograms(combined, year=2020)

    logging.info("✅ Done")


if __name__ == "__main__":
    pass
    # main()
