"""
Calculate exposure to change in heatwave occurrence relative to the reference period.

Outputs a single NetCDF with dimensions:
- age_band, baseline_period, latitude, longitude, year
and variables:
- heatwave_days, heatwave_counts
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from my_config import (
    DirsLocal,
    FilesLocal,
    Vars,
    ensure_directories,
    Labels,
    update_typst_json,
)
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


def assert_reference_years_available(
    hw_da: xr.DataArray, reference_period: tuple[int, int]
) -> None:
    """Ensure a reference period is fully present in the heatwave data."""
    ref_years = np.arange(reference_period[0], reference_period[1] + 1)
    available_years = hw_da["year"].values
    missing = sorted(set(ref_years) - set(available_years))
    if missing:
        raise ValueError(
            f"Missing reference years in heatwave data for baseline "
            f"{Vars.format_baseline_period(reference_period)}: {missing}"
        )


def compute_reference_mean(
    hw_da: xr.DataArray, reference_period: tuple[int, int]
) -> xr.DataArray:
    """Compute mean for a specific baseline period."""
    assert_reference_years_available(hw_da, reference_period)
    return hw_da.sel(year=slice(reference_period[0], reference_period[1])).mean(
        dim="year"
    )


def compute_delta(
    hw_da: xr.DataArray, reference_period: tuple[int, int]
) -> xr.DataArray:
    """Compute anomaly relative to a specific baseline-period mean."""
    ref_mean = compute_reference_mean(hw_da, reference_period)
    return hw_da - ref_mean


def _ensure_baseline_dimension(ds: xr.Dataset) -> xr.Dataset:
    """Add a default baseline dimension if loading an older single-baseline file."""
    if Vars.baseline_period in ds.dims:
        return ds

    default_baseline = Vars.format_baseline_period(
        (Vars.year_reference_start, Vars.year_reference_end)
    )
    return ds.expand_dims({Vars.baseline_period: [default_baseline]})


def build_exposure_change_dataset(
    pop_inf: xr.DataArray,
    pop_old: xr.DataArray,
    hw_ds: xr.Dataset,
    baseline_periods: list[tuple[int, int]] | None = None,
) -> xr.Dataset:
    """Build exposure-to-change dataset for infants and over_65 across baselines."""
    if baseline_periods is None:
        baseline_periods = Vars.get_baseline_periods()
    if not baseline_periods:
        raise ValueError("At least one baseline period is required.")

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

    combined_baselines = []
    for baseline_period in baseline_periods:
        baseline_label = Vars.format_baseline_period(baseline_period)
        logging.info("Computing deltas for baseline %s...", baseline_label)

        days_delta = compute_delta(hw_days, baseline_period)
        counts_delta = compute_delta(hw_counts, baseline_period)

        logging.info(
            "Calculating exposure change for baseline %s and age group %s...",
            baseline_label,
            Vars.infants,
        )
        days_inf = calculate_exposure(pop_inf, days_delta)
        counts_inf = calculate_exposure(pop_inf, counts_delta)

        logging.info(
            "Calculating exposure change for baseline %s and age group %s...",
            baseline_label,
            Vars.over_65,
        )
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
        combined = combined.expand_dims({Vars.baseline_period: [baseline_label]})
        combined_baselines.append(combined)

    combined = xr.concat(combined_baselines, dim=Vars.baseline_period)
    combined = combined.transpose(
        Vars.age_band, Vars.baseline_period, "latitude", "longitude", "year"
    )
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
    """Plot population-weighted mean exposure change for each baseline with a shared Y label."""
    baseline_labels = combined[Vars.baseline_period].values
    fig, axs = plt.subplots(
        len(baseline_labels),
        1,
        constrained_layout=True,
        sharex=True,
        figsize=(8, 3 * len(baseline_labels)),
    )
    if len(baseline_labels) == 1:
        axs = [axs]

    for ax, baseline in zip(axs, baseline_labels):
        ds_hw_days = combined[Vars.hw_days].sel({Vars.baseline_period: baseline})
        for data, pop in zip([pop_inf, pop_old], [Vars.infants, Vars.over_65]):
            days = weighted_mean(ds_hw_days.sel(age_band=pop), data)
            ax.plot(
                days["year"].values,
                days,
                label=Labels.get_label(pop),
                marker="o",
            )
            update_typst_json(
                {
                    "hw_change": {
                        "avg": {
                            str(baseline): {
                                pop: {
                                    str(year): day.item()
                                    for year, day in zip(
                                        days["year"].values, days.values
                                    )
                                }
                            }
                        }
                    }
                }
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend()
        ax.set_title(f"Baseline: {baseline}")

    # Add a shared Y label for all subplots
    fig.text(
        0,
        0.5,
        "Pop.-weighted mean change in heatwave days",
        va="center",
        ha="right",
        size=12,
        rotation="vertical",
    )
    axs[-1].set_xlabel("Year")
    sns.despine()
    plt.savefig(DirsLocal.figures / "hw_change_weighted_mean_by_baseline.pdf")
    plt.show()


def plot_total_exposure_change(combined: xr.Dataset) -> None:
    """Plot total exposure change over time for each baseline."""
    baseline_labels = combined[Vars.baseline_period].values
    fig, axs = plt.subplots(
        len(baseline_labels),
        1,
        constrained_layout=True,
        sharex=True,
        figsize=(8, 3 * len(baseline_labels)),
    )
    if len(baseline_labels) == 1:
        axs = [axs]

    for ax, baseline in zip(axs, baseline_labels):
        ds_hw_days = combined[Vars.hw_days].sel({Vars.baseline_period: baseline})
        for age_band in [Vars.infants, Vars.over_65]:
            data = ds_hw_days.sel(age_band=age_band).sum(
                dim=["latitude", "longitude"], skipna=True
            )
            ax.plot(
                data["year"].values,
                data / 1e9,
                label=Labels.get_label(age_band),
                marker="o",
            )
            update_typst_json(
                {
                    "hw_change": {
                        "total": {
                            str(baseline): {
                                age_band: {
                                    str(year): day.item() / 1e9
                                    for year, day in zip(
                                        data["year"].values, data.values
                                    )
                                }
                            }
                        }
                    }
                }
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend()
        ax.set_title(f"Baseline: {baseline}")

    fig.text(
        0,
        0.5,
        "Total change in heatwave days (billions)",
        va="center",
        ha="right",
        size=12,
        rotation="vertical",
    )
    axs[-1].set_xlabel("Year")
    sns.despine()
    plt.savefig(DirsLocal.figures / "hw_change_total_exposure_by_baseline.pdf")
    plt.show()


def assert_expected_output(ds: xr.Dataset) -> None:
    """Validate output structure for exposure change."""
    expected_vars = {Vars.hw_days, Vars.hw_count}
    missing = expected_vars - set(ds.data_vars)
    if missing:
        raise ValueError(f"Missing expected variables in output: {missing}")

    expected_dims = {
        Vars.age_band,
        Vars.baseline_period,
        "latitude",
        "longitude",
        "year",
    }
    if set(ds.dims) != expected_dims:
        raise ValueError(
            f"Unexpected dims in output: {set(ds.dims)} (expected {expected_dims})"
        )


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
    baseline_periods = Vars.get_baseline_periods()
    logging.info(
        "Using baseline periods: %s",
        ", ".join(Vars.format_baseline_period(period) for period in baseline_periods),
    )
    combined = build_exposure_change_dataset(
        pop_inf,
        pop_old,
        hw_ds,
        baseline_periods=baseline_periods,
    )
    assert_expected_output(combined)

    ensure_directories([FilesLocal.hw_change_combined.parent, DirsLocal.figures])

    logging.info("Saving output: %s", FilesLocal.hw_change_combined)
    combined.to_netcdf(
        FilesLocal.hw_change_combined,
        encoding={
            Vars.hw_days: {"zlib": True, "complevel": 0},
            Vars.hw_count: {"zlib": True, "complevel": 0},
        },
    )


def plot():
    setup_logging()
    combined = xr.open_dataset(FilesLocal.hw_change_combined)
    combined = _ensure_baseline_dimension(combined)

    years = slice(Vars.year_min_analysis, Vars.year_max_analysis)

    logging.info("Loading population data...")
    pop_inf = load_population(FilesLocal.pop_inf, "pop", years)
    pop_old = load_population(FilesLocal.pop_over_65, "pop", years)

    logging.info("Generating plots...")
    plot_weighted_mean_change(combined=combined, pop_inf=pop_inf, pop_old=pop_old)
    plot_total_exposure_change(combined)

    logging.info("✅ Done")


if __name__ == "__main__":
    pass
    # main()
    plot()
