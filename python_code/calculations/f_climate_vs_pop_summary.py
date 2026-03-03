"""Concise summary of global heatwave exposure change driven by climate vs population.

Methodology
============
This script quantifies the contributions of climate change and population growth to changes in global heatwave exposure for two vulnerable groups (infants and older adults). The workflow is as follows:

1. Data Loading:
   - Loads gridded annual heatwave metrics (days) and population data for each group, all on the ERA5-Land grid.

2. Baseline and Target Periods:
   - For each baseline period (e.g., 1986-2005), calculates the mean heatwave days and mean population (the "baseline").
   - Compares this baseline to a target period, which can be either a single year (e.g., 2025) or a range of years (e.g., 2006-2025). By default, the target is the last year in the analysis.

3. Decomposition of Change:
   - For each group and baseline, decomposes the change in total person-days of heatwave exposure into:
     a. Climate Effect: Change in heatwave days, holding population fixed at the baseline mean.
     b. Population Effect: Change in population, holding heatwave days fixed at the baseline mean.
     c. Combined Effect: The sum of the above two effects (not including interaction terms).
   - Mathematically:
       - Climate effect = pop_base * (hw_days_target - hw_days_base)
       - Population effect = (pop_target - pop_base) * hw_days_base
       - Combined = climate effect + population effect

4. Aggregation:
   - Sums the results over all grid cells to obtain global totals for each effect, group, and baseline.

5. Output and Visualization:
   - Results are saved as a pandas DataFrame and visualized as bar plots (absolute and percent contributions).
   - Logging is configured to record the summary of results.

This approach allows attribution of changes in heatwave exposure to climate and demographic drivers, supporting robust global-scale analysis. The comparison is always between the mean of the baseline period and the mean (or value) of the specified target period (year or range).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from my_config import DirsLocal, FilesLocal, Vars, ensure_directories, Labels
from python_code.calculations.a_heatwave_exposure_pop_abs import (
    load_heatwave_metrics,
    load_population,
    resolve_hw_var,
    standardize_grid,
)
from python_code.log_config import setup_logging
from my_config import update_typst_json

logger = setup_logging(DirsLocal.project, "f_climate_vs_pop_summary.log")


def _sum_latlon(data: xr.DataArray) -> float:
    result = data.sum(dim=["latitude", "longitude"], skipna=True)
    # If result is a Dask array, compute it
    if hasattr(result, "compute"):
        result = result.compute()
    # Convert to scalar
    return float(result.values) if hasattr(result, "values") else float(result)


def _mean_over_period(data: xr.DataArray, year_slice: slice) -> xr.DataArray:
    return data.sel(year=year_slice).mean(dim="year")


def _select_target(data: xr.DataArray, target: int | slice) -> xr.DataArray:
    """Select a single year or mean over a range of years as the target."""
    if isinstance(target, slice):
        return data.sel(year=target).mean(dim="year")
    else:
        return data.sel(year=target)


def _build_record(
    baseline_label: str,
    age_label: str,
    climate_effect: float,
    population_effect: float,
    interaction_effect: float,
    combined_effect: float,
) -> dict:
    return {
        "baseline_label": baseline_label,
        "age_band": age_label,
        "climate": climate_effect,
        "population": population_effect,
        "interaction": interaction_effect,
        "combined": combined_effect,
    }


def calculate_global_contributions(target: int | slice | None = None) -> pd.DataFrame:
    if target is None:
        target = Vars.year_max_analysis

    years = slice(Vars.year_min_analysis, Vars.year_max_analysis)
    hw_ds = load_heatwave_metrics(years)
    hw_days = resolve_hw_var(hw_ds, Vars.hw_days)
    hw_days = standardize_grid(hw_days)

    pop_inf = load_population(FilesLocal.pop_inf, "pop", years)
    pop_old = load_population(FilesLocal.pop_over_65, "pop", years)
    pop_inf = standardize_grid(pop_inf)
    pop_old = standardize_grid(pop_old)

    records: list[dict] = []

    for baseline in Vars.get_baseline_periods():
        start, end = baseline
        label = Vars.format_baseline_period(baseline)
        climate_base = _mean_over_period(hw_days, slice(start, end))
        climate_target = _select_target(hw_days, target)

        for age_label, pop_da in ((Vars.infants, pop_inf), (Vars.over_65, pop_old)):
            logger.info(
                "Calculating contributions for baseline %s, age group %s, target %s",
                label,
                age_label,
                target,
            )
            pop_base = _mean_over_period(pop_da, slice(start, end))
            pop_target = _select_target(pop_da, target)
            logger.info(
                "Baseline climate (heatwave days) mean: %.2f", _sum_latlon(climate_base)
            )
            logger.info(
                "Target climate (heatwave days) mean: %.2f", _sum_latlon(climate_target)
            )
            logger.info(
                "Baseline population mean: %.2f (millions)", _sum_latlon(pop_base) / 1e6
            )
            logger.info(
                "Target population: %.2f (millions)", _sum_latlon(pop_target) / 1e6
            )

            climate_effect = pop_base * (climate_target - climate_base)
            population_effect = (pop_target - pop_base) * climate_base
            interaction_effect = (pop_target - pop_base) * (
                climate_target - climate_base
            )
            combined_effect = climate_effect + population_effect + interaction_effect

            records.append(
                _build_record(
                    label,
                    age_label,
                    _sum_latlon(climate_effect),
                    _sum_latlon(population_effect),
                    _sum_latlon(interaction_effect),
                    _sum_latlon(combined_effect),
                )
            )

    summary = pd.DataFrame(records)
    summary = summary.sort_values(["age_band", "baseline_label"])
    return summary


def log_summary(summary: pd.DataFrame) -> None:
    for _, group in summary.groupby("baseline_label"):
        logger.info("Baseline: %s", group["baseline_label"].iloc[0])
        for _, row in group.iterrows():
            logger.info(
                "  %s: climate %.0f, population %.0f, interaction %.0f, combined %.0f Millions of person-days",
                row["age_band"],
                row["climate"] / 1e6,
                row["population"] / 1e6,
                row["interaction"] / 1e6,
                row["combined"] / 1e6,
            )


def plot_contributions(summary: pd.DataFrame) -> None:
    ensure_directories([DirsLocal.figures])
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    age_order = [Vars.infants, Vars.over_65]

    for ax, age in zip(axs, age_order):
        sub = summary[summary["age_band"] == age]
        x = np.arange(len(sub))
        width = 0.25
        ax.bar(x - width, sub["climate"] / 1e9, width, label="Climate change")
        ax.bar(x, sub["population"] / 1e9, width, label="Population growth")
        ax.bar(
            x + width,
            sub["interaction"] / 1e9,
            width,
            label="Interaction",
            color="tab:purple",
        )
        ax.set_title(Labels.get_label(age))
        ax.set_xticks(x)
        ax.set_xticklabels(sub["baseline_label"])
        ax.set_ylabel(f"{Labels.person_days} (billions)")
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.legend()
        ax.grid(False, axis="x")
        ax.set_xlabel("Baseline period")

    fig.tight_layout()
    fig.savefig(DirsLocal.figures / "f_climate_vs_pop_contributions.pdf")
    plt.show()

    # summary_copy = summary.copy()
    # summary_copy["total_abs"] = (
    #     summary_copy[["climate", "population", "interaction"]].abs().sum(axis=1)
    # )
    # for col in ["climate", "population", "interaction"]:
    #     summary_copy[f"{col}_pct"] = (
    #         summary_copy[col].abs() / summary_copy["total_abs"].replace(0, np.nan)
    #     ) * 100
    #
    # fig2, axs2 = plt.subplots(1, 2, sharey=True)
    # for ax, age in zip(axs2, age_order):
    #     sub = summary_copy[summary_copy["age_band"] == age]
    #     x = np.arange(len(sub))
    #
    #     # Bar 1: Climate Change
    #     ax.barh(x, sub["climate_pct"], label="Climate change", color="tab:red")
    #
    #     # Bar 2: Population Growth (Starts where Climate ends)
    #     ax.barh(
    #         x,
    #         sub["population_pct"],
    #         left=sub["climate_pct"],
    #         label="Population growth",
    #         color="tab:blue",
    #     )
    #
    #     # Bar 3: Interaction (Starts where Climate + Population ends)
    #     ax.barh(
    #         x,
    #         sub["interaction_pct"],
    #         left=sub["climate_pct"] + sub["population_pct"],
    #         label="Interaction",
    #         color="tab:purple",  # You can change this color to fit your report's palette
    #     )
    #
    #     ax.set_yticks(x)
    #     ax.set_yticklabels(sub["baseline_label"])
    #
    #     # Note: If your Labels dictionary has a nice string for age, use Labels.get_label(age)
    #     ax.set_title(Labels.get_label(age) if hasattr(Labels, "get_label") else age)
    #
    #     if ax is axs2[0]:
    #         ax.set_xlabel("Percent of absolute contribution")
    #     ax.legend()
    #     ax.grid(False, axis="y")
    #
    # fig2.tight_layout()
    # fig2.savefig(DirsLocal.figures / "f_climate_vs_pop_shares.pdf")
    # plt.show()


def export_to_typst(summary: pd.DataFrame, target: int | slice | None = None):
    """
    Export the summary DataFrame to Typst JSON using update_typst_json.
    The exported keys will be under 'climate_vs_pop_summary'.
    """
    # Prepare a nested dict for Typst export
    typst_data = {}
    for _, row in summary.iterrows():
        baseline = row["baseline_label"]
        age = row["age_band"]
        if baseline not in typst_data:
            typst_data[baseline] = {}
        typst_data[baseline][age] = {
            "climate": row["climate"] / 1e9,
            "climate_pct": row["climate"] / row["combined"] * 100,
            "population": row["population"] / 1e9,
            "population_pct": row["population"] / row["combined"] * 100,
            "interaction": row["interaction"] / 1e9,
            "interaction_pct": row["interaction"] / row["combined"] * 100,
            "combined": row["combined"] / 1e9,
        }
    # Add info about the target period
    typst_data["target"] = str(target) if target is not None else "default"
    update_typst_json({"climate_vs_pop_summary": typst_data})
    logger.info("Exported climate_vs_pop_summary to Typst JSON.")


def main(target: int | slice | None = None):
    """
    Run the summary calculation and plotting.
    Args:
        target: int (year), slice (range of years), or None (default: last year in analysis)
    """
    summary = calculate_global_contributions(target)
    log_summary(summary)
    plot_contributions(summary)
    export_to_typst(summary, target)


if __name__ == "__main__":
    # Example usage:
    # main(2025)  # Compare baseline to 2025
    # main(slice(2006, 2025))  # Compare baseline to mean of 2006-2025
    main(target=Vars.year_max_analysis)  # Default: compare to last year in analysis
