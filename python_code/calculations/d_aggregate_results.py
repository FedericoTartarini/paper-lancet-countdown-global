"""
Aggregate ERA5-Land heatwave exposure results by country and region groupings.

This script reads:
- combined absolute exposure dataset (person-days and person-events)
- combined exposure-change dataset (person-days change)

It aggregates totals and per-person values for each region defined by raster masks.
Outputs are saved as NetCDF files per grouping in the aggregates folder.

Run locally:
    python python_code/calculations/d_aggregate_results.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import seaborn as sns
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from my_config import (
    DirsLocal,
    FilesLocal,
    Vars,
    ensure_directories,
    Labels,
)
from python_code.calculations.a_heatwave_exposure_pop_abs import (
    assert_matching_grid,
    load_population,
    standardize_grid,
)

SHEET_NAME = "ISO3 - Name - LC - WHO - HDI"
CHANGE_TOTAL_VAR = f"{Vars.hw_days}_change_total"
CHANGE_PER_PERSON_VAR = f"{Vars.hw_days}_change_per_person"


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_exposure_dataset() -> xr.Dataset:
    """Load combined absolute exposure dataset (all ages)."""
    if not FilesLocal.hw_combined_q.exists():
        raise FileNotFoundError(f"Exposure file not found: {FilesLocal.hw_combined_q}")

    ds = xr.open_dataset(FilesLocal.hw_combined_q)
    ds = standardize_grid(ds)

    missing = [v for v in [Vars.hw_days, Vars.hw_count] if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Missing expected exposure variables: {missing}. Found: {list(ds.data_vars)}"
        )

    if Vars.age_band not in ds.dims:
        raise ValueError(f"Exposure dataset is missing '{Vars.age_band}' dimension.")

    return ds


def _ensure_baseline_dimension(ds: xr.Dataset) -> xr.Dataset:
    """Add default baseline period for backwards compatibility."""
    if Vars.baseline_period in ds.dims:
        return ds

    default_baseline = Vars.format_baseline_period(
        (Vars.year_reference_start, Vars.year_reference_end)
    )
    return ds.expand_dims({Vars.baseline_period: [default_baseline]})


def load_exposure_change_dataset() -> xr.Dataset:
    """Load combined exposure-change dataset (all ages, all baselines)."""
    if not FilesLocal.hw_change_combined.exists():
        raise FileNotFoundError(
            f"Exposure change file not found: {FilesLocal.hw_change_combined}"
        )

    ds = xr.open_dataset(FilesLocal.hw_change_combined)
    ds = standardize_grid(ds)
    ds = _ensure_baseline_dimension(ds)

    if Vars.hw_days not in ds.data_vars:
        raise ValueError(
            f"Missing expected change variable '{Vars.hw_days}'. "
            f"Found: {list(ds.data_vars)}"
        )
    if Vars.age_band not in ds.dims:
        raise ValueError(
            f"Exposure change dataset is missing '{Vars.age_band}' dimension."
        )
    if Vars.baseline_period not in ds.dims:
        raise ValueError(
            f"Exposure change dataset is missing '{Vars.baseline_period}' dimension."
        )

    return ds


def load_population_dataset() -> xr.DataArray:
    """Load population data and combine age bands to match exposure dataset."""
    pop_inf = load_population(FilesLocal.pop_inf, name_hint="pop", years=None)
    pop_old = load_population(FilesLocal.pop_over_65, name_hint="pop", years=None)

    pop_inf = standardize_grid(pop_inf)
    pop_old = standardize_grid(pop_old)

    pop = xr.concat(
        [pop_inf, pop_old],
        dim=pd.Index([Vars.infants, Vars.over_65], name=Vars.age_band),
    )
    pop.name = "population"
    return pop


def load_country_groupings() -> pd.DataFrame:
    """Load Lancet country groupings for HDI and LC mapping."""
    if not FilesLocal.country_names_groupings.exists():
        raise FileNotFoundError(
            f"Grouping file not found: {FilesLocal.country_names_groupings}"
        )
    return pd.read_excel(FilesLocal.country_names_groupings, sheet_name=SHEET_NAME)


def load_shapefile_iso3() -> set[str]:
    """Load ISO3 codes available in the shapefile."""
    if not FilesLocal.world_bank_shapefile.exists():
        raise FileNotFoundError(
            f"Shapefile not found: {FilesLocal.world_bank_shapefile}"
        )
    gdf = gpd.read_file(FilesLocal.world_bank_shapefile)
    iso_col = resolve_column(gdf, ["ISO_A3", "ISO3", "ISO_3_CODE"], "ISO3")
    iso3 = gdf[iso_col].dropna().astype(str)
    iso3 = iso3[iso3.str.len() == 3]
    return set(iso3.unique())


def filter_groupings_to_shapefile(groupings: pd.DataFrame) -> pd.DataFrame:
    """Filter groupings to ISO3 codes present in the shapefile."""
    iso3_set = load_shapefile_iso3()
    all_iso3 = set(groupings["ISO3"].dropna().astype(str))
    dropped = sorted(all_iso3 - iso3_set)
    if dropped:
        logging.warning(
            "Dropping %s ISO3 codes not in shapefile: %s",
            len(dropped),
            ", ".join(dropped),
        )

    filtered = groupings[groupings["ISO3"].isin(iso3_set)].copy()
    if filtered.empty:
        raise ValueError("No ISO3 codes overlap between groupings and shapefile.")
    return filtered


def build_id_map(values: pd.Series) -> dict[str, int]:
    """Build a stable id mapping from sorted unique values."""
    unique_vals = sorted(v for v in values.dropna().unique())
    return {value: idx + 1 for idx, value in enumerate(unique_vals)}


def build_region_table(
    regions_df: pd.DataFrame,
    id_col: str,
    name_col: str,
) -> pd.DataFrame:
    """Return a de-duplicated table of region IDs and names."""
    if id_col not in regions_df.columns or name_col not in regions_df.columns:
        raise ValueError(
            f"Missing columns in regions table. Expected {id_col}, {name_col}."
        )
    table = regions_df[[id_col, name_col]].drop_duplicates()
    table = table[table[id_col].notna()]
    return table


def resolve_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Resolve a column name from candidates in a DataFrame."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find {label} column in {list(df.columns)}")


def resolve_mask_var(mask_ds: xr.Dataset, candidates: list[str], label: str) -> str:
    """Resolve a mask variable name from candidates in a Dataset."""
    for candidate in candidates:
        if candidate in mask_ds.data_vars:
            return candidate
    raise ValueError(f"Could not find {label} in {list(mask_ds.data_vars)}")


def plot_aggregate_summary(
    agg_ds: xr.Dataset,
    region_dim: str,
    output_prefix: str,
) -> None:
    """Plot total exposure by year and top regions for each aggregate."""
    ensure_directories([DirsLocal.aggregates_figures])

    totals = agg_ds[[f"{Vars.hw_days}_total", f"{Vars.hw_count}_total"]]

    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for idx, metric in enumerate([Vars.hw_days, Vars.hw_count]):
        data = totals[f"{metric}_total"].sum(dim=region_dim)
        for age_band in data[Vars.age_band].values:
            series = data.sel({Vars.age_band: age_band})
            axs[idx].plot(series["year"].values, series, label=str(age_band))
        axs[idx].set_ylabel(f"Total {metric} exposure")
        axs[idx].legend(title=Vars.age_band)
    axs[-1].set_xlabel("Year")
    fig.suptitle(f"Global totals ({region_dim})")
    fig.savefig(DirsLocal.aggregates_figures / f"{output_prefix}_totals.png")
    plt.close(fig)

    latest_year = int(agg_ds["year"].max())
    top_metric = totals[f"{Vars.hw_days}_total"].sel(year=latest_year)
    top_metric = top_metric.sum(dim=Vars.age_band)
    top_metric = top_metric.sortby(top_metric, ascending=False)
    top_metric = top_metric.isel({region_dim: slice(0, 10)})

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(top_metric[region_dim].values, top_metric.values)
    ax.set_title(f"Top 10 regions by heatwave days exposure ({latest_year})")
    ax.set_ylabel("Exposure")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(DirsLocal.aggregates_figures / f"{output_prefix}_top10.png")
    plt.close(fig)


def aggregate_by_region(
    exposure: xr.Dataset,
    population: xr.DataArray,
    mask_ds: xr.Dataset,
    mask_var: str,
    region_table: pd.DataFrame,
    region_id_col: str,
    region_name_col: str,
    region_dim: str,
    output_path: Path,
) -> xr.Dataset:
    """Aggregate exposure totals and per-person values for each region."""
    if mask_var not in mask_ds.data_vars:
        raise ValueError(
            f"Mask variable '{mask_var}' not found. Available: {list(mask_ds.data_vars)}"
        )

    mask_da = standardize_grid(mask_ds[mask_var])
    exposure = standardize_grid(exposure)
    population = standardize_grid(population)

    assert_matching_grid(population, exposure[Vars.hw_days])
    assert_matching_grid(population, mask_da)

    region_ids = region_table[region_id_col].to_numpy()
    region_names = region_table[region_name_col].tolist()
    if len(region_ids) != len(region_names):
        raise ValueError("Region table has mismatched id/name lengths.")

    mask_da = mask_da.where(mask_da.isin(region_ids)).rename("region_id")

    exposure_masked = exposure.where(mask_da.notnull())
    population_masked = population.where(mask_da.notnull())

    sum_population = population_masked.groupby(mask_da).sum(
        dim=["latitude", "longitude"], skipna=True
    )
    sum_exposure = exposure_masked.groupby(mask_da).sum(
        dim=["latitude", "longitude"], skipna=True
    )

    sum_population = sum_population.reindex({"region_id": region_ids})
    sum_exposure = sum_exposure.reindex({"region_id": region_ids})

    sum_population = sum_population.rename({"region_id": region_dim})
    sum_exposure = sum_exposure.rename({"region_id": region_dim})

    sum_population = sum_population.assign_coords({region_dim: region_names})
    sum_exposure = sum_exposure.assign_coords({region_dim: region_names})

    region_weighted = sum_exposure / sum_population

    aggregated = xr.Dataset(
        {
            "population": sum_population,
            f"{Vars.hw_days}_total": sum_exposure[Vars.hw_days],
            f"{Vars.hw_count}_total": sum_exposure[Vars.hw_count],
            f"{Vars.hw_days}_per_person": region_weighted[Vars.hw_days],
            f"{Vars.hw_count}_per_person": region_weighted[Vars.hw_count],
        }
    )

    output_path.unlink(missing_ok=True)
    aggregated.to_netcdf(output_path)
    return aggregated


def aggregate_change_by_region(
    exposure_days_change: xr.DataArray,
    population: xr.DataArray,
    mask_ds: xr.Dataset,
    mask_var: str,
    region_table: pd.DataFrame,
    region_id_col: str,
    region_name_col: str,
    region_dim: str,
    output_path: Path,
) -> xr.Dataset:
    """Aggregate heatwave-days change totals and per-person values for each region."""
    if mask_var not in mask_ds.data_vars:
        raise ValueError(
            f"Mask variable '{mask_var}' not found. Available: {list(mask_ds.data_vars)}"
        )

    mask_da = standardize_grid(mask_ds[mask_var])
    exposure_days_change = standardize_grid(exposure_days_change)
    population = standardize_grid(population)

    assert_matching_grid(population, exposure_days_change)
    assert_matching_grid(population, mask_da)

    region_ids = region_table[region_id_col].to_numpy()
    region_names = region_table[region_name_col].tolist()
    if len(region_ids) != len(region_names):
        raise ValueError("Region table has mismatched id/name lengths.")

    mask_da = mask_da.where(mask_da.isin(region_ids)).rename("region_id")

    exposure_masked = exposure_days_change.where(mask_da.notnull())
    population_masked = population.where(mask_da.notnull())

    sum_population = population_masked.groupby(mask_da).sum(
        dim=["latitude", "longitude"], skipna=True
    )
    sum_exposure_change = exposure_masked.groupby(mask_da).sum(
        dim=["latitude", "longitude"], skipna=True
    )

    sum_population = sum_population.reindex({"region_id": region_ids})
    sum_exposure_change = sum_exposure_change.reindex({"region_id": region_ids})

    sum_population = sum_population.rename({"region_id": region_dim})
    sum_exposure_change = sum_exposure_change.rename({"region_id": region_dim})

    sum_population = sum_population.assign_coords({region_dim: region_names})
    sum_exposure_change = sum_exposure_change.assign_coords({region_dim: region_names})

    change_per_person = sum_exposure_change / sum_population

    aggregated_change = xr.Dataset(
        {
            "population": sum_population,
            CHANGE_TOTAL_VAR: sum_exposure_change,
            CHANGE_PER_PERSON_VAR: change_per_person,
        }
    )

    output_path.unlink(missing_ok=True)
    aggregated_change.to_netcdf(output_path)
    return aggregated_change


def aggregate_by_country(
    exposure: xr.Dataset,
    population: xr.DataArray,
    groupings: pd.DataFrame,
) -> None:
    """Aggregate exposure results by country."""
    mask_ds = xr.open_dataset(FilesLocal.raster_country)
    mask_var = "country_id"

    groupings = groupings[groupings["ISO3"].notna()].copy()
    country_ids = build_id_map(groupings["ISO3"])

    region_table = pd.DataFrame(
        {
            "region_id": [country_ids[iso] for iso in sorted(country_ids)],
            "region_name": [
                groupings.loc[groupings["ISO3"] == iso, "Country Name to use"].iloc[0]
                for iso in sorted(country_ids)
            ],
        }
    )

    aggregate_by_region(
        exposure=exposure,
        population=population,
        mask_ds=mask_ds,
        mask_var=mask_var,
        region_table=region_table,
        region_id_col="region_id",
        region_name_col="region_name",
        region_dim="country",
        output_path=FilesLocal.aggregate_country,
    )


def aggregate_by_who_region(
    exposure: xr.Dataset,
    population: xr.DataArray,
    groupings: pd.DataFrame,
) -> None:
    """Aggregate exposure results by WHO region."""
    mask_ds = xr.open_dataset(FilesLocal.raster_who)
    mask_var = "who_id"

    who_ids = build_id_map(groupings["WHO Region"])
    region_table = pd.DataFrame(
        {
            "region_id": [who_ids[name] for name in sorted(who_ids)],
            "region_name": sorted(who_ids),
        }
    )

    aggregate_by_region(
        exposure=exposure,
        population=population,
        mask_ds=mask_ds,
        mask_var=mask_var,
        region_table=region_table,
        region_id_col="region_id",
        region_name_col="region_name",
        region_dim="who_region",
        output_path=FilesLocal.aggregate_who,
    )


def aggregate_by_hdi(
    exposure: xr.Dataset,
    population: xr.DataArray,
    groupings: pd.DataFrame,
) -> None:
    """Aggregate exposure results by HDI grouping."""
    mask_ds = xr.open_dataset(FilesLocal.raster_hdi)
    mask_var = "hdi_id"

    hdi_ids = build_id_map(groupings["HDI Group 2025"])
    region_table = pd.DataFrame(
        {
            "region_id": [hdi_ids[name] for name in sorted(hdi_ids)],
            "region_name": sorted(hdi_ids),
        }
    )

    aggregate_by_region(
        exposure=exposure,
        population=population,
        mask_ds=mask_ds,
        mask_var=mask_var,
        region_table=region_table,
        region_id_col="region_id",
        region_name_col="region_name",
        region_dim="hdi_group",
        output_path=FilesLocal.aggregate_hdi,
    )


def aggregate_by_lancet_grouping(
    exposure: xr.Dataset,
    population: xr.DataArray,
    groupings: pd.DataFrame,
) -> None:
    """Aggregate exposure results by Lancet grouping."""
    mask_ds = xr.open_dataset(FilesLocal.raster_lancet)
    mask_var = "lc_id"

    lc_ids = build_id_map(groupings["LC Grouping"])
    region_table = pd.DataFrame(
        {
            "region_id": [lc_ids[name] for name in sorted(lc_ids)],
            "region_name": sorted(lc_ids),
        }
    )

    aggregate_by_region(
        exposure=exposure,
        population=population,
        mask_ds=mask_ds,
        mask_var=mask_var,
        region_table=region_table,
        region_id_col="region_id",
        region_name_col="region_name",
        region_dim="lc_group",
        output_path=FilesLocal.aggregate_lancet,
    )


def aggregate_change_by_country(
    exposure_change: xr.Dataset,
    population: xr.DataArray,
    groupings: pd.DataFrame,
) -> None:
    """Aggregate exposure change (heatwave person-days only) by country."""
    mask_ds = xr.open_dataset(FilesLocal.raster_country)
    mask_var = "country_id"

    groupings = groupings[groupings["ISO3"].notna()].copy()
    country_ids = build_id_map(groupings["ISO3"])

    region_table = pd.DataFrame(
        {
            "region_id": [country_ids[iso] for iso in sorted(country_ids)],
            "region_name": [
                groupings.loc[groupings["ISO3"] == iso, "Country Name to use"].iloc[0]
                for iso in sorted(country_ids)
            ],
        }
    )

    aggregate_change_by_region(
        exposure_days_change=exposure_change[Vars.hw_days],
        population=population,
        mask_ds=mask_ds,
        mask_var=mask_var,
        region_table=region_table,
        region_id_col="region_id",
        region_name_col="region_name",
        region_dim="country",
        output_path=FilesLocal.aggregate_country_change,
    )


def aggregate_change_by_who_region(
    exposure_change: xr.Dataset,
    population: xr.DataArray,
    groupings: pd.DataFrame,
) -> None:
    """Aggregate exposure change (heatwave person-days only) by WHO region."""
    mask_ds = xr.open_dataset(FilesLocal.raster_who)
    mask_var = "who_id"

    who_ids = build_id_map(groupings["WHO Region"])
    region_table = pd.DataFrame(
        {
            "region_id": [who_ids[name] for name in sorted(who_ids)],
            "region_name": sorted(who_ids),
        }
    )

    aggregate_change_by_region(
        exposure_days_change=exposure_change[Vars.hw_days],
        population=population,
        mask_ds=mask_ds,
        mask_var=mask_var,
        region_table=region_table,
        region_id_col="region_id",
        region_name_col="region_name",
        region_dim="who_region",
        output_path=FilesLocal.aggregate_who_change,
    )


def aggregate_change_by_hdi(
    exposure_change: xr.Dataset,
    population: xr.DataArray,
    groupings: pd.DataFrame,
) -> None:
    """Aggregate exposure change (heatwave person-days only) by HDI grouping."""
    mask_ds = xr.open_dataset(FilesLocal.raster_hdi)
    mask_var = "hdi_id"

    hdi_ids = build_id_map(groupings["HDI Group 2025"])
    region_table = pd.DataFrame(
        {
            "region_id": [hdi_ids[name] for name in sorted(hdi_ids)],
            "region_name": sorted(hdi_ids),
        }
    )

    aggregate_change_by_region(
        exposure_days_change=exposure_change[Vars.hw_days],
        population=population,
        mask_ds=mask_ds,
        mask_var=mask_var,
        region_table=region_table,
        region_id_col="region_id",
        region_name_col="region_name",
        region_dim="hdi_group",
        output_path=FilesLocal.aggregate_hdi_change,
    )


def aggregate_change_by_lancet_grouping(
    exposure_change: xr.Dataset,
    population: xr.DataArray,
    groupings: pd.DataFrame,
) -> None:
    """Aggregate exposure change (heatwave person-days only) by Lancet grouping."""
    mask_ds = xr.open_dataset(FilesLocal.raster_lancet)
    mask_var = "lc_id"

    lc_ids = build_id_map(groupings["LC Grouping"])
    region_table = pd.DataFrame(
        {
            "region_id": [lc_ids[name] for name in sorted(lc_ids)],
            "region_name": sorted(lc_ids),
        }
    )

    aggregate_change_by_region(
        exposure_days_change=exposure_change[Vars.hw_days],
        population=population,
        mask_ds=mask_ds,
        mask_var=mask_var,
        region_table=region_table,
        region_id_col="region_id",
        region_name_col="region_name",
        region_dim="lc_group",
        output_path=FilesLocal.aggregate_lancet_change,
    )


def build_excel_table(agg_ds: xr.Dataset, region_dim: str) -> pd.DataFrame:
    """Return a long-form table with population and exposure metrics."""
    return (
        agg_ds[
            [
                "population",
                f"{Vars.hw_days}_total",
                f"{Vars.hw_count}_total",
                f"{Vars.hw_days}_per_person",
                f"{Vars.hw_count}_per_person",
            ]
        ]
        .to_dataframe()
        .reset_index()
        .rename(columns={region_dim: "region"})
    )


def build_excel_table_change(agg_ds: xr.Dataset, region_dim: str) -> pd.DataFrame:
    """Return a long-form table for heatwave person-days change metrics."""
    return (
        agg_ds[
            [
                "population",
                CHANGE_TOTAL_VAR,
                CHANGE_PER_PERSON_VAR,
            ]
        ]
        .to_dataframe()
        .reset_index()
        .rename(columns={region_dim: "region"})
    )


def build_global_summary(
    population: xr.DataArray,
    heatwave_days: xr.DataArray,
) -> pd.DataFrame:
    """Build a global summary table for total population and heatwave days."""
    population_total = population.sum(dim=["latitude", "longitude"], skipna=True)
    population_total = population_total.sum(dim=Vars.age_band)

    if np.issubdtype(heatwave_days.dtype, np.timedelta64):
        heatwave_days = heatwave_days / np.timedelta64(1, "D")

    heatwave_total = heatwave_days.sum(dim=["latitude", "longitude"], skipna=True)

    return pd.DataFrame(
        {
            "year": population_total["year"].values,
            "population_total": population_total.values,
            "heatwave_days_total": heatwave_total.values,
        }
    )


def export_excel(datasets, change_datasets: dict | None = None) -> None:
    """Export aggregate tables and global summary to Excel."""
    output_path = FilesLocal.aggregate_submission
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        writer = pd.ExcelWriter(
            output_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        )
    else:
        writer = pd.ExcelWriter(output_path, engine="openpyxl", mode="w")

    with writer:
        for sheet_name, (region_dim, ds) in datasets.items():
            table = build_excel_table(ds, region_dim)
            table.to_excel(writer, sheet_name=sheet_name, index=False)

        if change_datasets:
            for sheet_name, (region_dim, ds) in change_datasets.items():
                table = build_excel_table_change(ds, region_dim)
                table.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    """Run all aggregations for absolute exposure and exposure change results."""
    setup_logging()
    ensure_directories([DirsLocal.aggregates, DirsLocal.aggregates_figures])

    logging.info("Loading groupings table...")
    groupings = load_country_groupings()
    groupings = filter_groupings_to_shapefile(groupings)

    logging.info("Loading exposure dataset...")
    exposure = load_exposure_dataset()

    logging.info("Loading population dataset...")
    population = load_population_dataset()

    logging.info("Aggregating by country...")
    aggregate_by_country(exposure, population, groupings)

    logging.info("Aggregating by WHO region...")
    aggregate_by_who_region(exposure, population, groupings)

    logging.info("Aggregating by HDI...")
    aggregate_by_hdi(exposure, population, groupings)

    logging.info("Aggregating by Lancet grouping...")
    aggregate_by_lancet_grouping(exposure, population, groupings)

    logging.info("Loading exposure change dataset...")
    exposure_change = load_exposure_change_dataset()

    logging.info("Aggregating exposure change by country...")
    aggregate_change_by_country(exposure_change, population, groupings)

    logging.info("Aggregating exposure change by WHO region...")
    aggregate_change_by_who_region(exposure_change, population, groupings)

    logging.info("Aggregating exposure change by HDI...")
    aggregate_change_by_hdi(exposure_change, population, groupings)

    logging.info("Aggregating exposure change by Lancet grouping...")
    aggregate_change_by_lancet_grouping(exposure_change, population, groupings)


def plot_trends_by(data, region_dim: str, num_regions: int) -> None:
    data = data.copy().to_dataframe().reset_index()
    f, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 4.5))
    country_color_map = {}
    if num_regions >= 5:
        country_colors = sns.color_palette("tab20")
    else:
        country_colors = sns.color_palette("tab10")
    for ix, age_band in enumerate(data[Vars.age_band].unique()):
        subset = data[data[Vars.age_band] == age_band]
        total_by_year = subset.groupby("year")[[f"{Vars.hw_days}_total"]].sum()
        sns.barplot(
            x=total_by_year.index.astype(str),
            y=f"{Vars.hw_days}_total",
            data=total_by_year / 1e9,
            ax=axs[ix],
            color="gray",
            label="Global Total",
        )
        top_countries = (
            subset.groupby(region_dim)[[f"{Vars.hw_days}_total"]]
            .sum()
            .sort_values(f"{Vars.hw_days}_total", ascending=False)
            .head(num_regions)
            .index
        )
        baseline = total_by_year[f"{Vars.hw_days}_total"].values * 0
        for ix_country, country in enumerate(top_countries):
            if country not in country_color_map.keys():
                country_color_map[country] = country_colors[
                    ix_country + num_regions * ix
                ]
            country_data = (
                subset[subset[region_dim] == country]
                .groupby("year")[[f"{Vars.hw_days}_total"]]
                .sum()
            )
            axs[ix].bar(
                x=country_data.index.astype(str),
                height=country_data[f"{Vars.hw_days}_total"] / 1e9,
                label=country,
                bottom=baseline,
                color=country_color_map[country],
            )
            baseline += country_data[f"{Vars.hw_days}_total"].values / 1e9
            axs[ix].legend()
    axs[0].set_ylabel(f"{Labels.get_label(Vars.hw_days)} (billions)\nUnder 1 year")
    axs[1].set_ylabel(f"{Labels.get_label(Vars.hw_days)} (billions)\nAge 65+")
    plt.xticks(rotation=90, fontsize=10)
    for ax in axs:
        ax.set_xlabel("", visible=False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        ax.set(ylim=(0, None))
    plt.tight_layout()
    sns.despine()
    plt.savefig(DirsLocal.figures / f"{Vars.hw_days}_by_{region_dim}.pdf")
    plt.show()


def plot_change_trends_by(
    data: xr.Dataset,
    region_dim: str,
    num_regions: int,
    baseline_period: str | bytes,
) -> None:
    """Plot change trends by region for a selected baseline period."""
    if Vars.baseline_period in data.dims:
        data = data.sel({Vars.baseline_period: baseline_period})

    df = data.copy().to_dataframe().reset_index()
    f, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 4.5))

    country_color_map = {}
    country_colors = sns.color_palette("tab20" if num_regions >= 5 else "tab10")

    for ix, age_band in enumerate(df[Vars.age_band].unique()):
        subset = df[df[Vars.age_band] == age_band]
        total_by_year = subset.groupby("year")[[CHANGE_TOTAL_VAR]].sum().sort_index()

        x_labels = total_by_year.index.astype(str)
        axs[ix].bar(
            x=x_labels,
            height=total_by_year[CHANGE_TOTAL_VAR].values / 1e9,
            color="gray",
            alpha=0.35,
            label="Global Total",
        )

        top_regions = (
            subset.groupby(region_dim)[[CHANGE_TOTAL_VAR]]
            .sum()
            .abs()
            .sort_values(CHANGE_TOTAL_VAR, ascending=False)
            .head(num_regions)
            .index
        )

        baseline_pos = np.zeros(len(total_by_year), dtype=float)
        baseline_neg = np.zeros(len(total_by_year), dtype=float)

        for ix_country, country in enumerate(top_regions):
            if country not in country_color_map:
                country_color_map[country] = country_colors[
                    ix_country + num_regions * ix
                ]

            country_data = (
                subset[subset[region_dim] == country]
                .groupby("year")[[CHANGE_TOTAL_VAR]]
                .sum()
                .reindex(total_by_year.index, fill_value=0)
            )
            heights = country_data[CHANGE_TOTAL_VAR].values / 1e9
            bottoms = np.where(heights >= 0, baseline_pos, baseline_neg)

            axs[ix].bar(
                x=x_labels,
                height=heights,
                label=country,
                bottom=bottoms,
                color=country_color_map[country],
            )

            baseline_pos = np.where(heights >= 0, baseline_pos + heights, baseline_pos)
            baseline_neg = np.where(heights < 0, baseline_neg + heights, baseline_neg)

        axs[ix].legend()
        axs[ix].axhline(0, color="black", linewidth=0.8, linestyle="--")

    axs[0].set_ylabel(
        f"Change in {Labels.get_label(Vars.hw_days)} (billions)\nUnder 1 year"
    )
    axs[1].set_ylabel(f"Change in {Labels.get_label(Vars.hw_days)} (billions)\nAge 65+")
    plt.xticks(rotation=90, fontsize=10)
    for ax in axs:
        ax.set_xlabel("", visible=False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    baseline_label = (
        baseline_period.decode()
        if isinstance(baseline_period, bytes)
        else str(baseline_period)
    )
    axs[0].set_title(f"Baseline: {baseline_label}")
    plt.tight_layout()
    sns.despine()

    baseline_tag = baseline_label.replace("-", "_")
    plt.savefig(
        DirsLocal.figures
        / f"{Vars.hw_days}_change_by_{region_dim}_baseline_{baseline_tag}.pdf"
    )
    plt.show()


def plot_comparison_aggregates(global_summary, countries, who, hdi, lancet) -> None:
    """Plot a comparison of total heatwave days across all aggregates.
    To make sure that the aggregation and grouping processes are consistent"""

    # add a plot in which we compare the yearly sums for the all the ds
    f, axs = plt.subplots(2, 1, sharex=True)
    for ds, label in zip(
        [global_summary, countries, who, hdi, lancet],
        ["Global Summary", "Country", "WHO Region", "HDI Group", "LC Region"],
    ):
        for age_band in ds[Vars.age_band].values:
            subset = ds.sel({Vars.age_band: age_band})
            subset = subset.reset_coords(drop=True).to_dataframe().reset_index()
            subset = subset.groupby("year").sum()
            if age_band == Vars.infants:
                axs[0].plot(
                    subset.index,
                    subset[f"{Vars.hw_days}_total"],
                    label=f"{label} - {age_band}",
                )
            else:
                axs[1].plot(
                    subset.index,
                    subset[f"{Vars.hw_days}_total"],
                    label=f"{label} - {age_band}",
                )
    axs[0].set_title("Total heatwave person-days exposure - Infants")
    axs[1].set_title("Total heatwave person-days exposure - Age 65+")
    axs[1].set_xlabel("Year")
    axs[0].legend()
    plt.tight_layout()
    sns.despine()
    axs[0].set(ylim=(0, None))
    axs[1].set(ylim=(0, None))
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[1].grid(True, linestyle="--", alpha=0.7)
    # plt.savefig(DirsLocal.figures / "comparison_total_heatwave_days.pdf")
    plt.show()


def plot_and_summary() -> None:
    """Create summary plots and tables for the aggregate results."""
    # import datasets for plotting
    population = load_population_dataset()
    countries = xr.open_dataset(FilesLocal.aggregate_country)
    lancet = xr.open_dataset(FilesLocal.aggregate_lancet)
    who = xr.open_dataset(FilesLocal.aggregate_who)
    hdi = xr.open_dataset(FilesLocal.aggregate_hdi)
    countries_change = xr.open_dataset(FilesLocal.aggregate_country_change)
    lancet_change = xr.open_dataset(FilesLocal.aggregate_lancet_change)
    who_change = xr.open_dataset(FilesLocal.aggregate_who_change)
    hdi_change = xr.open_dataset(FilesLocal.aggregate_hdi_change)

    # Plot trends by region
    plot_trends_by(countries, region_dim="country", num_regions=5)
    plot_trends_by(lancet, region_dim="lc_group", num_regions=3)
    plot_trends_by(who, region_dim="who_region", num_regions=3)
    plot_trends_by(hdi, region_dim="hdi_group", num_regions=3)

    baseline_periods = (
        countries_change[Vars.baseline_period].values
        if Vars.baseline_period in countries_change.dims
        else [
            Vars.format_baseline_period(
                (Vars.year_reference_start, Vars.year_reference_end)
            )
        ]
    )
    for baseline_period in baseline_periods:
        plot_change_trends_by(
            countries_change,
            region_dim="country",
            num_regions=5,
            baseline_period=baseline_period,
        )
        plot_change_trends_by(
            lancet_change,
            region_dim="lc_group",
            num_regions=3,
            baseline_period=baseline_period,
        )
        plot_change_trends_by(
            who_change,
            region_dim="who_region",
            num_regions=3,
            baseline_period=baseline_period,
        )
        plot_change_trends_by(
            hdi_change,
            region_dim="hdi_group",
            num_regions=3,
            baseline_period=baseline_period,
        )

    # create a global summary
    global_summary = xr.open_dataset(FilesLocal.hw_combined_q)
    global_summary = global_summary.groupby("year").sum(
        dim=["latitude", "longitude"], skipna=True
    )
    # merge population and heatwave days to create a global summary table
    global_summary = global_summary.merge(
        population.sum(dim=["latitude", "longitude"], skipna=True)
    )
    # add weighted mean heatwave days per person to the global summary
    global_summary[f"{Vars.hw_days}_per_person"] = (
        global_summary[Vars.hw_days] / global_summary["population"]
    )
    global_summary[f"{Vars.hw_count}_per_person"] = (
        global_summary[Vars.hw_count] / global_summary["population"]
    )
    # rename Vars.hw_count to match the naming convention in the aggregates
    global_summary = global_summary.rename({Vars.hw_count: f"{Vars.hw_count}_total"})
    global_summary = global_summary.rename({Vars.hw_days: f"{Vars.hw_days}_total"})

    # group country-level results by year for global summary
    global_summary_from_country = countries.to_dataframe().reset_index()
    global_summary_from_country = (
        global_summary_from_country.groupby(["year", Vars.age_band])
        .sum()
        .reset_index()
        .drop(columns="country")
        .set_index(["year", Vars.age_band])
    ).to_xarray()

    logging.info("Exporting Excel summary...")
    export_excel(
        datasets={
            "Global Summary": ("year", global_summary),
            "Country": ("country", countries),
            "WHO Region": ("who_region", who),
            "HDI Group": ("hdi_group", hdi),
            "LC Region": ("lc_group", lancet),
        },
        change_datasets={
            "Country Change": ("country", countries_change),
            "WHO Region Change": ("who_region", who_change),
            "HDI Group Change": ("hdi_group", hdi_change),
            "LC Region Change": ("lc_group", lancet_change),
        },
    )

    logging.info("All aggregations completed.")

    plot_comparison_aggregates(global_summary, countries, who, hdi, lancet)


if __name__ == "__main__":
    pass
    # main()
    plot_and_summary()
