from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt

from cartopy import crs as ccrs
from scipy import stats
from tqdm.notebook import tqdm
import os
import sys

from tqdm import tqdm
import dask

from my_config import (
    dir_pop_hybrid,
    max_year,
    dir_results_pop_exposure,
    min_year,
    path_local,
)

infants_totals_file = (
    dir_pop_hybrid / f"worldpop_infants_1950_{max_year}_era5_compatible.nc"
)
elderly_totals_file = (
    dir_pop_hybrid / f"worldpop_elderly_1950_{max_year}_era5_compatible.nc"
)
population_over_65 = xr.open_dataarray(elderly_totals_file)
population_infants = xr.open_dataarray(infants_totals_file)

population_over_65["age_band_lower_bound"] = 65
population = xr.concat(
    [population_infants, population_over_65], dim="age_band_lower_bound"
)
population.name = "population"
# chunk for parallel
population = population.chunk(dict(age_band_lower_bound=1, year=20))
population = population.assign_coords(
    longitude=(((population.longitude + 180) % 360) - 180)
).sortby("longitude", ascending=False)

exposures_over65 = xr.open_dataset(
    dir_results_pop_exposure
    / f"heatwave_exposure_change_over65_multi_threshold_{min_year}-{max_year}_worldpop.nc"
)
exposures_over65 = exposures_over65.assign_coords(
    longitude=(((exposures_over65.longitude + 180) % 360) - 180)
).sortby("longitude", ascending=False)

exposures_infants = xr.open_dataset(
    dir_results_pop_exposure
    / f"heatwave_exposure_change_infants_multi_threshold_{min_year}-{max_year}_worldpop.nc"
)
exposures_infants = exposures_infants.assign_coords(
    longitude=(((exposures_infants.longitude + 180) % 360) - 180)
).sortby("longitude", ascending=False)

exposures_change = xr.concat(
    [exposures_infants, exposures_over65],
    dim=pd.Index([0, 65], name="age_band_lower_bound"),
)
exposures_change = exposures_change.chunk(dict(age_band_lower_bound=1, year=20))
exposures_change = exposures_change.assign_coords(
    longitude=(((exposures_change.longitude + 180) % 360) - 180)
).sortby("longitude", ascending=False)

exposures_abs = xr.open_dataset(
    dir_results_pop_exposure
    / f"heatwave_exposure_multi_threshold_{min_year}-{max_year}_worldpop.nc",
    chunks=dict(age_band_lower_bound=1, year=20),
)
exposures_abs = exposures_abs.assign_coords(
    longitude=(((exposures_abs.longitude + 180) % 360) - 180)
).sortby("longitude", ascending=False)

country_lc_grouping = pd.read_excel(
    path_local
    / "admin_boundaries"
    / "2025 Global Report Country Names and Groupings.xlsx",
    header=1,
)

country_polygons = gpd.read_file(
    path_local / "admin_boundaries" / "Detailed_Boundary_ADM0" / "GLOBAL_ADM0.shp"
)

countries_raster = xr.open_dataset(
    path_local / "admin_boundaries" / "admin0_raster_report_2024.nc"
)

dir_worldpop_exposure_by_region = (
    dir_results_pop_exposure / "exposure_by_region_or_grouping"
)
dir_worldpop_exposure_by_region.mkdir(parents=True, exist_ok=True)


def exposure_weighted_change_by_country():
    # Calculate Exposure weighted change by country (population normalised)
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        weighted_results = []

        for _, row in tqdm(country_polygons.iterrows(), total=len(country_polygons)):
            grid_code = row.OBJECTID
            country_mask = countries_raster["OBJECTID"] == grid_code
            country_population = (country_mask * population).sum(
                dim=["latitude", "longitude"]
            )
            country_exposures = (country_mask * exposures_change).sum(
                dim=["latitude", "longitude"]
            ) / country_population
            country_exposures = country_exposures.expand_dims(
                dim={"country": [row.ISO_3_CODE]}
            )
            weighted_results.append(country_exposures)

        weighted_results = xr.concat(weighted_results, dim="country")
        weighted_results.to_netcdf(
            dir_worldpop_exposure_by_region
            / f"countries_heatwaves_exposure_weighted_change_1980-{max_year}_worldpop.nc"
        )


def exposure_total_change_by_country():
    # Exposure to change by country, total
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        results_tot = []

        for _, row in tqdm(country_polygons.iterrows(), total=len(country_polygons)):
            grid_code = row.OBJECTID
            country_mask = countries_raster["OBJECTID"] == grid_code
            country_population = (country_mask * population).sum(
                dim=["latitude", "longitude"]
            )
            country_exposures = (country_mask * exposures_change).sum(
                dim=["latitude", "longitude"]
            )
            country_exposures = country_exposures.expand_dims(
                dim={"country": [row.ISO_3_CODE]}
            )
            results_tot.append(country_exposures)

        results_tot = xr.concat(results_tot, dim="country")
        results_tot.to_netcdf(
            dir_worldpop_exposure_by_region
            / f"countries_heatwaves_exposure_change_{min_year}-{max_year}_worldpop.nc"
        )


def exposure_absolute_by_country():
    # Exposures absolute by country
    pop = []
    results = []
    results_weight = []

    for _, row in tqdm(country_polygons.iterrows(), total=len(country_polygons)):
        grid_code = row.OBJECTID
        country_mask = countries_raster["OBJECTID"] == grid_code

        country_population = (
            (country_mask * population)
            .sum(dim=["latitude", "longitude"])
            .expand_dims(dim={"country": [row.ISO_3_CODE]})
            .compute()
        )
        pop.append(country_population)

        country_exposures = (
            (exposures_abs * country_mask)
            .sum(dim=["latitude", "longitude"])
            .expand_dims(dim={"country": [row.ISO_3_CODE]})
            .compute()
        )
        results.append(country_exposures.heatwaves_days)

        country_exposure_per_person = (
            country_exposures.heatwaves_days / country_population
        )
        results_weight.append(country_exposure_per_person.compute())

    results_pop = xr.concat(pop, dim="country")
    results_pop = results_pop.to_dataset(name="population")

    results_abs = xr.concat(results, dim="country")
    results_abs = results_abs.to_dataset(name="exposures_total")

    results_weight = xr.concat(results_weight, dim="country")
    results_weight = results_weight.to_dataset(name="exposures_weighted")

    exposures_countries = xr.merge([results_pop, results_abs, results_weight])

    exposures_countries.to_netcdf(
        dir_worldpop_exposure_by_region
        / f"countries_heatwaves_exposure_{min_year}-{max_year}_worldpop.nc"
    )


def exposure_absolute_by_who_region():
    # Exposures absolute by WHO region
    region_to_id = {
        region: i
        for i, region in enumerate(country_polygons["WHO_REGION"].unique(), start=1)
    }
    # Apply the mapping to create a new column with numerical identifiers
    country_polygons["WHO_REGION_ID"] = country_polygons["WHO_REGION"].map(region_to_id)

    # Rasterize the WHO regions
    who_region_raster = xr.open_dataset(
        path_local / "admin_boundaries" / "WHO_regions_raster_report_2024.nc"
    )

    who_regions = country_polygons[["WHO_REGION", "WHO_REGION_ID"]]
    who_regions = who_regions.drop_duplicates()

    print(who_regions)

    pop = []
    results = []
    results_weight = []

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):

        for _, row in tqdm(who_regions.iterrows(), total=len(who_regions.WHO_REGION)):
            mask = who_region_raster["WHO_REGION_ID"] == row.WHO_REGION_ID

            masked_population = (
                (mask * population)
                .sum(dim=["latitude", "longitude"])
                .expand_dims(dim={"who_region": [row.WHO_REGION]})
                .compute()
            )
            pop.append(masked_population)

            masked_exposures = (
                (exposures_abs * mask)
                .sum(dim=["latitude", "longitude"])
                .expand_dims(dim={"who_region": [row.WHO_REGION]})
                .compute()
            )
            results.append(masked_exposures.heatwaves_days)

            masked_exposure_per_person = (
                masked_exposures.heatwaves_days / masked_population
            )
            results_weight.append(masked_exposure_per_person.compute())

        results_pop = xr.concat(pop, dim="who_region")
        results_pop = results_pop.to_dataset(name="population")

        results_abs = xr.concat(results, dim="who_region")
        results_abs = results_abs.to_dataset(name="exposures_total")

        results_weight = xr.concat(results_weight, dim="who_region")
        results_weight = results_weight.to_dataset(name="exposures_weighted")

        exposures_who = xr.merge([results_pop, results_abs, results_weight])

    exposures_who.to_netcdf(
        dir_worldpop_exposure_by_region
        / f"who_regions_heatwaves_exposure_{min_year}-{max_year}_worldpop.nc"
    )
    print(exposures_who.sel(year=2020).population.sum())

    results = []
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):

        for _, row in tqdm(who_regions.iterrows(), total=len(who_regions.WHO_REGION)):
            mask = who_region_raster == row.WHO_REGION_ID

            masked_exposures = (exposures_change * mask).sum(
                dim=["latitude", "longitude"]
            )
            masked_exposures = masked_exposures.expand_dims(
                dim={"who_region": [row.WHO_REGION]}
            )
            results.append(masked_exposures)

        results = xr.concat(results, dim="who_region")
        results.to_netcdf(
            dir_worldpop_exposure_by_region
            / f"who_regions_heatwaves_exposure_change_{min_year}-{max_year}_worldpop.nc"
        )


if __name__ == "__main___":
    exposure_weighted_change_by_country()  # working 3-sec
    exposure_total_change_by_country()  # working 2-sec
    exposure_absolute_by_country()  # working 2-min
    exposure_absolute_by_who_region()
    # # todo implement
    # - exposure by HDI
    # - Exposure to change weighted by LC Grouping
