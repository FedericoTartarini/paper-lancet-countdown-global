import dask
import geopandas as gpd
import pandas as pd
import xarray as xr
from tqdm import tqdm

from my_config import (
    year_max_analysis,
    dir_results_pop_exposure,
    year_min_analysis,
    dir_pop_infants_file,
    dir_pop_elderly_file,
    dir_file_elderly_exposure_change,
    dir_file_infants_exposure_change,
    dir_file_lancet_country_info,
    dir_file_country_polygons,
    dir_file_country_raster_report,
    dir_worldpop_exposure_by_region,
    dir_file_who_raster_report,
    dir_file_hdi_raster_report,
    dir_file_lancet_raster_report,
)

population_over_65 = xr.open_dataarray(dir_pop_elderly_file)
population_infants = xr.open_dataarray(dir_pop_infants_file)

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

exposures_over65 = xr.open_dataset(dir_file_elderly_exposure_change)
exposures_infants = xr.open_dataset(dir_file_infants_exposure_change)

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
    / f"heatwave_exposure_multi_threshold_{year_min_analysis}-{year_max_analysis}_worldpop.nc",
    chunks=dict(age_band_lower_bound=1, year=20),
)
exposures_abs = exposures_abs.assign_coords(
    longitude=(((exposures_abs.longitude + 180) % 360) - 180)
).sortby("longitude", ascending=False)

country_lc_grouping = pd.read_excel(
    dir_file_lancet_country_info,
    header=1,
)

countries_raster = xr.open_dataset(dir_file_country_raster_report)


def exposure_weighted_change_by_country():

    country_polygons = gpd.read_file(dir_file_country_polygons)
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
            / f"countries_heatwaves_exposure_weighted_change_1980-{year_max_analysis}_worldpop.nc"
        )


def exposure_total_change_by_country():
    country_polygons = gpd.read_file(dir_file_country_polygons)
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
            / f"countries_heatwaves_exposure_change_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
        )


def exposure_absolute_by_country():
    country_polygons = gpd.read_file(dir_file_country_polygons)
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
        / f"countries_heatwaves_exposure_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
    )


def exposure_absolute_by_who_region():
    country_polygons = gpd.read_file(dir_file_country_polygons)
    # Exposures absolute by WHO region
    region_to_id = {
        region: i
        for i, region in enumerate(country_polygons["WHO_REGION"].unique(), start=1)
    }
    # Apply the mapping to create a new column with numerical identifiers
    country_polygons["WHO_REGION_ID"] = country_polygons["WHO_REGION"].map(region_to_id)

    # Rasterize the WHO regions
    who_region_raster = xr.open_dataset(dir_file_who_raster_report)

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
        / f"who_regions_heatwaves_exposure_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
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
            / f"who_regions_heatwaves_exposure_change_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
        )


def exposure_absolute_by_hdi():
    country_polygons = gpd.read_file(dir_file_country_polygons)
    country_polygons = country_polygons.merge(
        country_lc_grouping.rename(columns={"ISO3": "ISO_3_CODE"})
    )

    hdi_col_name = "HDI Group (2023-24)"

    region_to_id = {
        region: i
        for i, region in enumerate(country_polygons[hdi_col_name].unique(), start=1)
    }

    # Apply the mapping to create a new column with numerical identifiers
    country_polygons["HDI_ID"] = country_polygons[hdi_col_name].map(region_to_id)

    hdi_raster = xr.open_dataset(dir_file_hdi_raster_report)

    hdi = country_polygons[["HDI_ID", hdi_col_name]].drop_duplicates()

    pop = []
    results = []
    results_weight = []

    hdi = hdi[hdi["HDI_ID"] > 1]
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):

        for _, row in tqdm(hdi.iterrows(), total=len(hdi[hdi_col_name])):
            mask = hdi_raster["HDI_ID"] == row.HDI_ID

            masked_population = (
                (mask * population)
                .sum(dim=["latitude", "longitude"])
                .expand_dims(dim={"level_of_human_development": [row[hdi_col_name]]})
                .compute()
            )
            pop.append(masked_population)

            masked_exposures = (
                (exposures_abs * mask)
                .sum(dim=["latitude", "longitude"])
                .expand_dims(dim={"level_of_human_development": [row[hdi_col_name]]})
                .compute()
            )
            results.append(masked_exposures.heatwaves_days)

            masked_exposure_per_person = (
                masked_exposures.heatwaves_days / masked_population
            )
            results_weight.append(masked_exposure_per_person.compute())

        results_pop = xr.concat(pop, dim="level_of_human_development")
        results_pop = results_pop.to_dataset(name="population")

        results_abs = xr.concat(results, dim="level_of_human_development")
        results_abs = results_abs.to_dataset(name="exposures_total")

        results_weight = xr.concat(results_weight, dim="level_of_human_development")
        results_weight = results_weight.to_dataset(name="exposures_weighted")

        exposures_hdi = xr.merge([results_pop, results_abs, results_weight])

    exposures_hdi.to_netcdf(
        exposures_hdi.to_netcdf(
            dir_worldpop_exposure_by_region
            / f"hdi_regions_heatwaves_exposure_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
        )
    )

    print(
        exposures_hdi.sel(
            year=2020, age_band_lower_bound=0, level_of_human_development="Very High"
        ).sum()
    )
    print(
        exposures_hdi.sel(
            year=2020, age_band_lower_bound=0, level_of_human_development="Low"
        ).sum()
    )

    results = []
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):

        for _, row in tqdm(hdi.iterrows(), total=len(hdi[hdi_col_name])):
            mask = hdi_raster["HDI_ID"] == row.HDI_ID

            masked_exposures = (exposures_change * mask).sum(
                dim=["latitude", "longitude"]
            )
            masked_exposures = masked_exposures.expand_dims(
                dim={"level_of_human_development": [row[hdi_col_name]]}
            )
            results.append(masked_exposures)

        results = xr.concat(results, dim="level_of_human_development")
        results.to_netcdf(
            dir_worldpop_exposure_by_region
            / f"hdi_regions_heatwaves_exposure_change_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
        )


def exposure_absolute_by_lc_grouping():
    country_polygons = gpd.read_file(dir_file_country_polygons)
    country_polygons = country_polygons.merge(
        country_lc_grouping.rename(columns={"ISO3": "ISO_3_CODE"})
    )

    region_to_id = {
        region: i
        for i, region in enumerate(country_polygons["LC Grouping"].unique(), start=1)
    }

    # Apply the mapping to create a new column with numerical identifiers
    country_polygons["LC_GROUPING_ID"] = country_polygons["LC Grouping"].map(
        region_to_id
    )

    lc_grouping_raster = xr.open_dataset(dir_file_lancet_raster_report)

    lc_grouping = country_polygons[["LC_GROUPING_ID", "LC Grouping"]].drop_duplicates()

    print(lc_grouping)

    pop = []
    results = []
    results_weight = []

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):

        for _, row in tqdm(
            lc_grouping.iterrows(), total=len(lc_grouping["LC Grouping"])
        ):
            mask = lc_grouping_raster["LC_GROUPING_ID"] == row.LC_GROUPING_ID

            masked_population = (
                (mask * population)
                .sum(dim=["latitude", "longitude"])
                .expand_dims(dim={"lc_group": [row["LC Grouping"]]})
                .compute()
            )
            pop.append(masked_population)

            masked_exposures = (
                (exposures_abs * mask)
                .sum(dim=["latitude", "longitude"])
                .expand_dims(dim={"lc_group": [row["LC Grouping"]]})
                .compute()
            )
            results.append(masked_exposures.heatwaves_days)

            masked_exposure_per_person = (
                masked_exposures.heatwaves_days / masked_population
            )
            results_weight.append(masked_exposure_per_person.compute())

        results_pop = xr.concat(pop, dim="lc_group")
        results_pop = results_pop.to_dataset(name="population")

        results_abs = xr.concat(results, dim="lc_group")
        results_abs = results_abs.to_dataset(name="exposures_total")

        results_weight = xr.concat(results_weight, dim="lc_group")
        results_weight = results_weight.to_dataset(name="exposures_weighted")

    exposures_lc_grouping = xr.merge([results_pop, results_abs, results_weight])

    exposures_lc_grouping.to_netcdf(
        dir_worldpop_exposure_by_region / f"exposures_abs_by_lc_group_worldpop.nc"
    )


if __name__ == "__main__":
    exposure_weighted_change_by_country()  # 3-sec
    exposure_total_change_by_country()  # 2-sec
    exposure_absolute_by_country()  # 2-min
    exposure_absolute_by_who_region()  # 2-sec
    exposure_absolute_by_hdi()  # 2-sec
    exposure_absolute_by_lc_grouping()
