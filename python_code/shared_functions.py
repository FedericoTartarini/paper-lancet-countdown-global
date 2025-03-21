import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from my_config import (
    Vars, Dirs
)


def get_lancet_country_data(hdi_column):

    country_polygons = gpd.read_file(Dirs.dir_file_country_polygons)

    country_lc_grouping = pd.read_excel(
        Dirs.dir_file_lancet_country_info,
        header=1,
    )

    country_polygons = country_polygons.merge(
        country_lc_grouping.rename(columns={"ISO3": "ISO_3_CODE"})
    )

    # Define the custom order for HDI categories
    hdi_order = [np.nan, "Low", "Medium", "High", "Very High"]

    # Create the mapping using the custom order
    region_to_id = {
        region: i + 1
        for i, region in enumerate(hdi_order)
        if region in country_polygons[hdi_column].unique()
    }
    # Apply the mapping to create a new column with numerical identifiers
    country_polygons["HDI_ID"] = country_polygons[hdi_column].map(region_to_id)

    return country_polygons


def read_pop_data_processed(get_pop_75=False):
    pop_inf = xr.open_dataset(Dirs.dir_pop_infants_file.value).sel(
        year=slice(Vars.year_min_analysis, Vars.year_max_analysis)
    )
    pop_eld = xr.open_dataset(Dirs.dir_pop_elderly_file.value).sel(
        year=slice(Vars.year_min_analysis, Vars.year_max_analysis)
    )
    pop_75 = xr.open_dataset(Dirs.dir_pop_above_75_file.value).sel(
        year=slice(Vars.year_min_analysis, Vars.year_max_analysis)
    )

    # I should save this file rather than two separate ones for infants and elderly
    population_worldpop = xr.concat(
        [
            pop_inf.rename({"infants": "pop"}),
            pop_eld.rename({"elderly": "pop"}),
        ],
        dim=pd.Index([0, 65], name="age_band_lower_bound"),
    )

    if get_pop_75:
        return pop_inf, pop_eld, population_worldpop, pop_75
    else:
        return pop_inf, pop_eld, population_worldpop


def calculate_exposure_population(data, heatwave_metrics):
    exposure = heatwave_metrics["heatwaves_days"].transpose(
        "year", "latitude", "longitude"
    ) * data.transpose("year", "latitude", "longitude")

    exposure = exposure.to_array()
    exposure = exposure.squeeze().drop_vars("variable")

    exposure = exposure.rename("heatwaves_days")

    try:
        return exposure.drop_vars("age_band_lower_bound")
    except ValueError:
        return exposure
