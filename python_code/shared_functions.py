import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from my_config import Vars, Dirs


def get_lancet_country_data(hdi_column="hdi_2019_grouping"):
    """
    Loads country polygons and merges with Lancet HDI grouping data.
    """
    country_polygons = gpd.read_file(Dirs.dir_file_country_polygons)

    country_lc_grouping = pd.read_excel(
        Dirs.dir_file_lancet_country_info,
        header=1,
    )

    # Safer merge specifying the "on" column
    country_polygons = country_polygons.merge(
        country_lc_grouping.rename(columns={"ISO3": "ISO_3_CODE"}),
        on="ISO_3_CODE",
        how="left",
    )

    # Use categorical type for correct sorting logic (Low < Medium < High)
    hdi_order = ["Low", "Medium", "High", "Very High"]

    # Create integer mapping
    # Note: We handle NaNs safely
    region_to_id = {val: i + 1 for i, val in enumerate(hdi_order)}

    def map_hdi_safe(val):
        return region_to_id.get(val, np.nan)

    country_polygons["HDI_ID"] = country_polygons[hdi_column].map(map_hdi_safe)

    return country_polygons


def read_pop_data_processed(get_pop_75=False):
    """
    Loads the processed ERA5-grid population data.
    Now uses the new file structure (t_0_..., t_65_...).
    """
    print("Loading Population Time Series...")

    # 1. Infants (<1 year) -> Age string "0"
    pop_inf = xr.open_dataset(Dirs.dir_pop_infants_file).sel(
        year=slice(Vars.year_min_analysis, Vars.year_max_analysis)
    )
    pop_eld = xr.open_dataset(Dirs.dir_pop_elderly_file).sel(
        year=slice(Vars.year_min_analysis, Vars.year_max_analysis)
    )
    pop_75 = xr.open_dataset(Dirs.dir_pop_above_75_file).sel(
        year=slice(Vars.year_min_analysis, Vars.year_max_analysis)
    )

    # Subset to analysis years if needed
    analysis_years = slice(Vars.year_min_analysis, Vars.year_max_analysis)
    pop_inf = pop_inf.sel(year=analysis_years)
    pop_eld = pop_eld.sel(year=analysis_years)
    pop_75 = pop_75.sel(year=analysis_years)

    # For backward compatibility with your main script
    population_worldpop = xr.concat(
        [pop_inf, pop_eld],
        dim=pd.Index([0, 65], name="age_band_lower_bound"),
    )

    if get_pop_75:
        return pop_inf, pop_eld, population_worldpop, pop_75
    else:
        return pop_inf, pop_eld, population_worldpop


def calculate_exposure_population(data, heatwave_metrics, metric="heatwave_days"):
    """
    Calculates exposure: Population * Metric
    Fixes 'AttributeError' by ensuring inputs are DataArrays.
    """
    # 1. Ensure Population Data is a DataArray
    # If it's a Dataset (e.g. opened with open_dataset), convert it.
    if isinstance(data, xr.Dataset):
        # to_array() converts variables to a dimension, .squeeze(drop=True) removes it
        # assuming the dataset contains only the population variable.
        data = data.to_array().squeeze(drop=True)

    # 2. Select the specific metric (Returns a DataArray)
    hw_data = heatwave_metrics[metric]

    # 3. Align Dimensions explicitly to prevent broadcasting errors
    # (year, latitude, longitude)
    hw_data = hw_data.transpose("year", "latitude", "longitude")
    data = data.transpose("year", "latitude", "longitude")

    # 4. Calculate Exposure
    # DataArray * DataArray = DataArray
    exposure = hw_data * data

    # 5. Set Metadata
    exposure.name = metric

    # 6. Drop auxiliary coordinates if they interfere with saving
    if "age_band_lower_bound" in exposure.coords:
        exposure = exposure.drop_vars("age_band_lower_bound")
    if "variable" in exposure.coords:
        exposure = exposure.drop_vars("variable")

    return exposure
