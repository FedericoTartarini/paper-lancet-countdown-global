import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from my_config import Vars, DirsLocal


def get_lancet_country_data(hdi_column="hdi_2019_grouping"):
    """
    Loads country polygons and merges with Lancet HDI grouping data.
    """
    country_polygons = gpd.read_file(DirsLocal.dir_file_country_polygons)

    country_lc_grouping = pd.read_excel(
        DirsLocal.dir_file_lancet_country_info,
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
