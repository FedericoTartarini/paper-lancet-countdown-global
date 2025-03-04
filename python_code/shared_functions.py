import numpy as np
import pandas as pd
import geopandas as gpd

from my_config import dir_file_lancet_country_info, dir_file_country_polygons


def get_lancet_country_data(hdi_column):

    country_polygons = gpd.read_file(dir_file_country_polygons)

    country_lc_grouping = pd.read_excel(
        dir_file_lancet_country_info,
        header=1,
    )

    country_polygons = country_polygons.merge(
        country_lc_grouping.rename(columns={"ISO3": "ISO_3_CODE"})
    )

    # Define the custom order for HDI categories
    hdi_order = [np.nan, 'Low', 'Medium', 'High', 'Very High']

    # Create the mapping using the custom order
    region_to_id = {
        region: i + 1
        for i, region in enumerate(hdi_order)
        if region in country_polygons[hdi_column].unique()
    }
    # Apply the mapping to create a new column with numerical identifiers
    country_polygons["HDI_ID"] = country_polygons[hdi_column].map(region_to_id)

    return country_polygons