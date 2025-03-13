import os

import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
from cartopy import crs as ccrs
from geocube.api.core import make_geocube

from my_config import Dirs
from python_code.shared_functions import get_lancet_country_data


def get_era5_data():

    # Open one year of ERA5 data to put population data on the same grid
    era5_data = xr.open_dataset(
        Dirs.dir_era_daily.value / "1980_temperature_summary.nc"
    )

    # Convert longitudes to -180 to 180 range
    era5_data = era5_data.assign_coords(
        longitude=(((era5_data.longitude + 180) % 360) - 180)
    ).sortby("longitude")

    # Assuming the ERA5 data is in WGS84 CRS
    era5_data.rio.set_crs("EPSG:4326", inplace=True)

    # Rename dimensions for consistency
    era5_data = era5_data.rename({"longitude": "x", "latitude": "y"})

    # Select a single time slice, assuming time dimension is present
    era5_data = era5_data.isel(time=0)

    # Set CRS again after renaming dimensions
    era5_data.rio.set_crs("EPSG:4326", inplace=True)

    return era5_data


def get_elderly_data():
    # opening the hybrid population data downloaded from Zenodo
    population_elderly = xr.open_dataarray(Dirs.dir_file_population_before_2000.value)
    population_elderly = population_elderly.sel(age_band_lower_bound=65)
    population_elderly = population_elderly.isel(year=0)
    population_elderly = population_elderly.assign_coords(
        longitude=(((population_elderly.longitude + 180) % 360) - 180)
    ).sortby("longitude")
    return population_elderly


def create_country_raster(country_polygons, era5_data):
    # Create rasterized data using make_geocube
    rasterized_data = make_geocube(
        vector_data=country_polygons, like=era5_data, measurements=["OBJECTID"]
    )
    rasterized_data = rasterized_data.rename({"y": "latitude", "x": "longitude"})

    # Plot world countries
    plot_world_map(rasterized_data["OBJECTID"])

    # Plot t_min worldwide
    plot_world_map(era5_data["t_min"] - 273)

    # get elderly data in china
    country_mask = rasterized_data == 41
    population_elderly = get_elderly_data()
    chn_data = country_mask["OBJECTID"] * population_elderly
    # Plot the rasterized data
    plot_world_map(chn_data)

    if os.path.exists(Dirs.dir_file_country_raster_report.value):
        os.remove(Dirs.dir_file_country_raster_report.value)
    rasterized_data.to_netcdf(Dirs.dir_file_country_raster_report.value)


def create_who_raster(country_polygons, era5_data):
    # WHO regions raster
    region_to_id = {
        region: i
        for i, region in enumerate(country_polygons["WHO_REGION"].unique(), start=1)
    }
    # Apply the mapping to create a new column with numerical identifiers
    country_polygons["WHO_REGION_ID"] = country_polygons["WHO_REGION"].map(region_to_id)

    rasterized_data = make_geocube(
        vector_data=country_polygons, like=era5_data, measurements=["WHO_REGION_ID"]
    )
    rasterized_data = rasterized_data.rename({"y": "latitude", "x": "longitude"})

    # Plot the WHO regions
    plot_world_map(rasterized_data["WHO_REGION_ID"])

    rasterized_data.to_netcdf(Dirs.dir_file_who_raster_report.value)


def create_hdi_raster(country_polygons, era5_data, hdi_column):

    # Create rasterized data using make_geocube
    rasterized_data = make_geocube(
        vector_data=country_polygons, like=era5_data, measurements=["HDI_ID"]
    )
    rasterized_data = rasterized_data.rename({"y": "latitude", "x": "longitude"})

    country_polygons[[hdi_column, "HDI_ID"]].drop_duplicates()

    reg_mask = rasterized_data == 2

    # who2_data = reg_mask["HDI_ID"] * population_elderly
    # who2_data = who2_data.assign_coords(longitude=(((who2_data.longitude + 180) % 360) - 180))

    # Plot the rasterized data
    plot_world_map(rasterized_data["HDI_ID"])

    rasterized_data.to_netcdf(Dirs.dir_file_hdi_raster_report.value)


def create_lancet_raster(country_polygons, era5_data, hdi_column):
    # Lancet Countdown (LC) raster
    region_to_id = {
        region: i
        for i, region in enumerate(country_polygons["LC Grouping"].unique(), start=1)
    }
    # Apply the mapping to create a new column with numerical identifiers
    country_polygons["LC_GROUPING_ID"] = country_polygons["LC Grouping"].map(
        region_to_id
    )

    # Create rasterized data using make_geocube
    rasterized_data = make_geocube(
        vector_data=country_polygons, like=era5_data, measurements=["LC_GROUPING_ID"]
    )
    rasterized_data = rasterized_data.rename({"y": "latitude", "x": "longitude"})

    # reg_mask = rasterized_data == 3
    # who2_data = reg_mask["LC_GROUPING_ID"] * population_elderly
    # who2_data = who2_data.assign_coords(longitude=(((who2_data.longitude + 180) % 360) - 180))

    # Plot the rasterized data
    plot_world_map(rasterized_data["LC_GROUPING_ID"])

    # era5_data = xr.open_dataset(WEATHER_SRC / "era5_0.25deg/daily_temperature_summary/1980_temperature_summary.nc")
    # rasterized_data = rasterized_data.assign_coords(longitude=era5_data.longitude)
    rasterized_data.to_netcdf(Dirs.dir_file_lancet_raster_report.value)


def create_admin1_raster(era5_data):

    admin1_polygons = gpd.read_file(Dirs.dir_file_admin1_polygons)

    # Create rasterized data using make_geocube
    rasterized_data = make_geocube(
        vector_data=admin1_polygons, like=era5_data, measurements=["OBJECTID"]
    )
    rasterized_data = rasterized_data.rename({"y": "latitude", "x": "longitude"})

    # Plot the rasterized data
    plot_world_map(rasterized_data["OBJECTID"])

    # era5_data = xr.open_dataset(WEATHER_SRC / "era5_0.25deg/daily_temperature_summary/1980_temperature_summary.nc")
    # rasterized_data = rasterized_data.assign_coords(longitude=era5_data.longitude)
    rasterized_data.to_netcdf(Dirs.dir_file_admin1_raster_report.value)


def plot_world_map(data, v_min_max=False):
    plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    if v_min_max:
        data.plot(
            ax=ax, transform=ccrs.PlateCarree(), vmin=v_min_max[0], vmax=v_min_max[1]
        )
    else:
        data.plot(ax=ax, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.show()


def main():

    hdi_column = "HDI Group (2023-24)"
    country_polygons = get_lancet_country_data(hdi_column)

    # Assuming country_polygons is in WGS84 CRS
    country_polygons = country_polygons.to_crs("EPSG:4326")

    era5_data = get_era5_data()

    create_country_raster(country_polygons, era5_data)

    create_who_raster(country_polygons, era5_data)

    create_hdi_raster(country_polygons, era5_data, hdi_column)

    create_lancet_raster(country_polygons, era5_data, hdi_column)

    create_admin1_raster(era5_data)


if __name__ == "__main__":
    main()
    pass
