import os

import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr
from icecream import ic
from shapely.geometry import Point
from tqdm import tqdm

from my_config import (
    dir_local,
    dir_pop_era_grid,
    dir_era_daily,
    VarsWorldPop,
    dir_pop_raw,
    dir_file_detailed_boundaries,
)


def process_and_combine_ages(ages, sex, year, directory, era5_grid):
    combined_files = []
    for age in ages:
        age_files = [
            f
            for f in os.listdir(directory)
            if f"_{sex}_{age}_{year}" in f and f.endswith(".tif")
        ]
        combined_files.extend(age_files)
    # Use sum_files to sum the data from the combined files
    summed_data = sum_files(combined_files, directory)

    # Clean and coarsen the summed data
    cleaned_and_coarsened_data = clean_and_coarsen(summed_data)
    pop = cleaned_and_coarsened_data.to_dataframe("pop").reset_index()
    # print(pop["pop"].sum())

    pop["geometry"] = pop.apply(lambda row: Point(row.longitude, row.latitude), axis=1)
    pop = gpd.GeoDataFrame(pop, geometry="geometry")
    pop.set_crs("EPSG:4326", inplace=True)
    pop = pop.to_crs("EPSG:3395")

    pop = pop[pop["pop"] > 0]
    pop_era5_grid = gpd.sjoin_nearest(pop, era5_grid, how="left")
    # print(pop_era5_grid["pop"].sum())

    pop_regrided = (
        pop_era5_grid.groupby(["longitude_right", "latitude_right"])[["pop"]]
        .sum()
        .reset_index()
    )
    # pop_regrided['geometry'] = pop_regrided.apply(lambda row: Point(row.longitude_right, row.latitude_right), axis=1)
    # pop_regrided = gpd.GeoDataFrame(pop_regrided, geometry='geometry')

    return pop_regrided


def clean_and_coarsen(data_array):
    """
    Cleans the data to get read of negative, infinite and nan values, as well as coarsens by suming

    Parameters:
    data_array (xarray.DataArray): The data array to be cleaned.

    Returns:
    xarray.DataArray: The cleaned data array.
    """
    data_array = data_array.where(data_array > 0, 0)
    data_array = data_array.where(data_array < 3.1e8, 0)
    data_array = data_array.where(data_array != -np.inf, 0)
    data_array = data_array.where(data_array != np.inf, 0)
    data_array = data_array.fillna(0)
    data_array = data_array.rename({"y": "latitude", "x": "longitude"})
    # ic(data_array.latitude)
    data_array = data_array.coarsen(latitude=5).sum()
    data_array = data_array.coarsen(longitude=5).sum()
    # ic(data_array.latitude)

    return data_array.fillna(0)


def sum_files(files, directory=""):
    """
    Sums data across a list of files. The summing is useful for the age over 65,
    made out of several age categories

    Parameters:
    files (list): A list of file paths to process.
    directory (str, optional): The directory path if it's not included in the file paths.

    Returns:
    xarray.DataArray: An array containing the summed data across the specified files.
    """
    total_sum = None
    for file in files:
        data_array = rioxarray.open_rasterio(os.path.join(directory, file))

        if total_sum is None:
            total_sum = data_array
        else:
            total_sum += data_array

    return total_sum


def get_era5_grid(year=1980):
    # open on year of era5 to put population data on the same grid
    era5_data = xr.open_dataset(dir_era_daily / f"{year}_temperature_summary.nc")
    era5_data = era5_data.assign_coords(
        longitude=(((era5_data.longitude + 180) % 360) - 180)
    )
    era_grid = era5_data.isel(time=0).to_dataframe().reset_index()
    era_grid["geometry"] = era_grid.apply(
        lambda row: Point(row.longitude, row.latitude), axis=1
    )
    era_grid = gpd.GeoDataFrame(era_grid, geometry="geometry")
    era_grid.set_crs("EPSG:4326", inplace=True)
    era_grid = era_grid[["longitude", "latitude", "geometry"]]

    gdf_countries = gpd.read_file(dir_file_detailed_boundaries)
    era_grid = gpd.sjoin(
        era_grid,
        gdf_countries[["ISO_3_CODE", "geometry"]],
        how="left",
        predicate="within",
    )
    era5_grid_on_land = era_grid[era_grid["ISO_3_CODE"].notna()]
    era_grid_3395 = era5_grid_on_land.to_crs("EPSG:3395")
    era_grid_3395 = era_grid_3395.drop(columns="index_right")
    return era_grid_3395, era_grid


def process_and_save_population_data(ages, year, sex, era5_grid_3395):

    out_path = (
        dir_pop_era_grid / f'{sex}_{"_".join(map(str, ages))}_{year}_era5_compatible.nc'
    )

    if out_path.exists():
        return

    ic(sex, ages, year)

    pop_gridded = process_and_combine_ages(
        ages=ages,
        sex=sex,
        year=year,
        directory=dir_pop_raw,
        era5_grid=era5_grid_3395,
    )
    pop_gridded = pop_gridded.rename(
        columns={"latitude_right": "latitude", "longitude_right": "longitude"}
    )
    pop_gridded = era5_grid_3395.merge(
        pop_gridded[["longitude", "latitude", "pop"]], how="left"
    )
    pivoted_df = pop_gridded.pivot(index="latitude", columns="longitude", values="pop")

    # Convert the pivoted DataFrame to an xarray DataArray
    da = xr.DataArray(pivoted_df, dims=["latitude", "longitude"])

    # Optionally, add the time coordinate (if you have multiple time points, this step will differ)
    da = da.expand_dims(time=[year])

    # Convert to Dataset if you want to add more variables or simply prefer a Dataset structure
    pop_resampled = da.to_dataset(name="pop")
    pop_resampled["longitude"] = xr.where(
        pop_resampled["longitude"] < 0,
        pop_resampled["longitude"] + 360,
        pop_resampled["longitude"],
    )
    pop_resampled = pop_resampled.sortby("longitude")

    pop_resampled.to_netcdf(out_path)


def main():
    ages_array = [[0], [65, 70, 75, 80], [75, 80]]
    years_array = np.arange(
        VarsWorldPop.year_worldpop_start, VarsWorldPop.year_worldpop_end + 1
    )
    total_iterations = (
        len(ages_array) * len(VarsWorldPop.worldpop_sex) * len(years_array)
    )

    era5_grid_3395, era5_grid = get_era5_grid()

    with tqdm(total=total_iterations) as pbar:
        for age in ages_array:
            for year in years_array:
                for sex in VarsWorldPop.worldpop_sex:
                    process_and_save_population_data(
                        ages=age, year=year, sex=sex, era5_grid_3395=era5_grid_3395
                    )
                    pbar.update(1)


if __name__ == "__main__":
    """
    This file process the population data for the years 2000-2020 and saves it in a format compatible with the ERA5 grid
    """
    main()
    # pass
