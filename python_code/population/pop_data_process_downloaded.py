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
)


def process_and_combine_ages(ages, sex, start_year, directory, era5_grid):
    combined_files = []
    for age in ages:
        age_files = [
            f
            for f in os.listdir(directory)
            if f"_{sex}_{age}_{start_year}" in f and f.endswith(".tif")
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

    gdf_countries = gpd.read_file(
        dir_local / "admin_boundaries" / "Detailed_Boundary_ADM0"
    )
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


def process_and_save_population_data(ages, start_year, sex):

    out_path = (
        dir_pop_era_grid
        / f'{sex}_{"_".join(map(str, ages))}_{start_year}_era5_compatible.nc'
    )

    if out_path.exists():
        return

    ic(sex, ages, start_year)

    era5_grid_3395, era5_grid = get_era5_grid()
    worldpop_dir = dir_local / "population"
    pop_regrided = process_and_combine_ages(
        ages=ages,
        sex=sex,
        start_year=start_year,
        directory=worldpop_dir,
        era5_grid=era5_grid_3395,
    )
    pop_regrided = pop_regrided.rename(
        columns={"latitude_right": "latitude", "longitude_right": "longitude"}
    )
    pop_regrided = era5_grid_3395.merge(
        pop_regrided[["longitude", "latitude", "pop"]], how="left"
    )
    pivoted_df = pop_regrided.pivot(index="latitude", columns="longitude", values="pop")

    # Convert the pivoted DataFrame to an xarray DataArray
    da = xr.DataArray(pivoted_df, dims=["latitude", "longitude"])

    # Optionally, add the time coordinate (if you have multiple time points, this step will differ)
    da = da.expand_dims(time=[start_year])

    # Convert to Dataset if you want to add more variables or simply prefer a Dataset structure
    pop_resampled = da.to_dataset(name="pop")
    pop_resampled["longitude"] = xr.where(
        pop_resampled["longitude"] < 0,
        pop_resampled["longitude"] + 360,
        pop_resampled["longitude"],
    )
    pop_resampled = pop_resampled.sortby("longitude")

    pop_resampled.to_netcdf(out_path)


if __name__ == "__main__":

    ages_array = [[0], [65, 70, 75, 80]]
    sex_array = ["f", "m"]
    start_year_array = np.arange(2000, 2021)
    total_iterations = len(ages_array) * len(sex_array) * len(start_year_array)
    with tqdm(total=total_iterations) as pbar:
        for ages in ages_array:
            for start_year in start_year_array:
                for sex in sex_array:
                    process_and_save_population_data(
                        ages=ages, start_year=start_year, sex=sex
                    )
                    pbar.update(1)
