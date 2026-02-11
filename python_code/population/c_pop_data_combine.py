"""
This module combines population data from historical sources (before 2000)
and WorldPop datasets (2000-2025) for infants and elderly age groups.

It loads the data, processes it, concatenates the datasets,
and saves the combined data to NetCDF files for further analysis.
"""

import os

import xarray as xr

from my_config import VarsWorldPop, DirsLocal


def load_population_data(age_group, years, suffix="era5_compatible.nc"):
    datasets = []
    for year in years:
        # Load t_{age}_{year} file
        file_path = DirsLocal.pop_e5l_grid / f"t_{age_group}_{year}_{suffix}"
        if file_path.exists():
            ds = xr.open_dataset(file_path)
            # Standardize time dimension to 'year'
            if "time" in ds.dims:
                ds = ds.rename({"time": "year"})
            datasets.append(ds)

    if not datasets:
        raise FileNotFoundError(f"No files found for age group: {age_group}")

    # Concatenate over time
    return xr.concat(datasets, dim="year")  # todo this is returning a warning


def main():
    years_range = range(2000, 2026)  # todo this should not be hardcoded

    infants_worldpop = load_population_data(age_group="under_1", years=years_range)
    elderly_worldpop = load_population_data(age_group="65_over", years=years_range)

    # Load and process data before 2000
    demographics_totals = xr.open_dataarray(DirsLocal.dir_file_population_before_2000)
    demographics_totals = demographics_totals.sel(year=slice(1950, 1999))
    infants_lancet = demographics_totals.sel(age_band_lower_bound=0).sel(
        year=VarsWorldPop.get_slice_years(period="before")
    )
    infants_lancet /= 5  # Divide by 5 to get the number of infants

    elderly_lancet = demographics_totals.sel(age_band_lower_bound=65).sel(
        year=VarsWorldPop.get_slice_years(period="before")
    )

    # Prepare historical datasets
    # Note: We keep variable name as 'pop' to avoid confusion and allow generic handling
    infants_lancet = infants_lancet.to_dataset().rename({"demographic_totals": "pop"})
    elderly_lancet = elderly_lancet.to_dataset().rename({"demographic_totals": "pop"})

    # Combine: simply concatenate historical (pre-2000) and new (2000-2025)
    # No extrapolation needed as we have data up to 2025
    infants_pop_analysis = xr.concat(
        [infants_lancet, infants_worldpop], dim="year"
    ).transpose("year", "latitude", "longitude")  # todo FutureWarning

    elderly_pop_analysis = xr.concat(
        [elderly_lancet, elderly_worldpop], dim="year"
    ).transpose("year", "latitude", "longitude")

    # Save the results to NetCDF files
    if DirsLocal.dir_pop_infants_file.exists():
        os.remove(DirsLocal.dir_pop_infants_file)
    infants_pop_analysis.to_netcdf(DirsLocal.dir_pop_infants_file)

    if DirsLocal.dir_pop_elderly_file.exists():
        os.remove(DirsLocal.dir_pop_elderly_file)
    elderly_pop_analysis.to_netcdf(DirsLocal.dir_pop_elderly_file)

    # elderly above 75 (only available for recent years in this flow)
    elderly_worldpop_75 = load_population_data(age_group="75_over", years=years_range)
    # Just save, no extrapolation
    if DirsLocal.dir_pop_above_75_file.exists():
        os.remove(DirsLocal.dir_pop_above_75_file)
    elderly_worldpop_75.to_netcdf(DirsLocal.dir_pop_above_75_file)


if __name__ == "__main__":
    main()
